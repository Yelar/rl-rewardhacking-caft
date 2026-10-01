"""Independently audit shared prefixes between auxiliary and core fixed caches.

CPU-only; no generated-code execution, model loading, or activation fitting.
Every auxiliary record is compared with every core peer for its problem, for
both models. Numerical failures are retained before the final failed verdict.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

from . import fixed_cache as cache


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def safe_file(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact-relative path')
    result = root / relative
    require(result.is_file() and not result.is_symlink() and result.resolve().is_relative_to(root.resolve()),
            'Missing, symlink, or escaped artifact')
    return result


def load_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Package:
    def __init__(self, root, digest, role):
        self.root = Path(root).resolve(strict=True)
        require(re.fullmatch('[0-9a-f]{64}', digest or '') is not None, 'Exact package manifest SHA256 is required')
        self.manifest_path = safe_file(self.root, 'artifact_manifest.json')
        require(sha(self.manifest_path) == digest, 'Package manifest hash mismatch')
        self.manifest_digest = digest
        self.files = json.loads(self.manifest_path.read_text())['files']
        self.verified = {}
        self.rows = cache.validate_groups(load_jsonl(self.verify('input/prepared_records.jsonl')), role)
        self.summary = json.loads(self.verify('extraction_summary.json').read_text())
        require(self.summary.get('status') == 'succeeded' and self.summary.get('records') == len(self.rows)
                and self.summary.get('layers') == cache.LAYERS and self.summary.get('hidden_size') == cache.HIDDEN
                and self.summary.get('raw_activations_retained') is True, 'Incomplete or incompatible fixed cache package')
        self.source_digest = self.summary['manifest_sha256']
        require(re.fullmatch('[0-9a-f]{64}', self.source_digest or '') is not None, 'Missing extraction manifest identity')
        index = load_jsonl(self.verify('activation_index.jsonl'))
        self.index = {row['record_id']: row for row in index}
        require(len(self.index) == len(index) and set(self.index) == {r['record_id'] for r in self.rows},
                'Raw index does not exactly cover prepared records')
        self.groups = defaultdict(list)
        split_by_problem = {}
        for row in self.rows:
            problem = str(row['problem_id_key'])
            self.groups[problem].append(row)
            require(split_by_problem.setdefault(problem, row['problem_split']) == row['problem_split'], 'Problem crosses splits')
            entry = self.index[row['record_id']]
            require(entry.get('verified') is True and entry['record_index'] == row['record_index']
                    and set(entry.get('models', {})) == {'h0', 'h60'}, 'Unverified or misaligned raw index row')
        if role == 'auxiliary':
            require(all(row['problem_split'] in ('direction_fit', 'configuration_validation') for row in self.rows),
                    'Preselection auxiliary audit cannot open untouched-test activations')

    def verify(self, name):
        path = safe_file(self.root, name)
        require(name in self.files, 'Unmanifested artifact')
        state = identity(path)
        if name in self.verified:
            require(self.verified[name] == state, 'Artifact changed after independent hash verification')
        else:
            bound = self.files[name]
            require(path.stat().st_size == bound['size_bytes'] and sha(path) == bound['sha256'], 'Artifact hash/size mismatch')
            require(identity(path) == state, 'Artifact changed while hashing')
            self.verified[name] = state
        return path

    def prefix(self, row, kind, count):
        """Return only prompt-final and shared-prefix values, never code contexts."""
        import torch
        from safetensors import safe_open
        require(0 <= count <= row['completion_token_count'], 'Requested shared prefix is outside completion')
        entry = self.index[row['record_id']]['models'][kind]
        name = entry['tensor_path']
        require(self.files.get(name, {}).get('sha256') == entry['sha256'] and
                self.files.get(name, {}).get('size_bytes') == entry['size_bytes'], 'Native index/manifest mismatch')
        path = self.verify(name)
        state = identity(path)
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            meta = handle.metadata()
            expected = cache.metadata(row, kind, self.source_digest)
            require(all(meta.get(key) == value for key, value in expected.items()), 'Fixed-cache native metadata/profile mismatch')
            view = handle.get_slice(kind)
            require(view.get_shape() == [cache.LAYERS, row['completion_token_count'], cache.HIDDEN] and view.get_dtype() == 'BF16',
                    'Native completion shape or dtype differs')
            prompt = handle.get_tensor('prompt_final')
            require(prompt.dtype == torch.bfloat16 and tuple(prompt.shape) == (cache.LAYERS, cache.HIDDEN),
                    'Prompt-final shape or dtype differs')
            prompt_position = handle.get_tensor('prompt_final_sequence_position')
            require(prompt_position.dtype == torch.int32 and prompt_position.ndim == 0 and
                    prompt_position.item() == row['prompt_token_count'] - 1, 'Prompt-final position is not p-1')
            ids = handle.get_tensor('input_ids')
            positions = handle.get_tensor('sequence_positions')
            require(ids.dtype == torch.int32 and ids.tolist() == row['input_ids'], 'Saved original input IDs differ')
            require(positions.dtype == torch.int32 and positions.tolist() == row['selected_token_positions'],
                    'Completion axis has shifted relative to original sequence')
            padded = handle.get_tensor('padded_input_ids')
            attention = handle.get_tensor('model_attention_mask')
            model_positions = handle.get_tensor('model_position_ids')
            length = len(row['input_ids'])
            require(padded.dtype == torch.int32 and padded.tolist() == row['input_ids'] + [cache.PAD_ID] * (cache.PADDED_LENGTH - length),
                    'Padded forward IDs differ from reviewed right-padding')
            require(attention.dtype == torch.bool and attention.tolist() == [True] * length + [False] * (cache.PADDED_LENGTH - length),
                    'Padded future attention mask is not zero')
            require(model_positions.dtype == torch.int32 and model_positions.tolist() == list(range(cache.PADDED_LENGTH)),
                    'Forward position IDs differ from reviewed arange')
            result = torch.cat([prompt[:, None], view[:, :count]], dim=1) if count else prompt[:, None]
            require(bool(torch.isfinite(result).all()), 'Nonfinite native prefix')
        require(identity(path) == state, 'Native artifact changed while reading')
        return result


def pair_plan(core, auxiliary, expected_auxiliary_records=51, expected_core_records=561, expected_core_problems=187):
    require(len(core.rows) == expected_core_records and len(core.groups) == expected_core_problems, 'Expected complete 187 core triplets')
    require(len(auxiliary.rows) == expected_auxiliary_records, 'Auxiliary record coverage differs from frozen selection')
    require(not ({r['record_id'] for r in core.rows} & {r['record_id'] for r in auxiliary.rows}), 'Core and auxiliary record IDs overlap')
    pairs = []
    for row in auxiliary.rows:
        peers = core.groups.get(str(row['problem_id_key']), [])
        require(len(peers) == 3, 'Auxiliary record lacks all three core peers')
        for peer in peers:
            require(row['problem_split'] == peer['problem_split'] and row['prompt_token_ids'] == peer['prompt_token_ids'],
                    'Auxiliary/core split or exact prompt IDs differ')
            require(row['model_revision'] == peer['model_revision'] and row['checkpoint_sha256'] == peer['checkpoint_sha256'],
                    'Auxiliary/core model provenance differs')
            pairs.append((row, peer, cache.common_prefix_count(row['completion_token_ids'], peer['completion_token_ids'])))
    return pairs


def audit(core_root, auxiliary_root, core_manifest_sha256, auxiliary_manifest_sha256, output):
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'Cross-cache audit must remain CPU-only')
    output = Path(output)
    require(not output.exists(), 'Audit output must be a fresh directory')
    core = Package(core_root, core_manifest_sha256, 'core')
    auxiliary = Package(auxiliary_root, auxiliary_manifest_sha256, 'auxiliary')
    require(core.root != auxiliary.root, 'Core and auxiliary packages must be separate')
    require(not output.resolve().is_relative_to(core.root) and not output.resolve().is_relative_to(auxiliary.root),
            'Audit outputs cannot mutate an input package')
    pairs = pair_plan(core, auxiliary)
    output.mkdir(parents=True, exist_ok=False)
    cache.write_json(output/'inputs.json', {'core_manifest_sha256':core_manifest_sha256,
        'auxiliary_manifest_sha256':auxiliary_manifest_sha256, 'core_extraction_manifest_sha256':core.source_digest,
        'auxiliary_extraction_manifest_sha256':auxiliary.source_digest, 'auxiliary_records':len(auxiliary.rows),
        'expected_pairs_both_models':len(pairs)*2, 'read_scope':'Only auxiliary fit/validation problems and their core peers; no test activations.'})
    failures, observed = [], 0
    try:
        for row, peer, count in pairs:
            for kind in ('h0', 'h60'):
                a, b = auxiliary.prefix(row, kind, count), core.prefix(peer, kind, count)
                prompt = cache.equality_report(a[:, :1], b[:, :1])
                completion = cache.equality_report(a[:, 1:], b[:, 1:]) if count else None
                result = {'auxiliary_record_id':row['record_id'], 'core_record_id':peer['record_id'],
                          'problem_id':row['problem_id_key'], 'problem_split':row['problem_split'], 'model':kind,
                          'common_completion_tokens':count, 'prompt_final':prompt, 'shared_completion':completion}
                cache.append_json(output/'comparisons.jsonl', result)
                observed += 1
                if not prompt['bitwise_equal'] or completion is not None and not completion['bitwise_equal']:
                    failures.append({'auxiliary_record_id':row['record_id'], 'core_record_id':peer['record_id'], 'model':kind})
        require(observed == len(pairs)*2, 'Comparison coverage incomplete')
    except Exception as error:
        cache.write_json(output/'execution_error.json', {'type':type(error).__name__, 'message':str(error), 'completed_comparisons':observed})
        raise
    report = {'status':'verified' if not failures else 'failed', 'core_records':len(core.rows),
              'auxiliary_records':len(auxiliary.rows), 'auxiliary_problems':len(auxiliary.groups),
              'expected_comparisons':len(pairs)*2, 'completed_comparisons':observed,
              'bitwise_failures':failures, 'all_shared_prefixes_bitwise_equal':not failures,
              'core_verified_files':len(core.verified), 'auxiliary_verified_files':len(auxiliary.verified),
              'no_test_activations_opened':True, 'cuda_initialized':torch.cuda.is_initialized()}
    cache.write_json(output/'audit.json', report)
    files = {p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}
    cache.write_json(output/'artifact_manifest.json', {'algorithm':'sha256','files':files})
    require(not failures, 'Auxiliary/core prefix invariance failed; full comparisons and failed verdict retained')
    return {**report, 'artifact_manifest_sha256':sha(output/'artifact_manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('core-root', 'auxiliary-root', 'core-manifest-sha256', 'auxiliary-manifest-sha256', 'output'):
        parser.add_argument('--'+key, required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), sort_keys=True))


if __name__ == '__main__':
    main()
