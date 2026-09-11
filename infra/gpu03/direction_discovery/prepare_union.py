"""Freeze exact core+auxiliary prepared bytes and original paired local selection.

CPU-only metadata operation: no activations, model inference or code execution.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

CORE_SHA = 'e2ce9ee0b23ebd44b304a57d9400b38d0fbc348cb99f84311f8faf32031ef65e'
AUX_SHA = 'd8f1bc40965b4ae63e6d0f8622dfb65c3c3aea0abb5b61ee28c3cf38efea3cc5'
PAIRS_SHA = '532fc77f9ecbe5b376242d1ab9e47807e8b2783971e863a459e8c49d8fd96eb4'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def digest(data): return hashlib.sha256(data).hexdigest()

def canonical(value): return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def checked_bytes(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or symlinked input')
    data = path.read_bytes()
    require(digest(data) == expected, 'Input hash changed: ' + str(path))
    return data


def inspect_rows(core_bytes, aux_bytes, pair_bytes, *, production=True):
    require(core_bytes.endswith(b'\n') and aux_bytes.endswith(b'\n'), 'Exact concatenation requires final newlines')
    core = [json.loads(line) for line in core_bytes.splitlines()]
    aux = [json.loads(line) for line in aux_bytes.splitlines()]
    rows = core + aux
    require(len({r['record_id'] for r in rows}) == len(rows), 'Duplicate core/auxiliary record IDs')
    require([r['record_index'] for r in rows] == list(range(len(rows))), 'Original core/auxiliary indices are not contiguous')
    by_id = {r['record_id']:r for r in rows}
    core_ids, aux_ids = {r['record_id'] for r in core}, {r['record_id'] for r in aux}
    splits = {}
    for row in rows:
        problem = str(row['problem_id'])
        require(problem not in splits or splits[problem] == row['problem_split'], 'Problem crosses source splits')
        splits[problem] = row['problem_split']
        p,c = row['prompt_token_count'],row['completion_token_count']
        require(type(p) is int and type(c) is int and 0 < p <= 1536 and 0 < c <= 1536 and
                len(row['prompt_token_ids']) == p and len(row['completion_token_ids']) == c and
                row['input_ids'] == row['prompt_token_ids'] + row['completion_token_ids'], 'Original token sequence is inconsistent')
        evaluator = row['regions']['evaluator']
        require(evaluator is not None and type(evaluator['first_executable_completion_token']) is int and
                0 < evaluator['first_executable_completion_token'] < c, 'Missing or invalid existing evaluator-body anchor')
    require(all(r['problem_split'] != 'untouched_test' for r in aux), 'Auxiliary source includes untouched test')
    manifest = json.loads(pair_bytes)
    require(manifest['original_labels_modified'] is False and manifest['test_records_selected'] == 0 and
            manifest['new_generations'] == 0 and manifest['generated_code_executed'] is False, 'Pair provenance changed')
    pairs = manifest['pairs']
    require(len(pairs) == len(aux) and {p['record_id'] for p in pairs} == aux_ids, 'Pair manifest does not exactly cover auxiliary rows')
    for pair in pairs:
        harmful = by_id[pair['record_id']]
        control_id = pair['paired_cached_control_record_id']
        require(control_id in core_ids, 'Paired control is not an existing core row')
        benign = by_id[control_id]
        require(pair['solution_correctness_equal'] is True and pair['prompt_checkpoint_sampling_equal'] is True and
                pair['unchanged_original_record'] is True, 'Pair provenance does not prove comparability')
        require(str(harmful['problem_id']) == str(benign['problem_id']) == str(pair['problem_id']) and
                harmful['problem_split'] == benign['problem_split'] == pair['problem_split'], 'Pair crosses a problem or split')
        require(harmful['is_test_modification_harmful'] is True and benign['is_test_modification_harmful'] is False and
                harmful['ground_truth_correctness'] is benign['ground_truth_correctness'] and
                type(harmful['ground_truth_correctness']) is bool, 'Harmfulness or correctness match changed')
        for field in ('prompt','prompt_token_ids','prompt_sha256','checkpoint_sha256','sampling_sha256'):
            require(harmful[field] == benign[field], 'Paired prompt/checkpoint/sampling changed')
        require(harmful['completion_sha256'] == pair['completion_sha256'] and
                benign['completion_sha256'] == pair['paired_cached_control_completion_sha256'] and
                benign['outcome_presence_class'] == pair['paired_cached_control_class'], 'Pair completion/class identity changed')
        for row in (harmful,benign):
            require(any(isinstance(node,ast.Assert) for node in ast.walk(ast.parse(row['generated_evaluator_function_source']))),
                    'Preselected paired evaluator no longer contains an assertion AST node')
    val_pairs = [p for p in pairs if p['problem_split'] == 'configuration_validation']
    harmful_ids = [p['record_id'] for p in val_pairs]
    control_ids = list(dict.fromkeys(p['paired_cached_control_record_id'] for p in val_pairs))
    selected = harmful_ids + control_ids
    require(len(set(selected)) == len(selected), 'Auxiliary harmful/control IDs overlap')
    report = {'original_core_records':len(core), 'auxiliary_records':len(aux), 'combined_records':len(rows),
        'combined_problems':len(splits), 'original_core_split_records':dict(Counter(r['problem_split'] for r in core)),
        'auxiliary_split_records':dict(Counter(r['problem_split'] for r in aux)),
        'all_ids_unique':True,'source_bytes_concatenated_exactly':True,'original_indices_labels_splits_tokens_and_anchors_unchanged':True,
        'new_generations':0,'model_forwards':0,'generated_code_executed':False,'test_activation_files_opened':0}
    selection = {'schema_version':1,'purpose':'assertion_present_correctness_matched_auxiliary_validation_local_prefixes',
        'evaluation_partition':'configuration_validation','scope':'local','sample_index':0,
        'selection_rule':'All22 validation auxiliary harmful records from the earlier frozen51-pair manifest; exact existing same-problem same-correctness benign controls, unique by record_id in first-pair order.',
        'pair_manifest_sha256':digest(pair_bytes),'combined_prepared_sha256':digest(core_bytes+aux_bytes),
        'harmful_record_ids':harmful_ids,'unique_control_record_ids':control_ids,'selected_record_ids':selected,'pairs':val_pairs,
        'counts':{'harmful_records':len(harmful_ids),'unique_control_records':len(control_ids),'selected_records':len(selected),
                  'pair_count':len(val_pairs),'unique_problems':len({p['problem_id'] for p in val_pairs}),
                  'harmful_groups':dict(Counter(p['auxiliary_group'] for p in val_pairs))},
        'conditions_not_yet_selected':True,'no_activation_or_behavior_outcomes_used':True,'no_test_records_selected':True}
    if production:
        require((len(core),len(aux),len(rows),len(splits)) == (561,51,612,187), 'Frozen dataset sizes changed')
        require(report['original_core_split_records'] == {'direction_fit':351,'configuration_validation':111,'untouched_test':99} and
                report['auxiliary_split_records'] == {'direction_fit':29,'configuration_validation':22}, 'Frozen source split counts changed')
        require(selection['counts'] == {'harmful_records':22,'unique_control_records':22,'selected_records':44,'pair_count':22,
            'unique_problems':18,'harmful_groups':{'correct_harmful':11,'validation_strict_assert':11}}, 'Validation pair inventory changed')
    return report,selection


def freeze(core, auxiliary, pairs, output):
    output = Path(output); require(not output.exists(),'Preserve existing immutable union package')
    cb,ab,pb = checked_bytes(core,CORE_SHA),checked_bytes(auxiliary,AUX_SHA),checked_bytes(pairs,PAIRS_SHA)
    report,selection = inspect_rows(cb,ab,pb)
    output.mkdir(parents=True,mode=0o700)
    (output/'input').mkdir()
    files = {'input/core_prepared_records.jsonl':cb,'input/auxiliary_prepared_records.jsonl':ab,
             'input/auxiliary_selected_manifest.json':pb,'prepared_records.jsonl':cb+ab,
             'source/prepare_union.py':Path(__file__).read_bytes(),
             'validation_auxiliary_selection.json':canonical(selection),'union_audit.json':canonical(report)}
    for name,data in files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);path.chmod(0o400)
    source = {'core':{'path':str(Path(core).resolve()),'sha256':CORE_SHA},
              'auxiliary':{'path':str(Path(auxiliary).resolve()),'sha256':AUX_SHA},
              'pairs':{'path':str(Path(pairs).resolve()),'sha256':PAIRS_SHA},
              'builder_sha256':digest(Path(__file__).read_bytes())}
    (output/'source_provenance.json').write_bytes(canonical(source));(output/'source_provenance.json').chmod(0o400)
    paths = sorted(p for p in output.rglob('*') if p.is_file())
    manifest = {'algorithm':'sha256','files':{str(p.relative_to(output)):{'sha256':digest(p.read_bytes()),'size_bytes':p.stat().st_size} for p in paths}}
    (output/'artifact_manifest.json').write_bytes(canonical(manifest));(output/'artifact_manifest.json').chmod(0o400)
    return {'status':'frozen_pending_independent_verification','output':str(output),'prepared_records_sha256':digest(cb+ab),
            'artifact_manifest_sha256':digest((output/'artifact_manifest.json').read_bytes()),'selection':selection['counts']}


def verify(output):
    output=Path(output);manifest=json.loads((output/'artifact_manifest.json').read_bytes())
    paths={str(p.relative_to(output)):p for p in output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    require(set(paths)==set(manifest['files']) and manifest['algorithm']=='sha256','Union package inventory changed')
    for name,path in paths.items():
        require(not path.is_symlink() and not path.stat().st_mode & 0o222 and
                path.stat().st_size==manifest['files'][name]['size_bytes'] and digest(path.read_bytes())==manifest['files'][name]['sha256'],
                'Union package hash/size/readonly verification failed')
    cb=checked_bytes(output/'input/core_prepared_records.jsonl',CORE_SHA)
    ab=checked_bytes(output/'input/auxiliary_prepared_records.jsonl',AUX_SHA)
    pb=checked_bytes(output/'input/auxiliary_selected_manifest.json',PAIRS_SHA)
    require((output/'prepared_records.jsonl').read_bytes()==cb+ab,'Union is not exact source byte concatenation')
    report,selection=inspect_rows(cb,ab,pb)
    require(json.loads((output/'union_audit.json').read_bytes())==report and
            json.loads((output/'validation_auxiliary_selection.json').read_bytes())==selection,'Recomputed union audit differs')
    source=json.loads((output/'source_provenance.json').read_bytes())
    require(source['builder_sha256']==digest((output/'source/prepare_union.py').read_bytes()),'Packaged builder source identity changed')
    for kind,expected in (('core',CORE_SHA),('auxiliary',AUX_SHA),('pairs',PAIRS_SHA)):
        require(source[kind]['sha256']==expected,'Original source identity changed')
        checked_bytes(source[kind]['path'],expected)
    return {'status':'independently_verified','output':str(output),'prepared_records_sha256':digest(cb+ab),
        'artifact_manifest_sha256':digest((output/'artifact_manifest.json').read_bytes()),'records':len(cb.splitlines())+len(ab.splitlines()),
        'selection':selection['counts'],'source_bytes_indices_splits_anchors_verified':True,'no_gpu_or_evaluation':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    for key in ('core','auxiliary','pairs'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.output) if a.verify else freeze(a.core,a.auxiliary,a.pairs,a.output),sort_keys=True))
if __name__=='__main__':main()
