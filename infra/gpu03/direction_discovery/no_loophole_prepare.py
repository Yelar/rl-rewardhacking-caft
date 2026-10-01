"""CPU-only token/Q preparation for the matched 37-problem no-hint audit.

Nothing is executed at import. The reviewed spec and pinned runtime are required
before tokenizer or tensor imports. Generated completions are provenance bytes;
this module never evaluates them, loads a model, plans requests, or launches GPUs.
Run as a module from an immutable source snapshot. The separate verification
operation requires an external observer's actual producer exit/release receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from infra.gpu03.direction_discovery import no_loophole_plan as plan

VERSIONS = {"peft": "0.17.1", "safetensors": "0.6.2", "tokenizers": "0.22.1",
            "torch": "2.8.0+cu128", "transformers": "4.57.1", "vllm": "0.11.0"}
SOURCE_PROOF_SHA = "4d5de0194258911bd99886fbf6603685e6e94c9fa30b9d9887395b7d91f834e7"
MODULE = "infra.gpu03.direction_discovery.no_loophole_prepare"
HERE = "infra/gpu03/direction_discovery/no_loophole_prepare.py"
PLANNER = "infra/gpu03/direction_discovery/no_loophole_plan.py"
AUDIT_FILES = ("canonical_nohint37.jsonl", "original_carriers37.jsonl", "prompt_pairs.jsonl",
               "alignment_audit.json", "request_alignment.json", "source_bindings.json",
               "tokenizer_policy.pending.json")
require = plan.require


def h(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path = Path(path)
    if not isinstance(data, bytes):
        data = (plan.canonical(data) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o400)
    return plan.ref(path)


def content_ref(ref):
    """Pinned caches/python may be symlinks; validate exact content, not mode."""
    path = plan.get_ref(ref)
    before = path.stat()
    require(path.is_file() and before.st_size <= 64 << 20, "Unbounded runtime/tokenizer file")
    data = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Runtime input changed during read")
    require(h(data) == ref["sha256"] and len(data) == ref["size_bytes"], "Pinned runtime/tokenizer content differs")
    return data


def check_runtime(spec):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and
            os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Preparation requires explicit CPU-only offline environment")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        require(os.environ.get(key) == ("false" if key == "TOKENIZERS_PARALLELISM" else "1"), "Unreviewed CPU thread setting")
    require(spec["runtime_versions"] == VERSIONS and
            {k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS, "Pinned package versions differ")
    require(list(sys.version_info[:3]) == [3, 12, 3] and spec["python_binary"]["path"] == sys.executable,
            "Pinned Python executable/version differs")
    content_ref(spec["python_binary"])


def check_sources(spec):
    root = Path(spec["source_root"])
    require(root.is_absolute() and root == Path(__file__).resolve().parents[3], "Preparation source root differs")
    require(Path(plan.__file__).resolve() == root / PLANNER, "Imported planner is outside the bound source snapshot")
    sources = spec["source_files"]
    require(isinstance(sources, dict) and {HERE, PLANNER, *plan.SOURCES} <= set(sources), "Incomplete preparation source bindings")
    actual = set()
    for directory in (root / "src", root / "infra"):
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                actual.add(str(path.relative_to(root)))
    require(actual == set(sources), "Source inventory omits or adds staged files")
    for relative, ref in sources.items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Invalid source path")
        require(set(ref) == {"sha256", "size_bytes"}, "Invalid source inventory entry")
        plan.read_ref({"path": str(root / relative), **ref})
    require(all(sources[k]["sha256"] == v for k, v in plan.SOURCES.items()), "Historical projection/engine changed")
    require(spec["historical_source_qualification"]["sha256"] == SOURCE_PROOF_SHA, "Wrong original source qualification")
    proof = plan.load_ref(spec["historical_source_qualification"])
    require(proof.get("status") == "independently_verified_remote_generation_source" and
            proof.get("qualified_jobs_released") is True and proof.get("producer_exit_status") == 0 and
            all(proof["source_inventory"][k] == v for k, v in plan.SOURCES.items()), "Historical source qualification is incomplete")


def context(spec_ref):
    spec = plan.load_ref(spec_ref)
    require(spec.get("schema_version") == 1 and spec.get("purpose") == "prepare_no_loophole_cpu_inputs" and
            spec.get("runnable") is True, "Preparation spec is pending or wrong purpose")
    check_sources(spec)
    check_runtime(spec)
    old = plan._old(spec["old_plan"])
    audit = plan.load_ref(spec["prompt_audit"])
    proof = plan.load_ref(spec["prompt_audit_verification"])
    require(audit.get("status") == "canonical_prompts_verified_tokenization_pending" and audit.get("runnable") is False and
            audit.get("tokenization_verified") is False and audit.get("prompt_alignment_verified") is True and
            audit.get("record_count") == 37 and audit.get("old_request_plan", {}).get("sha256") == plan.OLD_FULL_SHA and
            audit.get("old_prepared_source", {}).get("sha256") == plan.OLD_PREPARED_SHA and
            audit.get("problem_ids") == old["selected_problem_ids"] and
            audit.get("original_primary_source_records") == old["primary_source_records"], "Wrong canonical prompt audit")
    require(proof.get("status") == "independently_verified_canonical_prompt_alignment_tokenization_pending" and
            proof.get("prompt_package", {}).get("sha256") == spec["prompt_audit"]["sha256"] and
            proof.get("record_count") == 37 and proof.get("problem_ids") == old["selected_problem_ids"] and
            proof.get("old_request_plan_sha256") == plan.OLD_FULL_SHA and proof.get("old_prepared_source_sha256") == plan.OLD_PREPARED_SHA and
            proof.get("canonical_nohint37_sha256") == plan.DATASET_SHA and proof.get("original_carriers37_sha256") == plan.CARRIERS_SHA and
            all(proof.get(k) is True for k in ("prompt_alignment_verified", "original_hint_roundtrip_verified", "original_raw_subsets_byte_identical")),
            "Canonical prompt audit lacks positive independent proof")
    root = plan.get_ref(spec["prompt_audit"]).parent
    payloads = {name: plan.read_ref(audit["files"][name], base=root) for name in AUDIT_FILES}
    require(h(payloads["canonical_nohint37.jsonl"]) == plan.DATASET_SHA and
            h(payloads["original_carriers37.jsonl"]) == plan.CARRIERS_SHA, "Wrong actual37 raw subsets")
    rows = plan.jsonl(payloads["canonical_nohint37.jsonl"])
    carriers = plan.jsonl(payloads["original_carriers37.jsonl"])
    require(len(rows) == len(carriers) == 37 and [str(r["id"]) for r in rows] == old["selected_problem_ids"] and
            [str(r["problem_id"]) for r in carriers] == old["selected_problem_ids"] and
            all(r["record_id"] == old["primary_source_records"][str(r["problem_id"])] and
                r["problem_split"] == "configuration_validation" for r in carriers), "Prompt population/carriers differ")
    policy = json.loads(payloads["tokenizer_policy.pending.json"])
    require(policy["model_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c" and
            set(spec["tokenizer_files"]) == {Path(f["archive_path"]).name for f in policy["files"]}, "Tokenizer file coverage differs")
    tokenizer_bytes = {}
    for entry in policy["files"]:
        name = Path(entry["archive_path"]).name
        ref = spec["tokenizer_files"][name]
        require(ref["sha256"] == entry["sha256"] and ref["size_bytes"] == entry["size_bytes"], "Wrong tokenizer revision/content")
        tokenizer_bytes[name] = content_ref(ref)
    for condition in old["conditions"].values():
        for layer in condition["layers"]:
            if layer["kind"] == "candidate":
                plan.read_ref({"path": layer["path"], "sha256": layer["sha256"]})
    return {"spec": spec, "old": old, "payloads": payloads, "rows": rows, "carriers": carriers,
            "tokenizer_bytes": tokenizer_bytes}


def load_tokenizer(directory):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)


def tokenized(ctx, tokenizer):
    prepared, records, rendered = [], [], []
    # Replay every old prompt before creating any new token sequence/carrier.
    for old in ctx["carriers"]:
        ids = list(tokenizer.apply_chat_template(old["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False))
        require(ids == old["prompt_token_ids"] and plan.ids_hash(ids) == old["prompt_token_ids_sha256"] and
                ids == old.get("engine_prompt_token_ids", ids), "Old prompt token replay differs")
    for old, row in zip(ctx["carriers"], ctx["rows"]):
        ids = list(tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False))
        text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        require(isinstance(text, str) and tokenizer.encode(text, add_special_tokens=False) == ids and
                tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text,
                "New rendered prompt/token roundtrip differs")
        current = plan.make_prepared_row(old, row, ids, dataset_sha256=plan.DATASET_SHA)
        prepared.append(current)
        records.append({"problem_id": str(row["id"]), "old_prompt_sha256": plan.prompt_hash(old["prompt"]),
                        "new_prompt_sha256": plan.prompt_hash(row["prompt"]),
                        "old_prompt_token_ids_sha256": plan.ids_hash(old["prompt_token_ids"]),
                        "new_prompt_token_ids_sha256": plan.ids_hash(ids),
                        "old_prompt_token_count": len(old["prompt_token_ids"]), "new_prompt_token_count": len(ids),
                        "old_prompt_ids_equal": True, "new_prompt_roundtrip_equal": True})
        rendered.append({"problem_id": str(row["id"]), "rendered_prompt": text, "rendered_prompt_sha256": h(text.encode())})
    data = b"".join((plan.canonical(r) + "\n").encode() for r in prepared)
    audit = {"status": "passed", "record_count": 37, "old_prompt_ids_replayed": True,
             "new_prompt_ids_roundtrip": True, "same_tokenizer_chat_template": True, "thinking": False,
             "prepared_records_sha256": h(data), "dataset_sha256": plan.DATASET_SHA, "records": records,
             "rendered_prompts": rendered, "tokenizer_files": ctx["spec"]["tokenizer_files"], "runtime_versions": VERSIONS,
             "tokenization": {"add_generation_prompt": True, "enable_thinking": False, "truncate": False, "limit": 1536}}
    return data, audit


def projection_snapshot(conditions):
    import torch
    from safetensors.torch import save, load
    from infra.gpu03.direction_discovery import engine
    require(Path(engine.__file__).resolve() == Path(__file__).resolve().parents[3] / 'infra/gpu03/direction_discovery/engine.py' and
            plan.sha(engine.__file__) == plan.SOURCES['infra/gpu03/direction_discovery/engine.py'], "Imported projection engine differs")
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CUDA was initialized in CPU preparation")
    vectors = {}
    for name, condition in conditions.items():
        q = engine.load_projections(condition, device="cpu")
        if not condition["layers"]:
            require(q == {}, "Baseline unexpectedly has a projection")
            continue
        require(set(q) == {21} and q[21].shape == (2560, 1) and q[21].dtype == torch.float32 and
                q[21].device.type == "cpu" and bool(torch.isfinite(q[21]).all()), "Wrong Q shape/dtype/device")
        vectors[name] = q[21].contiguous()
    require(len(vectors) == 4, "Expected exactly target plus three random Qs")
    data = save(vectors)
    loaded = load(data)
    raw = {k: v.numpy().tobytes(order="C") for k, v in vectors.items()}
    require(set(loaded) == set(vectors) and all(loaded[k].numpy().tobytes(order="C") == raw[k] for k in raw), "Projection serialization changed bits")
    return data, {k: h(v) for k, v in raw.items()}


def projection_audit(ctx, data, hashes):
    nonbase = {k: v for k, v in ctx["old"]["conditions"].items() if v["layers"]}
    return {"status": "passed", "torch_version": VERSIONS["torch"], "source_sha256": plan.SOURCES,
            "conditions": nonbase, "target_tensor_sha256": nonbase["target:L21.transition.pc04"]["layers"][0]["sha256"],
            "saved_tensor_sha256": h(data), "vector_sha256": hashes, "historical_random_Q_files_available": False,
            "historical_Q_byte_comparison_performed": False,
            "construction": "unchanged_engine.load_projections_on_original_pinned_cpu_runtime"}


def verify_projections(spec_ref, directory):
    ctx = context(spec_ref)
    directory = Path(directory)
    require(directory == Path(ctx["spec"]["output"]), "Projection output differs from reviewed spec")
    saved_ref = plan.ref(directory / "projection_vectors.safetensors")
    saved = plan.read_ref(saved_ref)
    audit_ref = plan.ref(directory / "projection_audit.json")
    audit = plan.load_ref(audit_ref)
    reconstructed, hashes = projection_snapshot(ctx["old"]["conditions"])
    require(saved == reconstructed and audit == projection_audit(ctx, reconstructed, hashes), "Projection reconstruction differs")
    return write(directory / "projection_independent_verification.json",
                 {"status": "independently_verified_projection_reconstruction", "audit_sha256": audit_ref["sha256"],
                  "saved_tensor_sha256": saved_ref["sha256"], "vectors": 4,
                  "reconstruction_bitwise_equal": True, "readback_bitwise_equal": True,
                  "source_files": ctx["spec"]["source_files"], "spec_sha256": spec_ref["sha256"]})


def process_identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        text = (proc / 'stat').read_text(); fields = text[text.rfind(')') + 2:].split()
        return {"pid": pid, "pgid": int(fields[2]), "start_ticks": int(fields[19]), "uid": proc.stat().st_uid}
    except FileNotFoundError:
        return None


def require_released(identity):
    require(Path('/proc').is_dir(), "Independent process release check requires the pinned Linux runtime")
    require(set(identity) == {"pid", "pgid", "start_ticks", "uid"} and
            all(type(v) is int and v > 0 for v in identity.values()) and identity['uid'] == os.getuid() and
            identity['pid'] == identity['pgid'], "Invalid owned producer identity")
    require(process_identity(identity['pid']) != identity, "Producer is still present")
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            item = process_identity(int(path.name))
            require(item is None or item['uid'] != identity['uid'] or item['pgid'] != identity['pgid'], "Producer group is not released")


def projection_child(spec_ref, output, control):
    command = [sys.executable, '-B', '-m', MODULE, '--spec', spec_ref['path'], '--spec-sha256', spec_ref['sha256'],
               '--verify-projections', str(output)]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    identity = process_identity(child.pid)
    try:
        out, err = child.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        # This helper is the sole child, with its own session; no model workers.
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        try:
            out, err = child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            out, err = child.communicate(timeout=5)
        write(control / 'projection_child_failure.json', {"timeout": True, "returncode": child.returncode})
        raise RuntimeError('Independent projection reconstruction timed out')
    write(control / 'projection_child_stdout.txt', out)
    write(control / 'projection_child_stderr.txt', err)
    write(control / 'projection_child_exit.json', {"command": command, "returncode": child.returncode,
                                                "child_reaped": child.poll() is not None, "child_identity": identity})
    require(child.returncode == 0 and identity is not None, "Projection reconstruction child failed")
    require_released(identity)
    answer = json.loads(out)
    require(answer == plan.ref(output / 'projection_independent_verification.json'), "Projection child output identity differs")


def isolated_tokenizer(ctx, directory):
    directory.mkdir(exist_ok=False)
    for name, data in ctx['tokenizer_bytes'].items():
        write(directory / name, data)
    directory.chmod(0o500)
    return load_tokenizer(directory)


def prepare(spec_ref):
    ctx = context(spec_ref)
    output, control = Path(ctx['spec']['output']), Path(ctx['spec']['control'])
    require(output.is_absolute() and control.is_absolute() and output.parent == control.parent and output != control and
            not output.exists() and not control.exists(), "Preparation requires fresh sibling output/control directories")
    control.mkdir(exist_ok=False)
    tokenizer = isolated_tokenizer(ctx, control / 'tokenizer_files')
    prepared, tokens = tokenized(ctx, tokenizer)
    vectors, hashes = projection_snapshot(ctx['old']['conditions'])
    output.mkdir(exist_ok=False)
    for name, data in ctx['payloads'].items():
        write(output / name, data)
    write(output / 'prepared_records.jsonl', prepared)
    write(output / 'tokenization_audit.json', tokens)
    write(output / 'projection_vectors.safetensors', vectors)
    write(output / 'projection_audit.json', projection_audit(ctx, vectors, hashes))
    projection_child(spec_ref, output, control)
    require(set(p.name for p in output.iterdir()) == set(plan.FILES), "Unexpected preparation payload")
    files = {name: {**plan.ref(output / name), 'path': name} for name in plan.FILES}
    inventory = {name: {k: v for k, v in value.items() if k != 'path'} for name, value in files.items()}
    artifact = write(output / 'artifact_manifest.json', {'algorithm': 'sha256', 'files': inventory})
    bundle = {'schema_version': 1, 'purpose': 'canonical_no_loophole_prompt_package', 'status': 'tokenized_verified',
              'runnable': True, 'record_count': 37, 'prompt_alignment_verified': True, 'tokenization_verified': True,
              'old_request_plan_sha256': plan.OLD_FULL_SHA, 'old_prepared_source_sha256': plan.OLD_PREPARED_SHA,
              'prompt_audit': ctx['spec']['prompt_audit'], 'prompt_audit_verification': ctx['spec']['prompt_audit_verification'],
              'files': files, 'artifact_manifest': {**artifact, 'path': 'artifact_manifest.json'},
              'preparation_spec': spec_ref, 'preparation_source_files': ctx['spec']['source_files']}
    bundle_ref = write(output / 'prompt_package.json', bundle)
    validated = plan._package(bundle_ref, ctx['old']); validated['old_plan'] = ctx['old']
    plan._projection_check(validated)
    plan._input_check(validated, plan.jsonl(prepared), ctx['rows'])
    output.chmod(0o500)
    return {'prompt_package': bundle_ref, 'artifact_manifest': artifact, 'independent_verification_required': True,
            'model_calls': 0, 'gpu_calls': 0, 'requests_created': False}


def verify(spec_ref, bundle_ref, producer_exit_ref, output_proof):
    # Positive actual exit/release and result binding before any scientific rows.
    receipt = plan.load_ref(producer_exit_ref)
    require(receipt.get('returncode') == 0 and receipt.get('timed_out') is False and receipt.get('error_type') is None and
            receipt.get('child_reaped') is True and receipt.get('remaining_group_pids') == [], "Producer did not exit and release successfully")
    require_released(receipt['child_identity'])
    answer = json.loads(plan.read_ref(receipt['stdout']))
    actual_bundle_ref = answer.get('prompt_package', {})
    require(plan.get_ref(actual_bundle_ref) == plan.get_ref(bundle_ref) and
            actual_bundle_ref.get('sha256') == bundle_ref['sha256'] and answer.get('independent_verification_required') is True,
            "Actual producer stdout does not bind this package")
    ctx = context(spec_ref)
    require(plan.get_ref(bundle_ref) == Path(ctx['spec']['output']) / 'prompt_package.json', "Package path differs from spec")
    output_proof = Path(output_proof)
    require(output_proof.is_absolute() and not output_proof.exists() and
            not output_proof.is_relative_to(Path(ctx['spec']['output'])), "Independent proof must be fresh and outside the frozen package")
    bundle = plan.load_ref(bundle_ref)
    require(bundle.get('preparation_spec') == spec_ref and bundle.get('preparation_source_files') == ctx['spec']['source_files'], "Package source/spec differs")
    checked = plan._package(bundle_ref, ctx['old']); checked['old_plan'] = ctx['old']
    plan._projection_check(checked)
    require(set(p.name for p in plan.get_ref(bundle_ref).parent.iterdir()) == set(plan.FILES) | {'artifact_manifest.json', 'prompt_package.json'}, "Package has unexpected files")
    verify_dir = Path(output_proof).parent / (Path(output_proof).name + '.tokenizer')
    tokenizer = isolated_tokenizer(ctx, verify_dir)
    data, audit = tokenized(ctx, tokenizer)
    require(data == checked['payloads']['prepared_records.jsonl'] and audit == json.loads(checked['payloads']['tokenization_audit.json']), "Independent token reconstruction differs")
    vectors, hashes = projection_snapshot(ctx['old']['conditions'])
    require(vectors == checked['payloads']['projection_vectors.safetensors'] and
            projection_audit(ctx, vectors, hashes) == json.loads(checked['payloads']['projection_audit.json']), "Independent Q reconstruction differs")
    plan._input_check(checked, plan.jsonl(data), ctx['rows'])
    check_sources(ctx['spec'])
    require_released(receipt['child_identity'])
    return write(output_proof, {'status': 'independently_verified_tokenized_no_loophole_prompt_package',
                 'bundle_sha256': bundle_ref['sha256'], 'artifact_manifest_sha256': bundle['artifact_manifest']['sha256'],
                 'records': 37, 'prompt_token_replay': True, 'projection_reconstruction_bitwise_equal': True,
                 'producer_exit_verified': True, 'process_release_verified': True, 'producer_exit': producer_exit_ref,
                 'spec': spec_ref, 'source_files': ctx['spec']['source_files'], 'model_calls': 0, 'gpu_calls': 0,
                 'requests_created': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha256', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-projections')
    mode.add_argument('--verify-package')
    parser.add_argument('--package-sha256')
    parser.add_argument('--producer-exit')
    parser.add_argument('--producer-exit-sha256')
    parser.add_argument('--output-proof')
    args = parser.parse_args()
    spec = {'path': args.spec, 'sha256': args.spec_sha256}
    if args.verify_projections:
        result = verify_projections(spec, args.verify_projections)
    elif args.verify_package:
        require(all((args.package_sha256, args.producer_exit, args.producer_exit_sha256, args.output_proof)), 'Incomplete independent verification arguments')
        result = verify(spec, {'path': args.verify_package, 'sha256': args.package_sha256},
                        {'path': args.producer_exit, 'sha256': args.producer_exit_sha256}, args.output_proof)
    else:
        result = prepare(spec)
    print(plan.canonical(result))


if __name__ == '__main__':
    main()
