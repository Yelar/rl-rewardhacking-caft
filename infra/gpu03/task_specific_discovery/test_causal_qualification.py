"""Small synthetic retained records only; no generation/evaluation/CUDA calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from infra.gpu03.task_specific_discovery import causal_qualification as cq
from infra.gpu03.task_specific_discovery.io_contract import RequestJournal, canonical, hash_value, ContractError


def binding(path):
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")
    return binding(path)


def safetensors(path, tensors):
    header, payload = {}, b""
    for name, array in tensors.items():
        array = np.asarray(array, dtype="<f4")
        data = array.tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    raw = canonical(header)
    raw += b" " * ((-len(raw)) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return binding(path)


class Fixture:
    def __init__(self, root, *, pairs=1, mutate_request=None, mutate_result=None, mutate_logits=None, omit_coordinate=False):
        self.root = Path(root).resolve()
        self.source = self.root / "source"
        self.runtime = self.root / "runtime"
        self.out = self.root / "worker0"
        self.source.mkdir(); self.runtime.mkdir(); self.out.mkdir()
        source_files = {}
        for name in cq.REQUIRED_SOURCES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic file, never imported or executed\n")
            source_files[name] = binding(path)
        model = self.root / "model"; model.mkdir()
        checkpoint = self.root / "checkpoint"; checkpoint.mkdir()
        self.inputs = {}
        for name in cq.worker.BASE_FILES:
            self.inputs[str(model / name)] = write(model / name, {"vocab_size": 8} if name == "config.json" else {})
        self.adapter_hashes = {}
        for name in cq.worker.ADAPTERS:
            value = write(checkpoint / name, {"authored_inert_adapter": name})
            self.inputs[str(checkpoint / name)] = value
            self.adapter_hashes[name] = value["sha256"]
        q = np.zeros((2560, 1), np.float32); q[0] = 1
        tensors = {"axis": q, "center": np.zeros(2560, np.float32)}
        vector_path = self.root / "vectors.safetensors"
        self.inputs[str(vector_path)] = safetensors(vector_path, tensors)
        conditions = {"baseline": {"id": "baseline", "layer": None, "operator": "ordinary", "control": "zero_strength", "strength": 0}}
        null_pairs = []
        for i in range(pairs):
            operator = "centered" if i % 2 else "ordinary"
            signature = {"layer": i, "operator": operator, "vector_key": "axis", "rank": 1}
            if operator == "centered": signature["center_key"] = "center"
            for control in ("zero_strength", "materialization_only"):
                name = f"{control}{i}"
                conditions[name] = {"id": name, **signature, "control": control, "strength": 0}
            null_pairs.append({k: f"{k}{i}" for k in ("zero_strength", "materialization_only")})
        rows = []
        for pid in range(4):
            for protocol in cq.PROTOCOLS:
                rows.append({"record_id": f"p{pid}-{protocol}", "problem_id": pid, "problem_split": "direction_fit",
                             "protocol": protocol, "prompt": [{"role": "user", "content": f"inert-{pid}-{protocol}"}],
                             "prompt_token_ids": [pid + 1]})
        row_by_id = {r["record_id"]: r for r in rows}
        prepared = self.root / "prepared.jsonl"
        prepared.write_bytes(b"".join(canonical(r) + b"\n" for r in rows))
        self.inputs[str(prepared)] = binding(prepared)
        requests = []
        for cid in conditions:
            for row in rows:
                for sample in (0, 1):
                    request = {"request_id": f"{cid}-{row['record_id']}-{sample}", "record_id": row["record_id"],
                               "condition_id": cid, "problem_id": row["problem_id"], "protocol": row["protocol"],
                               "sample_index": sample, "seed": 100 + row["problem_id"] * 2 + sample, "scope": "primary"}
                    if mutate_request: mutate_request(request)
                    requests.append(request)
        if omit_coordinate: requests.pop()
        task = {"schema": "task_specific_causal_generation_v1", "stage": "qualification", "model_revision": cq.worker.REVISION,
                "run_token": "codex-authored-fixture", "sampling": {**cq.worker.SAMPLING, "eos_token_ids": [7]},
                "profile": cq.PROFILE, "batch_size": 8, "deadline_seconds": 3600,
                "model_snapshot": str(model), "checkpoint": str(checkpoint), "vectors": str(vector_path),
                "prepared_rows": str(prepared), "output": str(self.out), "requests": requests,
                "conditions": conditions, "bound_files": self.inputs, "qualified_source_files": source_files}
        self.task_path = self.root / "task.json"; task_ref = write(self.task_path, task); task_sha = task_ref["sha256"]
        uuid = "GPU-00000000-0000-0000-0000-000000000000"
        write(self.out / "STARTED.json", {"task_sha256": task_sha, "pid": 123, "requests": len(requests), "gpu_uuid": uuid, "stage": "qualification"})
        artifacts, exports = {}, []
        self.trace_files = []
        with RequestJournal(self.out / "journal", run_token=task["run_token"], input_sha256=task_sha,
                            request_hashes={r["request_id"]: hash_value(r) for r in requests}) as journal:
            for index, batch in enumerate(cq.batches(requests)):
                filename = f"batch_{index:04d}.first_logits.safetensors"
                logits = np.tile(np.arange(8, dtype=np.float32), (len(batch), 1))
                if mutate_logits: mutate_logits(batch, logits)
                artifacts[filename] = safetensors(self.out / filename, {"logits": logits}); self.trace_files.append(self.out / filename)
                trace = {"path": filename, **artifacts[filename], "request_ids": [r["request_id"] for r in batch],
                         "logical_positions_verified": True, "eos_exclusion_verified": True}
                for lane, request in enumerate(batch):
                    result = {"completion_token_ids": [1, 7], "generated_token_ids": [1, 7], "completion": "authored inert result",
                              "fixed_completion_prefix_token_count": 0, "batch_size_actual": len(batch), "batch_profile": cq.PROFILE,
                              "intervention_strength": 0, "stop_reason": "eos", "batch_wall_seconds": None, "energy": {}}
                    if mutate_result: mutate_result(request, result)
                    primitive = {"result": result}
                    journal.start(request["request_id"]); journal.record_primitive(request["request_id"], primitive)
                    journal.record_final(request["request_id"], {"returned_row": True, "batch_index": index, "batch_row": lane,
                                                               "first_logits": trace, "batch_wall_seconds": 1.0})
                    exports.append(cq.worker.export_row(request, row_by_id[request["record_id"]], primitive, task_sha))
        # Worker exports original task order (not arrival order).
        exports = {r["request_id"]: r for r in exports}
        (self.out / "generations.jsonl").write_bytes(b"".join(canonical(exports[r["request_id"]]) + b"\n" for r in requests))
        artifacts["generations.jsonl"] = binding(self.out / "generations.jsonl")
        write(self.out / "COMPLETE.json", {"status": "succeeded", "stage": "qualification", "task_sha256": task_sha,
              "requests": len(requests), "all_returned_requests_retained": True, "evaluation_performed": False,
              "causal_success_claimed": False, "cuda_release": {"allocated_bytes": 32 << 20, "reserved_bytes": 32 << 20},
              "artifacts": artifacts, "model_load_report": {"with_adapter": True, "active_adapters": ["default"],
                  "lora_parameter_tensors": 2, "nonzero_lora_parameter_tensors": 2,
                  "lora_parameter_elements": 8, "base_model_class": "AuthoredInertModel"}})
        cfg = {"schema": "task_specific_null_qualification_contract_v1", "problem_ids": [0, 1, 2, 3],
               "protocols": cq.PROTOCOLS, "sample_indices": [0, 1], "baseline_condition_id": "baseline", "null_pairs": null_pairs,
               "logits_atol": 1e-5, "logits_rtol": 1e-5, "source_files": source_files,
               "vectors_sha256": self.inputs[str(vector_path)]["sha256"]}
        manifest = {"schema": "gpu03_task_specific_resource_guard_v1", "host": "gpu-03", "uid": 3357,
                    "run_token": task["run_token"], "source_root": str(self.source), "runtime": str(self.runtime),
                    "qualification_contract": cfg, "gpu_uuids": {"0": uuid},
                    "bound_files": {str(self.task_path): task_ref, **{str(self.source / k): v for k, v in source_files.items()}},
                    "workers": [{"name": "worker0", "gpu_id": 0, "cwd": str(self.source),
                                 "env": {"CUDA_VISIBLE_DEVICES": uuid, "PYTHONPATH": str(self.source)},
                                 "argv": ["/usr/bin/python3", "-B", "-m", "infra.gpu03.task_specific_discovery.causal_worker",
                                          "--task", str(self.task_path), "--sha256", task_sha]}]}
        self.manifest_path = self.root / "phase.json"; self.digest = write(self.manifest_path, manifest)["sha256"]
        self.release = {"run_token": task["run_token"], "manifest_sha256": self.digest, "status": "workers_exited_zero",
                        "owned_processes_released": True, "gpu_release_verified": True, "scientific_success_verified": False,
                        "exit_codes": {"worker0": 0}, "gpus": {"0": {"uuid": uuid, "pids": [], "utilization": 0,
                                                                   "total_mib": 32000, "free_mib": 32000}}}
        write(self.runtime / "SUPERVISOR_RESULT.json", self.release)
        (self.runtime / "workers.jsonl").write_bytes(canonical({"name": "worker0", "pid": 123, "start": 111, "gpu_id": 0, "at": 1}) + b"\n")

    def run(self):
        with mock.patch.object(cq.worker, "ADAPTERS", self.adapter_hashes), mock.patch.object(cq.worker, "SAMPLING", {**cq.worker.SAMPLING, "eos_token_ids": [7]}):
            return cq.audit(self.manifest_path, self.digest)


class QualificationTests(unittest.TestCase):
    def setUp(self): self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
    def tearDown(self): self.temp.cleanup()
    def make(self, **kwargs): return Fixture(self.root, **kwargs)

    def test_full_success_checks_durable_mapping_and_no_mutation(self):
        f = self.make(); before = {str(p): binding(p) for p in self.root.rglob('*') if p.is_file()}
        proof = f.run()
        after = {str(p): binding(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(proof["requests"], 72); self.assertEqual(len(proof["retained_requests"]), 72)
        self.assertEqual(proof["qualified_signatures"], [{"layer": 0, "operator": "ordinary", "rank": 1}])
        self.assertTrue(proof["all_returned_requests_retained"]); self.assertTrue(proof["gpu_workers_released"])
        self.assertFalse(proof["behavioral_success_claimed"])

    def test_full_six_signatures_and_centered_paths(self):
        f = self.make(pairs=6); proof = f.run()
        self.assertEqual(proof["requests"], 312); self.assertEqual(proof["conditions"], 13)
        self.assertEqual(len(proof["qualified_signatures"]), 6)
        self.assertEqual(proof["qualified_signatures"][1]["center_key"], "center")

    def test_missing_coordinate_even_when_producer_claims_complete_rejected(self):
        f = self.make(omit_coordinate=True)
        with self.assertRaisesRegex(ContractError, "coverage|lane"): f.run()

    def test_token_difference_rejected_with_all_hashes_self_consistent(self):
        def mutate(req, result):
            if req['condition_id'] == 'zero_strength0': result.update(completion_token_ids=[2,7],generated_token_ids=[2,7])
        f = self.make(mutate_result=mutate)
        with self.assertRaisesRegex(ContractError, "token/text parity"): f.run()

    def test_logit_tolerance_failure_and_small_allowed_difference(self):
        f = self.make(mutate_logits=lambda batch, x: x.__setitem__((0,0), .001) if batch[0]['condition_id']=='materialization_only0' else None)
        with self.assertRaisesRegex(ContractError, "logit tolerance"): f.run()

    def test_small_logit_delta_inside_frozen_tolerance_passes(self):
        f = self.make(mutate_logits=lambda batch, x: x.__setitem__((0,0), 1e-6) if batch[0]['condition_id']=='materialization_only0' else None)
        self.assertTrue(f.run()['null_logits_within_tolerance'])

    def test_seed_pairing_difference_rejected(self):
        f = self.make(mutate_request=lambda r: r.update(seed=r['seed']+1) if r['condition_id']=='zero_strength0' else None)
        with self.assertRaisesRegex(ContractError, "pairing"): f.run()

    def test_positive_release_boolean_alone_not_enough(self):
        f = self.make(); f.release['gpus']['0']['pids']=[987]
        write(f.runtime/'SUPERVISOR_RESULT.json',f.release)
        with self.assertRaisesRegex(ContractError, 'GPU release'): f.run()

    def test_missing_worker_exit_rejected(self):
        f = self.make(); f.release['exit_codes']={}
        write(f.runtime/'SUPERVISOR_RESULT.json',f.release)
        with self.assertRaisesRegex(ContractError, 'supervisor'): f.run()

    def test_changed_source_rejected(self):
        f=self.make();(f.source/next(iter(cq.REQUIRED_SOURCES))).write_text('changed')
        with self.assertRaisesRegex(ContractError,'SHA differs'):f.run()

    def test_changed_logits_rejected_before_numerical_comparison(self):
        f=self.make();p=f.trace_files[0];p.write_bytes(p.read_bytes()+b'\0')
        with self.assertRaisesRegex(ContractError,'SHA differs'):f.run()

    def test_truncated_final_journal_rejected_without_adopting_payload(self):
        f=self.make();p=f.out/'journal/final_recorded.jsonl';p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaisesRegex(ContractError,'Truncated'):f.run()

    def test_missing_primitive_journal_event_rejected_without_adoption(self):
        f=self.make();p=f.out/'journal/primitive_returned.jsonl';p.write_bytes(b'\n'.join(p.read_bytes().splitlines()[:-1])+b'\n')
        with self.assertRaisesRegex(ContractError,'coverage'):f.run()

    def test_unexpected_partial_file_rejected(self):
        f=self.make();write(f.out/'batch_0000.partial.json',{})
        with self.assertRaisesRegex(ContractError,'unexpected worker output'):f.run()

    def test_worker_gpu_launch_mismatch_rejected(self):
        f=self.make();p=f.out/'STARTED.json';v=json.loads(p.read_text());v['pid']=124;write(p,v)
        with self.assertRaisesRegex(ContractError,'launch/start'):f.run()

    def test_adapter_load_must_be_positive(self):
        f=self.make();p=f.out/'COMPLETE.json';v=json.loads(p.read_text());v['model_load_report']['nonzero_lora_parameter_tensors']=0;write(p,v)
        with self.assertRaisesRegex(ContractError,'adapter load'):f.run()

    def test_unsafe_tolerance_not_allowed(self):
        f=self.make();p=f.manifest_path;m=json.loads(p.read_text());m['qualification_contract']['logits_atol']=.1
        f.digest=write(p,m)['sha256'];f.release['manifest_sha256']=f.digest;write(f.runtime/'SUPERVISOR_RESULT.json',f.release)
        with self.assertRaisesRegex(ContractError,'tolerance'):f.run()

    def test_safetensor_nonfinite_and_unmapped_bytes_rejected(self):
        path=self.root/'bad.safetensors';safetensors(path,{'logits':np.array([[np.nan]],np.float32)})
        with self.assertRaisesRegex(ContractError,'Nonfinite'):cq.load_tensors(path,cq.Evidence())
        safetensors(path,{'logits':np.array([[1.]],np.float32)});path.write_bytes(path.read_bytes()+b'\0')
        with self.assertRaisesRegex(ContractError,'Unmapped'):cq.load_tensors(path,cq.Evidence())

    def test_duplicate_json_and_symlink_evidence_rejected(self):
        with self.assertRaisesRegex(ContractError,'Duplicate'):cq.decode(b'{"a":1,"a":2}')
        p=self.root/'a';write(p,{});q=self.root/'alias';q.symlink_to(p)
        with self.assertRaisesRegex(ContractError,'Noncanonical'):cq.Evidence().bind(q)

    def test_hash_bound_vendor_json_need_not_have_trailing_newline(self):
        p=self.root/'config.json';p.write_bytes(b'{"vocab_size": 8}')
        self.assertEqual(cq.Evidence().json(p),{'vocab_size':8})
        with self.assertRaisesRegex(ContractError,'Noncanonical'):
            cq.Evidence().json(p,canonical_required=True)


if __name__ == '__main__': unittest.main()
