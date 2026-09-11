"""Authored37-problem metadata fixtures; no actual tokenizer/model or dataset."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.direction_discovery import no_loophole_plan as n


class Fixture:
    """Reusable real planner fixture for builder/evaluator metadata tests."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.patches = []
        self.pkg = self.root / "package"
        self.pkg.mkdir()
        self.target = self.save(self.root / "candidate.safetensors", b"authored candidate payload")
        self.conditions = {"baseline": {"role": "baseline", "layers": []},
            "target:L21.transition.pc04": {"role": "target", "layers": [{"layer": 21, "kind": "candidate",
                "path": str(self.target), "sha256": n.sha(self.target), "selectors": [{"key": "pca.pcs", "column": 4}]}]}}
        for base, seed in ((6101, 1069110841), (6102, 660972308), (6103, 1074926686)):
            self.conditions[f"random:L21r1:base{base}"] = {"role": "random", "random_seed_base": base,
                "layers": [{"layer": 21, "kind": "random", "rank": 1, "seed": seed}]}
        self.problems = [str(i) for i in range(37)]
        self.dataset = [{"id": i, "prompt": self.prompt("solve " + i), "hint": None, "evaluator": "code",
                         "answer": ["assert True"], "gt_answer": ["assert True"],
                         "setup_code": "", "func_name": "solve", "prompt_metadata": {"starter_code": "def solve(): pass"}}
                        for i in self.problems]
        self.carriers = [{"record_id": "old-" + i, "problem_id": int(i), "problem_split": "configuration_validation",
                          "prompt": self.prompt("solve " + i + "; run_tests loophole"), "prompt_token_ids": [1, 2, int(i) + 20],
                          "input_ids_sha256": n.ids_hash([1, 2, int(i) + 20, 3]), "completion_token_ids": [3],
                          "completion": "pass", "completion_sha256": n.text_hash("pass"),
                          "outcome_presence_class": "clean_correct_evaluator_present"} for i in self.problems]
        self.source_map = {str(r["problem_id"]): r["record_id"] for r in self.carriers}
        requests = [{"request_id": f"old-{i}-{sample}-{cond}", "record_id": "old-" + i, "problem_id": int(i),
                     "problem_split": "configuration_validation", "scope": "primary", "sample_index": sample,
                     "seed": n.stable_seed(i, sample), "condition_id": cond}
                    for i in self.problems for sample in range(4) for cond in self.conditions]
        self.old = {"purpose": "checkpoint60_paired_causal_behavior", "phase": "finalist_validation", "mode": "generate",
                    "evaluation_partition": "configuration_validation", "master_plan_sha256": n.MASTER_SHA,
                    "parent_plan_sha256": n.PARENT_SHA, "sampling": n.SAMPLING, "test_used_for_selection": False, "no_training": True,
                    "selected_problem_ids": self.problems, "selected_target_ids": ["target:L21.transition.pc04"],
                    "primary_source_records": self.source_map, "conditions": self.conditions, "requests": requests,
                    "new_generation_requests": 740, "new_tf_requests": 0, "scope_counts": {"primary": 740},
                    "counts_per_condition": {"primary": 148, "local": 0}, "local_source_records": [],
                    "prepared_records": str(self.root / "historical-prepared"), "source_bindings": {},
                    "previously_committed_generation_requests": 1786, "previously_committed_tf_requests": 9222}
        self.old_path = self.save(self.root / "old_plan.json", self.old)
        self.set_constant("OLD_FULL_SHA", n.sha(self.old_path))
        self.save(self.pkg / "canonical_nohint37.jsonl", self.lines(self.dataset))
        self.save(self.pkg / "original_carriers37.jsonl", self.lines(self.carriers))
        self.set_constant("DATASET_SHA", n.sha(self.pkg / "canonical_nohint37.jsonl"))
        self.set_constant("CARRIERS_SHA", n.sha(self.pkg / "original_carriers37.jsonl"))
        self.pairs = [{"problem_id": str(old["problem_id"]), "old_prompt": old["prompt"], "new_prompt": data["prompt"],
                       "old_prompt_sha256": n.prompt_hash(old["prompt"]), "new_prompt_sha256": n.prompt_hash(data["prompt"]),
                       "original_primary_source_record": old["record_id"], "all_repository_evaluator_names_absent_from_new_prompt": True}
                      for old, data in zip(self.carriers, self.dataset)]
        self.save(self.pkg / "prompt_pairs.jsonl", self.lines(self.pairs))
        for name in ("alignment_audit.json", "request_alignment.json", "source_bindings.json", "tokenizer_policy.pending.json"):
            self.save(self.pkg / name, {"authored": True})
        audit_files = {name: self.relative_ref(name) for name in n.FILES[:7]}
        self.audit = {"status": "canonical_prompts_verified_tokenization_pending", "runnable": False,
                      "prompt_alignment_verified": True, "old_request_plan": n.ref(self.old_path),
                      "old_prepared_source": {"sha256": n.OLD_PREPARED_SHA}, "problem_ids": self.problems,
                      "original_primary_source_records": self.source_map, "files": audit_files}
        self.audit_path = self.save(self.root / "prompt_audit.json", self.audit)
        self.audit_proof = {"status": "independently_verified_canonical_prompt_alignment_tokenization_pending",
                           "prompt_package": n.ref(self.audit_path), "record_count": 37, "problem_ids": self.problems,
                           "old_request_plan_sha256": n.OLD_FULL_SHA, "old_prepared_source_sha256": n.OLD_PREPARED_SHA,
                           "canonical_nohint37_sha256": n.DATASET_SHA, "original_carriers37_sha256": n.CARRIERS_SHA,
                           "prompt_alignment_verified": True, "original_hint_roundtrip_verified": True,
                           "original_raw_subsets_byte_identical": True, "tokenization_verified": False, "runnable": False}
        self.audit_proof_path = self.save(self.root / "prompt_audit_verification.json", self.audit_proof)
        self.prepared = [n.make_prepared_row(old, data, [1, int(data["id"]) + 100], dataset_sha256=n.DATASET_SHA)
                         for old, data in zip(self.carriers, self.dataset)]
        self.save(self.pkg / "prepared_records.jsonl", self.lines(self.prepared))
        token_records = [{"problem_id": str(old["problem_id"]), "old_prompt_sha256": n.prompt_hash(old["prompt"]),
                          "new_prompt_sha256": n.prompt_hash(new["prompt"]), "old_prompt_token_ids_sha256": n.ids_hash(old["prompt_token_ids"]),
                          "new_prompt_token_ids_sha256": n.ids_hash(new["prompt_token_ids"]), "old_prompt_token_count": len(old["prompt_token_ids"]),
                          "new_prompt_token_count": len(new["prompt_token_ids"]), "old_prompt_ids_equal": True, "new_prompt_roundtrip_equal": True}
                         for old, new in zip(self.carriers, self.prepared)]
        self.save(self.pkg / "tokenization_audit.json", {"status": "passed", "record_count": 37,
            "old_prompt_ids_replayed": True, "new_prompt_ids_roundtrip": True, "same_tokenizer_chat_template": True,
            "thinking": False, "prepared_records_sha256": n.sha(self.pkg / "prepared_records.jsonl"),
            "dataset_sha256": n.DATASET_SHA, "records": token_records})
        header, payload, raw_hashes = {}, bytearray(), {}
        nonbaseline = {k: v for k, v in self.conditions.items() if v["layers"]}
        for i, condition in enumerate(nonbaseline):
            raw = struct.pack("<2560f", *[float(j == i) for j in range(2560)])
            header[condition] = {"dtype": "F32", "shape": [2560, 1], "data_offsets": [len(payload), len(payload) + len(raw)]}
            payload.extend(raw); raw_hashes[condition] = n.hashlib.sha256(raw).hexdigest()
        header_bytes = n.canonical(header).encode()
        self.save(self.pkg / "projection_vectors.safetensors", len(header_bytes).to_bytes(8, "little") + header_bytes + payload)
        vector_hash = n.sha(self.pkg / "projection_vectors.safetensors")
        self.save(self.pkg / "projection_audit.json", {"status": "passed", "torch_version": "2.8.0+cu128",
            "source_sha256": n.SOURCES, "conditions": nonbaseline, "target_tensor_sha256": n.sha(self.target),
            "saved_tensor_sha256": vector_hash, "historical_random_Q_files_available": False,
            "historical_Q_byte_comparison_performed": False, "construction": "unchanged_engine.load_projections_on_original_pinned_cpu_runtime",
            "vector_sha256": raw_hashes})
        self.save(self.pkg / "projection_independent_verification.json", {"status": "independently_verified_projection_reconstruction",
            "audit_sha256": n.sha(self.pkg / "projection_audit.json"), "saved_tensor_sha256": vector_hash,
            "vectors": 4, "reconstruction_bitwise_equal": True, "readback_bitwise_equal": True})
        self.bundle = {"schema_version": 1, "purpose": "canonical_no_loophole_prompt_package", "status": "tokenized_verified",
                       "runnable": True, "record_count": 37, "prompt_alignment_verified": True, "tokenization_verified": True,
                       "old_request_plan_sha256": n.OLD_FULL_SHA, "old_prepared_source_sha256": n.OLD_PREPARED_SHA,
                       "prompt_audit": n.ref(self.audit_path), "prompt_audit_verification": n.ref(self.audit_proof_path)}
        self.budget = {"generation_requests": 2646, "tf_requests": 9228, "untouched_test_requests": 0,
                       "untouched_test_generation_requests": 0, "untouched_test_tf_requests": 0, "phases": []}
        self.refreeze()

    @staticmethod
    def prompt(text):
        return [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": text}]

    @staticmethod
    def lines(rows):
        return ("\n".join(n.canonical(row) for row in rows) + "\n").encode()

    def set_constant(self, key, value):
        p = patch.object(n, key, value); p.start(); self.patches.append(p)

    def close(self):
        for p in reversed(self.patches): p.stop()

    def save(self, path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): path.chmod(0o600)
        path.write_bytes(value if isinstance(value, (bytes, bytearray)) else (n.canonical(value) + "\n").encode())
        path.chmod(0o400)
        return path

    def relative_ref(self, name):
        return {**n.ref(self.pkg / name), "path": name}

    def refreeze(self, *, build=True):
        self.bundle["files"] = {name: self.relative_ref(name) for name in n.FILES}
        self.save(self.pkg / "artifact_manifest.json", {"files": {name: {k: v for k, v in x.items() if k != "path"}
                                                                for name, x in self.bundle["files"].items()}})
        self.bundle["artifact_manifest"] = self.relative_ref("artifact_manifest.json")
        self.bundle_path = self.save(self.pkg / "bundle.json", self.bundle)
        self.bundle_ref = n.ref(self.bundle_path)
        self.proof = {"status": "independently_verified_tokenized_no_loophole_prompt_package",
                      "bundle_sha256": self.bundle_ref["sha256"], "artifact_manifest_sha256": self.bundle["artifact_manifest"]["sha256"],
                      "records": 37, "prompt_token_replay": True, "projection_reconstruction_bitwise_equal": True,
                      "producer_exit_verified": True, "process_release_verified": True}
        self.proof_path = self.save(self.root / "package_verification.json", self.proof)
        self.proof_ref = n.ref(self.proof_path)
        if build: self.plan = self.build()

    def build(self):
        return n.build_plan(self.old_path, self.bundle_ref, self.budget, prompt_package_verification=self.proof_ref)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name).resolve()); self.addCleanup(self.f.close)

    def test_full_producer_verifier_real_inputs(self):
        f = self.f
        ctx = n.validate_full(f.plan)
        self.assertEqual(ctx["prepared_records"], f.pkg / "prepared_records.jsonl")
        n.validate_inputs(f.plan, f.prepared, f.dataset)
        self.assertEqual(f.plan["conditions"], f.old["conditions"])
        self.assertEqual([r["seed"] for r in f.plan["requests"]], [r["seed"] for r in f.old["requests"]])
        self.assertFalse(set(r["request_id"] for r in f.plan["requests"]) & set(r["request_id"] for r in f.old["requests"]))
        self.assertIn(f.target, n.bindings(f.plan))
        proof = n.write_plan(f.old_path, f.bundle_ref, f.budget, f.root / "output", prompt_package_verification=f.proof_ref)
        self.assertFalse(proof["budget_reserved"])
        n.validate_full(json.loads(Path(proof["request_plan"]).read_text()))
        with self.assertRaises(ValueError): n.write_plan(f.old_path, f.bundle_ref, f.budget, f.root / "output", prompt_package_verification=f.proof_ref)

    def test_list_prompt_hash_and_minimal_carrier(self):
        f = self.f
        self.assertEqual(n.prompt_hash(f.dataset[0]["prompt"]), n.digest(f.dataset[0]["prompt"]))
        self.assertEqual(f.prepared[0]["completion_token_ids"], f.carriers[0]["completion_token_ids"])
        self.assertTrue(f.prepared[0]["historical_carrier"]["completion_provenance_only"])
        self.assertFalse({"regions", "engine_prompt_token_ids", "selected_token_positions"} & set(f.prepared[0]))

    def test_request_condition_seed_scope_sample_and_source_drift(self):
        for key, value in (("seed", 1), ("scope", "local"), ("sample_index", 4), ("problem_id", 999), ("record_id", "wrong"), ("request_id", "old"), ("condition_id", "other")):
            with self.subTest(key=key):
                plan = copy.deepcopy(self.f.plan); plan["requests"][0][key] = value
                with self.assertRaises(ValueError): n.validate_full(plan)
        for modify in (lambda p: p["requests"].pop(), lambda p: p["requests"].append(p["requests"][0]),
                       lambda p: p["requests"].reverse(), lambda p: p["conditions"]["baseline"].update(role="target"),
                       lambda p: p["sampling"].update(temperature=.8)):
            plan = copy.deepcopy(self.f.plan); modify(plan)
            with self.assertRaises(ValueError): n.validate_full(plan)

    def test_source_and_historical_lineage_drift(self):
        for key in ("master_plan_sha256", "parent_plan_sha256", "builder_sha256", "authorization"):
            plan = copy.deepcopy(self.f.plan); plan[key] = "bad"
            with self.assertRaises(ValueError): n.validate_full(plan)
        plan = copy.deepcopy(self.f.plan); plan["no_loophole_capability"]["frozen_generation_sources"] = {}
        with self.assertRaises(ValueError): n.validate_full(plan)

    def test_no_partial_test_or_recovery(self):
        for key in ("execution_partition", "test_execution", "test_bundle", "finalist_recovery", "recovery_plan"):
            plan = copy.deepcopy(self.f.plan); plan[key] = {}
            with self.assertRaises(ValueError): n.validate_full(plan)

    def test_mutable_symlink_missing_or_changed_input(self):
        path = self.f.pkg / "prepared_records.jsonl"
        path.chmod(0o600)
        with self.assertRaises(ValueError): n.validate_full(self.f.plan)
        path.chmod(0o400)
        path.rename(path.with_suffix(".saved")); path.symlink_to(path.with_suffix(".saved"))
        with self.assertRaises(ValueError): n.validate_full(self.f.plan)
        path.unlink()
        with self.assertRaises(ValueError): n.validate_full(self.f.plan)

    def test_caller_input_and_gt_mutations(self):
        prepared = copy.deepcopy(self.f.prepared); prepared[0]["completion_token_ids"] = [4]
        with self.assertRaises(ValueError): n.validate_inputs(self.f.plan, prepared, self.f.dataset)
        data = copy.deepcopy(self.f.dataset); data[0]["gt_answer"] = "changed"
        with self.assertRaises(ValueError): n.validate_inputs(self.f.plan, self.f.prepared, data)

    def test_rebound_carrier_changes_rejected(self):
        rows = copy.deepcopy(self.f.prepared); rows[0]["engine_prompt_token_ids"] = [1]
        self.f.save(self.f.pkg / "prepared_records.jsonl", self.f.lines(rows)); self.f.refreeze(build=False)
        with self.assertRaises(ValueError): self.f.build()

    def test_pending_package_and_bad_release(self):
        self.f.bundle["runnable"] = False; self.f.refreeze(build=False)
        with self.assertRaises(ValueError): self.f.build()
        self.f.bundle["runnable"] = True; self.f.refreeze()
        self.f.proof["producer_exit_verified"] = False
        self.f.save(self.f.proof_path, self.f.proof); self.f.proof_ref = n.ref(self.f.proof_path)
        with self.assertRaises(ValueError): self.f.build()

    def test_projection_mutation_and_wrong_runtime(self):
        p = self.f.pkg / "projection_audit.json"; audit = json.loads(p.read_text())
        audit["torch_version"] = "other"; self.f.save(p, audit); self.f.refreeze(build=False)
        with self.assertRaises(ValueError): self.f.build()

    def test_projection_missing_or_not_independently_reconstructed(self):
        p = self.f.pkg / "projection_independent_verification.json"; proof = json.loads(p.read_text())
        proof["reconstruction_bitwise_equal"] = False; self.f.save(p, proof); self.f.refreeze(build=False)
        with self.assertRaises(ValueError): self.f.build()

    def test_ledger_collision_staleness_caps_and_test_use(self):
        n.validate_against_ledger(self.f.plan, self.f.budget)
        for field, value in (("generation_requests", 2647), ("tf_requests", 9229), ("untouched_test_requests", 1)):
            b = copy.deepcopy(self.f.budget); b[field] = value
            with self.assertRaises(ValueError): n.validate_against_ledger(self.f.plan, b)
        b = copy.deepcopy(self.f.budget); b["phases"] = [{"generation_request_ids": [self.f.plan["requests"][0]["request_id"]]}]
        with self.assertRaises(ValueError): n.validate_against_ledger(self.f.plan, b)
        self.f.budget["generation_requests"] = 4096 - 739
        with self.assertRaises(ValueError): self.f.build()

    def test_source_checker_and_direct_cli_help(self):
        root = self.f.root / "source"
        source = Path(n.__file__).resolve().parents[3]
        for rel in (*n.SOURCES, "infra/gpu03/direction_discovery/no_loophole_plan.py"):
            self.f.save(root / rel, (source / rel).read_bytes())
        n.validate_source(self.f.plan, root)
        self.f.save(root / next(iter(n.SOURCES)), b"changed")
        with self.assertRaises(ValueError): n.validate_source(self.f.plan, root)
        result = subprocess.run([sys.executable, str(Path(n.__file__).resolve()), "--help"], cwd=self.f.root,
                                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__": unittest.main()
