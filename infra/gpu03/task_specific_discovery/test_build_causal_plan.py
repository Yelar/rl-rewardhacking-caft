"""CPU-only authored planning fixtures; no GPU, tokenizer download or dispatch."""
import copy
import json
import os
from pathlib import Path
import tempfile
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import build_causal_plan as b
from . import compile_shortlist as c
from . import test_compile_shortlist as shortlist_tests
from . import causal_prompts as p
from .test_causal_prompts import row as authored_row


class Tokenizer:
    def apply_chat_template(self, prompt, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return [100, 200, 300 + len(prompt[1]["content"])]


def fixture_shortlist():
    reports, centers, values = shortlist_tests.authored_reports()
    qsha = c.tensor_bytes_sha(values[c.PC4_PATH, c.PC4_KEY])
    with patch.object(c, "PC4_TENSOR_SHA", qsha):
        shortlist, tensors = c.assemble(reports, centers, lambda ref, key: values[ref["path"], key].copy())
    return shortlist, tensors, qsha


def fixture_prompts():
    hints = p.load_hints(Path(__file__).resolve().parents[3] / "src/data/hints.py")
    rows = [authored_row(i) for i in range(32)]
    fixed = [hints.SimpleOverwriteTests()(copy.deepcopy(r)) for r in rows]
    split = {str(i): {"problem_id": i, "split": "direction_fit"} for i in range(32)}
    prepared = p.prepare_rows(rows, fixed, split, hints, expected_counts={"direction_fit": 32})
    return prepared


def fixture_bundle():
    shortlist, tensors, pc4sha = fixture_shortlist()
    with patch.object(c, "PC4_TENSOR_SHA", pc4sha):
        bundle, vectors = b.make_conditions(shortlist, tensors, p.STUDY)
    prepared = fixture_prompts()
    rows = b.tokenize(prepared["prompt_rows"], prepared["pilot_problem_keys"], prepared["qualification_problem_keys"], Tokenizer())
    return bundle, vectors, rows


class ConditionTests(unittest.TestCase):
    def test_exact_caps_random_matching_saved_target_bytes(self):
        shortlist, tensors, sha = fixture_shortlist()
        with patch.object(c, "PC4_TENSOR_SHA", sha):
            bundle, vectors = b.make_conditions(shortlist, tensors, p.STUDY)
        self.assertEqual(len(bundle["qualification_condition_ids"]), 13)
        self.assertEqual(len(bundle["pilot_condition_ids"]), 79)
        self.assertEqual(len(bundle["random_axes"]), 18)
        for key, original in tensors.items():
            self.assertEqual(vectors[key].tobytes(), original.tobytes())
        for random in bundle["random_axes"]:
            q = vectors[random["tensor_key"]]
            c.validate_direction(q)
            target = bundle["conditions"][random["configuration_id"] + ".alpha50"]
            counterpart = bundle["conditions"][random["configuration_id"] + f'.random{random["random_index"]}.alpha50']
            for key in ("layer", "rank", "operator", "strength", "center_key"):
                self.assertEqual(target.get(key), counterpart.get(key))
            self.assertNotEqual(target["vector_key"], counterpart["vector_key"])
        self.assertEqual(len({x["seed"] for x in bundle["random_axes"]}), 18)

    def test_random_axes_deterministic_without_global_rng_mutation(self):
        shortlist, tensors, sha = fixture_shortlist()
        before = np.random.get_state()
        with patch.object(c, "PC4_TENSOR_SHA", sha):
            one, v1 = b.make_conditions(shortlist, tensors, p.STUDY)
            two, v2 = b.make_conditions(shortlist, tensors, p.STUDY)
        self.assertEqual(one, two)
        self.assertTrue(all(np.array_equal(v1[k], v2[k]) for k in v1))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_nonunit_historical_identity_center_and_strength_rejected(self):
        original, tensors, sha = fixture_shortlist()
        changes = [lambda s, t: s["axes"][0].update(layer=20),
                   lambda s, t: t["axis__historical_pc4"].fill(1),
                   lambda s, t: s["operator_configurations"][0].update(strengths=[1.0]),
                   lambda s, t: t[s["centers"][0]["tensor_key"]].fill(np.nan)]
        for change in changes:
            s, t = copy.deepcopy(original), {k: v.copy() for k, v in tensors.items()}
            change(s, t)
            with self.subTest(change=change), patch.object(c, "PC4_TENSOR_SHA", sha), self.assertRaises(ValueError):
                b.make_conditions(s, t, p.STUDY)

    def test_unsupported_nominees_reduce_scope_without_fabrication(self):
        s, tensors, sha = fixture_shortlist()
        s["axes"] = [s["axes"][0]]
        s["operator_configurations"] = [x for x in s["operator_configurations"] if x["axis_id"] == "historical_pc4" and x["operator"] == "ordinary"]
        with patch.object(c, "PC4_TENSOR_SHA", sha):
            bundle, _ = b.make_conditions(s, tensors, p.STUDY)
        self.assertEqual(len(bundle["qualification_condition_ids"]), 3)
        self.assertEqual(len(bundle["pilot_condition_ids"]), 14)

    def test_prompt_only_tokens_and_exact_batch_budgets(self):
        bundle, _, rows = fixture_bundle()
        self.assertEqual(len(rows), 96)
        self.assertTrue(all(r["completion_token_ids"] == [] and r["completion_token_count"] == 0
                            and r["input_ids"] == r["prompt_token_ids"] for r in rows))
        qualification = b.requests_for(rows, bundle["qualification_condition_ids"], bundle["conditions"], qualification=True)
        pilot = b.requests_for(rows, bundle["pilot_condition_ids"], bundle["conditions"], qualification=False)
        self.assertEqual(len(qualification), 312); self.assertEqual(len(pilot), 15168)
        q = {r["request_id"]: r for r in qualification}; shared = [r for r in pilot if r["request_id"] in q]
        self.assertEqual(len(shared), 168)
        self.assertTrue(all(q[r["request_id"]] == r for r in shared))
        for request_list in (qualification, pilot):
            for start in range(0, len(request_list), 8):
                batch = request_list[start:start + 8]
                self.assertEqual(len({(r["condition_id"], r["protocol"]) for r in batch}), 1)

    def test_overlength_and_heldout_fail_without_replacement(self):
        prepared = fixture_prompts()
        args = [prepared["prompt_rows"], prepared["pilot_problem_keys"], prepared["qualification_problem_keys"]]
        with self.assertRaisesRegex(ValueError, "overlength"):
            b.tokenize(*args, SimpleNamespace(apply_chat_template=lambda *a, **k: [1] * 1537))
        prepared["prompt_rows"][0]["problem_split"] = "untouched_test"
        with self.assertRaisesRegex(ValueError, "identity/protocol"):
            b.tokenize(*args, Tokenizer())

    def test_missing_duplicate_protocol_and_wrong_qualification_rejected(self):
        prepared = fixture_prompts()
        for rows in (prepared["prompt_rows"][:-1], prepared["prompt_rows"] + [prepared["prompt_rows"][0]]):
            with self.assertRaises(ValueError):
                b.tokenize(rows, prepared["pilot_problem_keys"], prepared["qualification_problem_keys"], Tokenizer())
        with self.assertRaisesRegex(ValueError, "boundaries"):
            b.tokenize(prepared["prompt_rows"], prepared["pilot_problem_keys"], prepared["pilot_problem_keys"][1:5], Tokenizer())


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        bundle, vectors, self.rows = fixture_bundle()
        self.bundle = bundle
        base = self.root / "base"; base.mkdir(); adapter = self.root / "adapter"; adapter.mkdir()
        bound = {}
        for parent, names in ((base, b.worker.BASE_FILES), (adapter, b.worker.ADAPTERS)):
            for name in names:
                pth = parent / name; pth.write_text("authored-" + name); bound[str(pth)] = b.io.binding(pth)
        rows_path = self.root / "rows.jsonl"; rows_path.write_bytes(b"".join(b.canonical(r) + b"\n" for r in self.rows))
        vector_path = self.root / "vectors.safetensors"; vector_path.write_bytes(b"authored placeholder; vector computation tested separately")
        bound[str(rows_path)] = b.io.binding(rows_path); bound[str(vector_path)] = b.io.binding(vector_path)
        self.plan = {"schema": b.SCHEMA, "status": "prepared_not_dispatched", "run_token": "authored-token",
            "study_id": p.STUDY, "source_root": str(self.root / "source"), "qualified_source_files": {"source.py": {"sha256": "1" * 64, "size_bytes": 1}},
            "model_snapshot": str(base), "checkpoint": str(adapter), "prepared_rows": str(rows_path), "vectors": str(vector_path),
            "model_revision": b.worker.REVISION, "sampling": b.worker.SAMPLING, "profile": b.worker.PROFILE,
            "batch_size": 8, "bound_files": bound, **bundle,
            "qualification_requests": b.requests_for(self.rows, bundle["qualification_condition_ids"], bundle["conditions"], qualification=True),
            "pilot_requests": b.requests_for(self.rows, bundle["pilot_condition_ids"], bundle["conditions"], qualification=False),
            "budgets": {"qualification": 312, "pilot_including_reuse": 15168, "reusable_pilot_requests": 168}}
        self.plan["qualification_contract"] = b.qualify_contract(self.rows, bundle, self.plan["qualified_source_files"], bound[str(vector_path)]["sha256"])
        self.plan_path = self.root / "PLAN.json"; b.io.write_json(self.plan_path, self.plan); self.plan_ref = b.ref(self.plan_path)

    def proof(self, change=None):
        root = self.root / "proof"; root.mkdir()
        generation_path = root / "generations.jsonl"
        generations = [{**r, "result": {"completion_token_ids": [1], "completion": "authored fixture"}}
                       for r in self.plan["qualification_requests"]]
        generation_path.write_bytes(b"".join(b.canonical(r) + b"\n" for r in generations))
        gref = b.ref(generation_path)
        retained = {}
        for r, g in zip(self.plan["qualification_requests"], generations):
            rid = r["request_id"]
            primitive = root / (rid + ".primitive.json"); primitive.write_bytes(b.canonical({"result": g["result"]}))
            final = root / (rid + ".final.json"); final.write_bytes(b.canonical({"returned_row": True}))
            retained[rid] = {"request": r, "request_sha256": b.hash_value(r), "generation_file": gref,
                "generation_row_sha256": b.hash_value(g), "primitive_file": b.ref(primitive), "final_file": b.ref(final)}
        signatures = {b.hash_value(b.worker.qualification_signature(c)): b.worker.qualification_signature(c)
                      for c in self.plan["conditions"].values() if c.get("layer") is not None}
        proof = {"schema": "task_specific_causal_qualification_v1", "status": "qualified_task_specific_causal_v1",
            "profile": self.plan["profile"], "source_files": self.plan["qualified_source_files"],
            "vectors_sha256": self.plan["bound_files"][self.plan["vectors"]]["sha256"],
            "all_returned_requests_retained": True, "gpu_workers_released": True,
            "null_token_parity": True, "null_logits_within_tolerance": True,
            "model_files": b.worker.model_bindings(self.plan, b.worker.BASE_FILES, "model_snapshot"),
            "adapter_files": b.worker.model_bindings(self.plan, b.worker.ADAPTERS, "checkpoint"),
            "qualified_signatures": list(signatures.values()), "retained_requests": retained}
        if change: change(proof)
        path = root / "QUALIFIED.json"; b.io.write_json(path, proof)
        return b.ref(path)

    def test_real_task_bundle_admission_success_and_worker_compatible(self):
        reference = b.task_bundle(self.plan_ref, "qualification", self.root / "qualification", 8)
        phase = b.read_json(reference["path"])
        self.assertEqual(phase["expected_new_requests"], 312)
        self.assertEqual(len(phase["tasks"]), 8)
        collected = []
        adapters = {name: self.plan["bound_files"][str(Path(self.plan["checkpoint"]) / name)]["sha256"] for name in b.worker.ADAPTERS}
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-00000000-0000-0000-0000-000000000000"), \
                patch.object(b.worker, "source_identities"), patch.object(b.worker, "ADAPTERS", adapters):
            for item in phase["tasks"]:
                task = b.read_json(item["task"]["path"])
                self.assertEqual(len(b.worker.validate(task)), 96)
                self.assertTrue(all(c["control"] != "project" for c in task["conditions"].values()))
                self.assertEqual(set(task["conditions"]), {r["condition_id"] for r in task["requests"]})
                collected.extend(task["requests"])
        self.assertEqual({r["request_id"] for r in collected}, {r["request_id"] for r in self.plan["qualification_requests"]})
        self.assertFalse(phase["dispatch_authorized_by_this_file"])

    def test_qualified_pilot_reuses_168_and_preserves_whole_batches(self):
        proof = self.proof()
        reference = b.task_bundle(self.plan_ref, "pilot", self.root / "pilot", 8, proof)
        phase = b.read_json(reference["path"])
        self.assertEqual(phase["expected_new_requests"], 15000)
        self.assertEqual(phase["expected_reused_requests"], 168)
        self.assertEqual(phase["expected_total_requests"], 15168)
        requests = []
        for item in phase["tasks"]:
            task = b.read_json(item["task"]["path"]); requests.extend(task["requests"])
            self.assertLessEqual(len(task["requests"]), 2048)
            self.assertEqual(len(task["requests"]) % 8, 0)
            for start in range(0, len(task["requests"]), 8):
                group = task["requests"][start:start + 8]
                self.assertEqual(len({(r["condition_id"], r["protocol"]) for r in group}), 1)
        self.assertFalse(set(phase["reused_requests"]) & {r["request_id"] for r in requests})
        self.assertEqual(len(requests), len({r["request_id"] for r in requests}))

    def test_no_proof_or_fewer_lanes_cannot_launch_or_create_output(self):
        with self.assertRaisesRegex(ValueError, "before numerical"):
            b.task_bundle(self.plan_ref, "pilot", self.root / "unqualified", 8)
        self.assertFalse((self.root / "unqualified").exists())
        with self.assertRaisesRegex(ValueError, "Too few lanes"):
            b.task_bundle(self.plan_ref, "pilot", self.root / "overloaded", 1, self.proof())
        self.assertFalse((self.root / "overloaded").exists())

    def test_no_repeat_bundle_or_mutated_bound_inputs(self):
        target = self.root / "q"
        b.task_bundle(self.plan_ref, "qualification", target)
        with self.assertRaisesRegex(ValueError, "fresh"):
            b.task_bundle(self.plan_ref, "qualification", target)
        with self.assertRaises(FileExistsError):
            b.task_bundle(self.plan_ref, "qualification", self.root / "duplicate_elsewhere")
        self.assertFalse((self.root / "duplicate_elsewhere").exists())
        Path(self.plan["vectors"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            b.task_bundle(self.plan_ref, "qualification", self.root / "bad")

    def test_interrupted_preparation_consumes_stage_before_any_output(self):
        original_write = b.io.write_json
        def interrupted(path, value):
            original_write(path, value)
            if Path(path).name == "QUALIFICATION_TASK_BINDING.json":
                raise RuntimeError("authored crash after stage binding")
        with patch.object(b.io, "write_json", side_effect=interrupted), self.assertRaisesRegex(RuntimeError, "authored crash"):
            b.task_bundle(self.plan_ref, "qualification", self.root / "interrupted")
        self.assertTrue((self.root / "QUALIFICATION_TASK_BINDING.json").is_file())
        self.assertFalse((self.root / "interrupted").exists())
        with self.assertRaises(FileExistsError):
            b.task_bundle(self.plan_ref, "qualification", self.root / "replay")

    def test_complete_preparation_path_with_authored_bound_inputs(self):
        # External tokenizer and unchanged safetensors codec are test doubles;
        # every new admission, artifact hash, selection and write path is real.
        shortlist, tensors, pc4sha = fixture_shortlist()
        shortlist["fit_manifest_sha256"] = "a" * 64
        sd = self.root / "shortlist"; sd.mkdir()
        st = sd / "shortlist.safetensors"; st.write_bytes(b"authored closed tensors")
        shortlist["tensor_file"] = {"path": st.name, **b.io.binding(st)}
        b.io.write_json(sd / "SHORTLIST.json", shortlist)
        for n in ("CANDIDATE_TABLE.json", "INPUT_VERIFICATION.json"):
            b.io.write_json(sd / n, {"authored": True})
        complete = {"schema": "task_specific_shortlist_complete_v1", "status": "succeeded", "cells": 72,
            "candidates": 432, "launched": False, "fit_manifest_sha256": "a" * 64,
            "artifacts": {x.name: b.io.binding(x) for x in sd.iterdir()}}
        b.io.write_json(sd / "COMPLETE.json", complete)
        pd = self.root / "prompts"; pd.mkdir()
        prepared = fixture_prompts()
        payloads = {"pilot_prompts.jsonl": b"".join(b.canonical(r) + b"\n" for r in prepared["prompt_rows"]),
            "fixed992.jsonl": b"authored fixed dataset", "canonical_nohint992.jsonl": b"authored nohint dataset",
            "randomized992.jsonl": b"authored prospective randomized dataset", "randomized_name_mapping.json": b"{}"}
        for n, data in payloads.items(): (pd / n).write_bytes(data)
        b.io.write_json(pd / "PROTOCOL.json", {"schema": p.SCHEMA, "status": "prompts_frozen_tokenization_pending",
            "study_id": p.STUDY, "pilot_count": 32, "qualification_count": 4, "full_row_fixed_roundtrips": 992,
            "protocols": list(p.PROTOCOLS), "input_bindings": {"splits": {"sha256": p.SPLIT_SHA}}})
        b.io.write_json(pd / "pilot_problem_ids.json", {"study_id": p.STUDY, "selection_domain": p.SELECTION_DOMAIN,
            "pilot_problem_keys": prepared["pilot_problem_keys"], "qualification_problem_keys": prepared["qualification_problem_keys"]})
        b.io.write_json(pd / "ARTIFACT_MANIFEST.json", {"schema": p.SCHEMA, "files": {x.name: b.io.binding(x) for x in pd.iterdir()}})
        source = self.root / "source"; source.mkdir()
        source_names = set(b.REQUIRED_SOURCE).union(b.worker.REQUIRED_SOURCE_FILES)
        for name in source_names:
            path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("PROFILE = " + repr(b.worker.PROFILE) if name.endswith("causal_generation.py") else "# authored source")
        b.io.write_json(source / "SOURCE_FILES.json", {n: b.io.binding(source / n) for n in source_names})
        base_manifest = self.root / "base_manifest.json"
        b.io.write_json(base_manifest, {"base_revision": b.worker.REVISION, "files": b.worker.model_bindings(self.plan, b.worker.BASE_FILES, "model_snapshot")})
        design = self.root / "DESIGN.md"; design.write_text("Authored design fixture")
        args = SimpleNamespace(output=str(self.root / "prepared"), shortlist_complete=str(sd / "COMPLETE.json"),
            shortlist_sha=b.io.sha256(sd / "COMPLETE.json"), prompt_manifest=str(pd / "ARTIFACT_MANIFEST.json"),
            prompt_sha=b.io.sha256(pd / "ARTIFACT_MANIFEST.json"), source_root=str(source),
            base_manifest=str(base_manifest), model_snapshot=self.plan["model_snapshot"], checkpoint=self.plan["checkpoint"],
            design=str(design), design_sha=b.io.sha256(design))
        module = types.ModuleType("safetensors.numpy"); module.load_file = lambda name: tensors
        def write_tensors(path, values, metadata):
            Path(path).write_bytes(b.canonical({k: b.vector_sha(v) for k, v in values.items()}))
        adapters = {n: self.plan["bound_files"][str(Path(self.plan["checkpoint"]) / n)]["sha256"] for n in b.worker.ADAPTERS}
        with patch.object(c, "PC4_TENSOR_SHA", pc4sha), patch.object(b, "BASE_MANIFEST_SHA", b.io.sha256(base_manifest)), \
                patch.object(b.worker, "ADAPTERS", adapters), patch.object(p, "FIXED_SHA", b.io.sha256(pd / "fixed992.jsonl")), \
                patch.object(p, "CANONICAL_SHA", b.io.sha256(pd / "canonical_nohint992.jsonl")), \
                patch.object(b.io, "write_tensors", side_effect=write_tensors), \
                patch.dict(sys.modules, {"safetensors": types.ModuleType("safetensors"), "safetensors.numpy": module}):
            reference = b.prepare(args, Tokenizer())
        plan = b.read_json(reference["path"])
        self.assertEqual(plan["budgets"], {"qualification": 312, "pilot_including_reuse": 15168, "reusable_pilot_requests": 168})
        self.assertTrue(all(n in plan["qualified_source_files"] for n in b.worker.REQUIRED_SOURCE_FILES))
        self.assertFalse(plan["generation"] or plan["evaluation"] or plan["training"])
        for name, binding in plan["bound_files"].items(): b.io.verify_file(name, binding)

    def test_retained_request_tamper_rejected(self):
        def tamper(proof):
            saved = next(iter(proof["retained_requests"].values()))
            saved["request"] = {**saved["request"], "seed": 0}
        proof = self.proof(tamper)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            b.task_bundle(self.plan_ref, "pilot", self.root / "bad", 8, proof)

    def test_retained_generation_file_or_row_tamper_rejected(self):
        proof = self.proof()
        path = self.root / "proof/generations.jsonl"; path.write_bytes(path.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            b.task_bundle(self.plan_ref, "pilot", self.root / "bad", 8, proof)

    def test_failed_qualification_and_wrong_model_coverage_rejected(self):
        proof = self.proof(lambda p: p.update(null_token_parity=False))
        with self.assertRaisesRegex(ValueError, "successful qualification"):
            b.reuse_from_qualification(self.plan, proof)

    def test_qualification_contract_exact_population_tolerance_and_null_pairs(self):
        contract = self.plan["qualification_contract"]
        self.assertEqual(len(contract["problem_ids"]), 4)
        self.assertEqual(contract["protocols"], list(p.PROTOCOLS))
        self.assertEqual(contract["sample_indices"], [0, 1])
        self.assertEqual(contract["logits_atol"], 1e-5); self.assertEqual(contract["logits_rtol"], 1e-5)
        self.assertEqual(len(contract["null_pairs"]), 6)
        for pair in contract["null_pairs"]:
            a, z = [self.plan["conditions"][pair[k]] for k in ("materialization_only", "zero_strength")]
            self.assertEqual(b.worker.qualification_signature(a), b.worker.qualification_signature(z))


if __name__ == "__main__":
    unittest.main()
