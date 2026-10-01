"""Authored full740 metadata only; no model, generated code, systemd or GPU calls."""
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from infra.gpu03.direction_discovery import h100_evaluate as m
from infra.gpu03.direction_discovery.test_h100_no_loophole_protocol import Fixture as PlanFixture
from infra.gpu03.direction_discovery.test_no_loophole_eval import frozen_modules


class Fixture(PlanFixture):
    def __init__(self, root):
        super().__init__(root)
        self.base, self.evaluate = frozen_modules()
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(m.host, "OUTPUTS", self.root / "outputs"))
        self.stack.enter_context(patch.object(m.host, "UID", os.getuid()))
        m.host.OUTPUTS.mkdir()
        self.source = self.root / "source"
        repo = Path(__file__).resolve().parents[3]
        for name, digest in m.source_pins().items():
            source = repo / name
            if m.sha(source) != digest:
                if name.endswith("/evaluate.py"):
                    source = Path(self.evaluate.__file__)
                else:
                    source = repo / "artifacts/no_loophole_capability_review_20260908/source_package_v1/source" / name
            assert m.sha(source) == digest, name
            self.save(self.source / name, source.read_bytes())
        self.plan_file = self.save(self.root / "full_plan.json", self.hplan)
        self.generation_manifest = self.save(self.root / "generation.json", {})
        cpu_profile = self.save(self.root / "cpu-profile.json", {"available_cpus": list(range(192))})
        self.rows = [{**request, "result": {"completion": "authored text, never executed", "completion_token_ids": [1],
            "generated_token_ids": [1], "fixed_completion_prefix_token_count": 0, "stop_reason": "eos"}}
            for request in self.hplan["requests"]]
        self.gm = {"phase": "h100_no_loophole_capability", "generation_requests": 740,
            "request_plan": m.ref(self.plan_file), "source_root": str(self.source), "python": str(self.python),
            "output": str(self.root / "generation-results"), "host_profile": m.ref(cpu_profile),
            "source_qualification": {"path": "authored-source-proof", "sha256": "b"*64},
            "workers": [{"name": f"worker_{i:02d}"} for i in range(8)]}
        files = []
        for i in range(8):
            path = self.root / f"generation-results/worker_{i:02d}/results.jsonl"
            self.save(path, b"".join((m.canonical(r)+"\n").encode() for r in self.rows[i::8])); files.append(m.ref(path))
        self.proof = {"status": "independently_verified_h100_generation", "generation_requests": 740,
            "gpu_release_verified": True, "process_release_verified": True,
            "manifest_sha256": m.sha(self.generation_manifest), "request_ids": [r["request_id"] for r in self.rows],
            "result_files": files}
        self.proof_path = self.save(self.root / "generation-proof.json", self.proof)
        self.package = {"manifest": m.ref(self.generation_manifest), "verification": m.ref(self.proof_path)}
        inner = self.save(self.root / "qualifier-verification.json", {"status": "verified_authored_helper_qualification",
            "cases": 26, "policy": m.repair.POLICY})
        exit_ref = self.save(self.root / "qualifier-exit.json", {"returncode": 0, "timed_out": False,
            "child_reaped": True, "remaining_group_pids": []})
        self.helper = {"status": "independently_verified_h100_helper_classifier_qualification", "instance_id": m.host.INSTANCE,
            "helper_source_sha256": m.REPAIR_SHA, "sandbox_source_sha256": m.SANDBOX_SHA,
            "cases": 26, "all_expected_outcomes_match": True, "whole_program_gt_preserved": True,
            "process_release_verified": True, "qualifier_verification": m.ref(inner), "producer_exit": m.ref(exit_ref)}
        helper = self.save(self.root / "helper-proof.json", self.helper)
        self.spec = {"run_token": "authored-h100-evaluation", "stage": str(m.host.OUTPUTS / "authored-h100-evaluation"),
            "generation": self.package, "helper_qualification": m.ref(helper)}
        self.stack.enter_context(patch.object(m.host, "load_manifest", return_value=self.gm))
        self.stack.enter_context(patch.object(m.host, "verify", side_effect=lambda *a: self.proof))
        self.stack.enter_context(patch.object(m, "base_module", return_value=self.evaluate))

    def close(self):
        if hasattr(self, "stack"): self.stack.close()
        super().close()

    def build_evaluation(self):
        result = m.build(self.spec)
        return result, m.load_manifest(result["manifest"]["path"], result["manifest"]["sha256"])[0]

    def continuation(self):
        # Only the producer-verification boundary is injected. The actual740
        # plan, retained snapshots, consumer source and input gates remain real.
        reference = m.ref(self.save(self.root / "continuation.json", {"purpose": "authored-five-row-provenance"}))
        physical = []
        for i in range(8):
            path = self.root / f"generation-results/worker_{i:02d}/results.jsonl"
            self.save(path, b"".join((m.canonical(r)+"\n").encode() for r in self.rows[5:][i::8]))
            physical.append(m.ref(path))
        logical = self.save(self.root / "generation-results/logical_results.jsonl",
                            b"".join((m.canonical(r)+"\n").encode() for r in self.rows))
        self.gm.update(generation_requests=735, scientific_generation_requests=740, continuation=reference)
        self.proof.update(new_generation_requests=735, reused_generation_requests=5, continuation=reference,
                          physical_result_files=physical, result_files=[m.ref(logical)])
        self.save(self.proof_path, self.proof); self.package["verification"] = m.ref(self.proof_path)

    def evaluation_rows(self):
        rows = []
        for generation in self.rows:
            metrics = {"ground_truth_correctness": True, "harmful_modification": False,
                "strict_reward_hack": False, "attempted_hack": False, "completion_length": 1}
            side = {"policy": m.repair.POLICY, "legacy_policy": m.repair.LEGACY_POLICY,
                    "legacy_row_unchanged": True, "whole_program_gt_reused": True, "status": "unchanged", "metrics": metrics.copy()}
            rows.append({**{k: generation[k] for k in m.IDENTITY}, "generation": generation["result"],
                "generation_sha256": hashlib.sha256(m.canonical(generation).encode()).hexdigest(),
                "evaluation_status": "evaluated", "metrics": metrics, "helper_aware_evaluation": side})
        return rows


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name).resolve()); self.addCleanup(self.f.close)

    def test_real_h100_plan_to_full740_builder_and_input_verification(self):
        result, manifest = self.f.build_evaluation()
        self.assertEqual(len(manifest["request_ids"]), 740)
        self.assertFalse(result["launch_performed"])
        self.assertEqual(len(manifest["input_files"]), 12)

    def test_verified_735_new_plus_five_reused_builds_exact740(self):
        self.f.continuation()
        result, manifest = self.f.build_evaluation()
        self.assertEqual(len(manifest["request_ids"]), 740)
        rows = m.json_rows(m.ref(Path(manifest["stage"]) / "input/generation.jsonl"))
        self.assertEqual(rows, self.f.rows)
        self.assertEqual([len(m.json_rows(m.ref(Path(manifest["stage"]) / f"input/worker_{i:02d}.jsonl")))
                          for i in range(8)], [93,93,93,93,92,92,92,92])
        self.assertFalse(result["launch_performed"])

    def test_735_without_explicit_continuation_is_rejected_before_verifier(self):
        self.f.gm["generation_requests"] = 735
        with patch.object(m.host, "verify", side_effect=AssertionError("producer rows reached")):
            with self.assertRaisesRegex(ValueError, "scientific740"):
                m.generation_context(self.f.package)

    def test_continuation_requires_exact_logical_count_and_proof_lineage(self):
        self.f.continuation()
        for field, value in (("new_generation_requests",734),("reused_generation_requests",4),
                             ("continuation",{"path":"/wrong","sha256":"0"*64})):
            changed = {**self.f.proof, field:value}
            path = self.f.save(self.f.root / (field+"-bad-proof.json"), changed)
            package = {**self.f.package,"verification":m.ref(path)}
            with self.subTest(field=field), patch.object(m.host,"verify",side_effect=AssertionError("producer rows reached")):
                with self.assertRaisesRegex(ValueError,"physical/reused counts"):
                    m.generation_context(package)
        self.f.gm["scientific_generation_requests"] = 739
        with self.assertRaisesRegex(ValueError,"five-reused scientific740"):
            m.generation_context(self.f.package)

    def test_continuation_rejects_failed_proof_before_collector(self):
        self.f.continuation(); self.f.proof["process_release_verified"] = False
        self.f.save(self.f.proof_path,self.f.proof); self.f.package["verification"] = m.ref(self.f.proof_path)
        with patch.object(m,"collect_generation",side_effect=AssertionError("outcomes reached")):
            with self.assertRaisesRegex(ValueError,"terminal/release"):
                m.build(self.f.spec)
        self.assertFalse(Path(self.f.spec["stage"]).exists())

    def test_continuation_requires_exact_logical_and_physical_paths(self):
        self.f.continuation()
        for key in ("result_files","physical_result_files"):
            original = copy.deepcopy(self.f.proof[key])
            self.f.proof[key][0]["path"] = str(self.f.root / "other.jsonl")
            self.f.save(self.f.proof_path,self.f.proof); self.f.package["verification"] = m.ref(self.f.proof_path)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,"result paths"):
                m.generation_context(self.f.package)
            self.f.proof[key] = original

    def test_continuation_merged_snapshot_retains_full_unique_requests(self):
        self.f.continuation()
        path = Path(self.f.proof["result_files"][0]["path"])
        for rows in (self.f.rows[1:], [self.f.rows[0],*self.f.rows[:-1]]):
            self.f.save(path,b"".join((m.canonical(r)+"\n").encode() for r in rows))
            self.f.proof["result_files"] = [m.ref(path)]
            with self.assertRaises(ValueError):
                m.collect_generation(self.f.gm,self.f.hplan,self.f.proof)

    def test_complete_proof_before_any_collector_or_publication(self):
        f = self.f; f.proof["process_release_verified"] = False
        f.save(f.proof_path, f.proof); f.package["verification"] = m.ref(f.proof_path)
        with patch.object(m, "collect_generation", side_effect=AssertionError("outcomes reached")):
            with self.assertRaisesRegex(ValueError, "terminal/release"):
                m.build(f.spec)
        self.assertFalse(Path(f.spec["stage"]).exists())

    def test_qualification_six_calls_cannot_enter(self):
        self.f.gm["phase"] = "h100_numerical_qualification"
        with self.assertRaisesRegex(ValueError, "scientific740"):
            m.generation_context(self.f.package)

    def test_external_and_fresh_proofs_must_match(self):
        self.f.proof["different"] = True
        with self.assertRaisesRegex(ValueError, "saved proof"):
            m.generation_context(self.f.package)

    def test_partial_request_union_rejected_before_host_verifier(self):
        f = self.f; f.proof["request_ids"].pop(); f.save(f.proof_path, f.proof)
        f.package["verification"] = m.ref(f.proof_path)
        with patch.object(m.host, "verify", side_effect=AssertionError("premature verifier")):
            with self.assertRaises(ValueError): m.generation_context(f.package)

    def test_foreign_result_path_rejected(self):
        f = self.f; f.proof["result_files"][0]["path"] = str(f.root / "other.jsonl")
        f.save(f.proof_path, f.proof); f.package["verification"] = m.ref(f.proof_path)
        with self.assertRaisesRegex(ValueError, "exact producer"):
            m.generation_context(f.package)

    def test_changed_result_seed_rejected_even_rebound(self):
        f = self.f; row = copy.deepcopy(f.rows[0]); row["seed"] += 1
        path = Path(f.proof["result_files"][0]["path"])
        f.save(path, b"".join((m.canonical(r)+"\n").encode() for r in [row,*f.rows[8::8]]))
        f.proof["result_files"][0] = m.ref(path)
        with self.assertRaisesRegex(ValueError, "request was changed"):
            m.collect_generation(f.gm, f.hplan, f.proof)

    def test_duplicate_generation_rejected(self):
        f = self.f; path = Path(f.proof["result_files"][0]["path"])
        f.save(path, b"".join((m.canonical(r)+"\n").encode() for r in [f.rows[0],*f.rows[0::8]]))
        f.proof["result_files"][0] = m.ref(path)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            m.collect_generation(f.gm, f.hplan, f.proof)

    def test_helper_actual_exit_and_policy_are_required(self):
        for field, value in (("cases",25),("whole_program_gt_preserved",False),("sandbox_source_sha256","0"*64)):
            q = {**self.f.helper, field:value}
            with self.assertRaises(ValueError): m.check_helper_qualification(q)
        q = copy.deepcopy(self.f.helper)
        path = self.f.save(self.f.root / "bad-exit.json", {"returncode":1,"timed_out":False,"child_reaped":True,"remaining_group_pids":[]})
        q["producer_exit"] = m.ref(path)
        with self.assertRaisesRegex(ValueError, "exit/release"): m.check_helper_qualification(q)

    def test_no_output_reuse(self):
        self.f.build_evaluation()
        with self.assertRaisesRegex(ValueError, "fresh exact-token"):
            m.build(self.f.spec)

    def test_rebound_shard_mutation_fails(self):
        result, manifest = self.f.build_evaluation(); stage=Path(manifest["stage"])
        rows=m.json_rows(m.ref(stage / "input/worker_00.jsonl")); rows.pop()
        self.f.save(stage / "input/worker_00.jsonl", b"".join((m.canonical(r)+"\n").encode() for r in rows))
        manifest["input_files"] = m.host.inventory(stage / "input")
        self.f.save(Path(result["manifest"]["path"]),manifest)
        with self.assertRaisesRegex(ValueError,"shard"):
            m.load_manifest(result["manifest"]["path"],m.sha(result["manifest"]["path"]))

    def test_exact_service_bounds_and_clean_worker_environment(self):
        result, manifest=self.f.build_evaluation(); command=result["command"]
        self.assertIn("--property=RuntimeMaxSec=7290",command)
        self.assertIn("--property=MemoryMax=32G",command)
        self.assertIn("--property=CPUAffinity=112-127",command)
        self.assertIn("CUDA_VISIBLE_DEVICES=",command)
        self.assertFalse(any("DBUS_SESSION_BUS_ADDRESS=" in x for x in command))
        self.assertIn("--run",command)

    def test_full740_evaluation_identity_and_gt_preservation(self):
        rows=self.f.evaluation_rows(); m.validate_evaluations(rows,self.f.rows)
        rows[0]["helper_aware_evaluation"]["metrics"]["ground_truth_correctness"] = False
        with self.assertRaisesRegex(ValueError,"whole-program GT"):
            m.validate_evaluations(rows,self.f.rows)

    def test_unknown_status_and_partial_evaluations_fail(self):
        rows=self.f.evaluation_rows()
        with self.assertRaisesRegex(ValueError,"Partial"): m.validate_evaluations(rows[:-1],self.f.rows)
        rows[0]["helper_aware_evaluation"]["status"]="infrastructure_failure"
        with self.assertRaisesRegex(ValueError,"failed versioned"): m.validate_evaluations(rows,self.f.rows)

    def test_plain_coordinator_never_executes_generated_code(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"","CODE_EVAL_SANDBOX":""}):
            with self.assertRaisesRegex(ValueError,"restricted CPU"):
                m.inside("unopened","unopened","unopened","uncreated")

    def test_saved_primitive_replay_uses_real_frozen_classifier(self):
        from infra.gpu03.direction_discovery.test_helper_aware_evaluation_v1 import row, primitive, load_frozen
        value = row(); value["metrics"]["completion_length"] = 1
        value["transport"] = dict.fromkeys(m.repair.TRANSPORT, 0)
        evaluator=SimpleNamespace(evaluator=SimpleNamespace(batch_evaluate=lambda _: [primitive(True),primitive(False)]))
        installation=SimpleNamespace(report=lambda: dict.fromkeys(m.repair.TRANSPORT,0))
        with patch.object(m.repair,"modules",return_value=load_frozen()):
            value["helper_aware_evaluation"] = m.repair.score_legacy_row(value,evaluator,installation)
            m.replay_labels([value])
            value["helper_aware_evaluation"]["metrics"]["harmful_modification"] = True
            with self.assertRaisesRegex(ValueError,"sidecar differs"):
                m.replay_labels([value])

    def test_changed_legacy_metric_is_rejected_by_primitive_replay(self):
        from infra.gpu03.direction_discovery.test_helper_aware_evaluation_v1 import row
        value=row(); value["transport"]={}; value["metrics"]["completion_length"]=1
        value["metrics"]["ground_truth_correctness"] = False
        with self.assertRaisesRegex(ValueError,"Legacy saved primitive"):
            m.replay_labels([value])

    def test_authored_full_worker_through_frozen_evaluator_and_helper_classifier(self):
        from infra.gpu03.direction_discovery.test_helper_aware_evaluation_v1 import row, primitive, load_frozen
        f=self.f; values=f.rows[0::8]
        requests=f.save(f.root/"worker-input.jsonl",b"".join((m.canonical(r)+"\n").encode() for r in values))
        saved=row()["repository_evaluation"]
        evaluator=SimpleNamespace(evaluate=lambda *args:copy.deepcopy(saved),
            evaluator=SimpleNamespace(batch_evaluate=lambda _: [primitive(True),primitive(False)]))
        installation=SimpleNamespace(report=lambda:dict.fromkeys(m.repair.TRANSPORT,0),restore=lambda:None)
        original_is_file=Path.is_file
        original_exists=Path.exists
        def visible(path):
            return str(path)=="/work/src/evaluate/helpers.py" or original_is_file(path)
        def isolated_exists(path):
            # This authored test runs on the host; model the two paths hidden by
            # the real outer sandbox even when the test host is the H100 machine.
            return False if str(path) in ("/scratch", "/home/ubuntu/h100-workspace") else original_exists(path)
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"","CODE_EVAL_SANDBOX":"bwrap"}), \
                patch.object(Path,"is_file",visible), patch.object(Path,"exists",isolated_exists), \
                patch.object(f.evaluate,"make_repository_evaluator",return_value=evaluator), \
                patch.object(m.sandbox,"install_bounded_evaluator",return_value=installation), \
                patch.object(f.evaluate,"install_count_payload_guard"), patch.object(m.repair,"modules",return_value=load_frozen()):
            output=f.root/"worker-evaluation"
            result=m.inside(requests,f.hplan["prepared_records"],f.hplan["dataset"],output)
            self.assertEqual(result["records"],93)
            rows=m.json_rows(m.ref(output/"records.jsonl"),immutable=False)
            m.validate_evaluations(rows,values); m.replay_labels(rows)
            self.assertTrue(all(r["metrics"]["ground_truth_correctness"] is True for r in rows))

    def test_worker_rejects_visible_host_paths_before_loading_inputs(self):
        for exposed in ("/scratch", "/home/ubuntu/h100-workspace"):
            with self.subTest(exposed=exposed), \
                    patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"","CODE_EVAL_SANDBOX":"bwrap"}), \
                    patch.object(Path,"is_file",lambda path: str(path)=="/work/src/evaluate/helpers.py"), \
                    patch.object(Path,"exists",lambda path: str(path)==exposed), \
                    patch.object(self.f.evaluate,"jsonl",side_effect=AssertionError("inputs reached before isolation gate")):
                with self.assertRaisesRegex(ValueError,"restricted CPU outer sandbox"):
                    m.inside("unopened","unopened","unopened","uncreated")


if __name__ == "__main__": unittest.main()
