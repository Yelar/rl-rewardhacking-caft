"""Authored portability fixtures only; no actual model, tokenizer or host calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.direction_discovery import no_loophole_plan as original
from infra.gpu03.direction_discovery import h100_no_loophole_protocol as m
from infra.gpu03.direction_discovery.test_no_loophole_plan import Fixture as OriginalFixture


class Fixture(OriginalFixture):
    def save(self, path, value):
        if Path(path).name == "candidate.safetensors" and isinstance(value, bytes):
            value += b"\0" * (361240 - len(value))
        return super().save(path, value)

    def __init__(self, root):
        super().__init__(root)
        self.hpatches = []
        for key in ("OLD_FULL_SHA", "DATASET_SHA", "CARRIERS_SHA"):
            self.hpatch(key, getattr(original, key))
        self.work = self.root / "h100-work"
        self.work.mkdir()
        self.hpatch("WORK_ROOT", str(self.work))
        self.mapped = self.save(self.work / "input/layer21.safetensors", self.target.read_bytes())
        self.python = self.save(self.work / "runtime/python", b"authored replacement Python binary")
        self.env = self.save(self.work / "input/original_environment.json", {
            "python": "3.12.3", "credentials_recorded": False,
            "packages": {k: v for k, v in m.VERSIONS.items() if k != "tokenizers"}})
        self.hpatch("HISTORICAL_ENVIRONMENT_SHA", m.sha(self.env))
        self.runtime_proof = {
            "status": "verified_h100_runtime_replacement", "host": m.HOST, "instance_id": m.INSTANCE,
            "python_version": "3.12.3", "python_binary": m.ref(self.python), "runtime_versions": m.VERSIONS,
            "original_environment_sha256": m.sha(self.env), "original_python_binary_reused": False,
            "numerical_libraries_imported": True, "process_release_verified": True,
            "implementation": "CPython", "compiler": "authored GCC", "build": ["authored", "date"],
            "platform": "authored Linux", "libc": ["glibc", "2.35"]}
        self.runtime_proof_path = self.save(self.work / "input/runtime_qualification.json", self.runtime_proof)
        self.portability = {
            "schema_version": 1, "protocol": m.PORTABILITY_PROTOCOL, "work_root": str(self.work),
            "candidate_map": {str(self.target): m.ref(self.mapped)},
            "runtime_replacement": {"historical_environment": m.ref(self.env), "python_binary": m.ref(self.python),
                                    "qualification": m.ref(self.runtime_proof_path)}}
        self.bundle["h100_portability"] = self.portability
        self.hfreeze()

    def hpatch(self, key, value):
        p = patch.object(m, key, value); p.start(); self.hpatches.append(p)

    def close(self):
        for p in reversed(getattr(self, "hpatches", [])): p.stop()
        super().close()

    def hfreeze(self, *, build=True):
        audit = json.loads((self.pkg / "projection_audit.json").read_bytes())
        audit.update(construction=m.CONSTRUCTION, h100_portability_sha256=m.digest(self.portability),
                     physical_conditions={k: v for k, v in m.translate_conditions(self.conditions, self.portability).items() if v["layers"]})
        self.save(self.pkg / "projection_audit.json", audit)
        proof = json.loads((self.pkg / "projection_independent_verification.json").read_bytes())
        proof["audit_sha256"] = m.sha(self.pkg / "projection_audit.json")
        self.save(self.pkg / "projection_independent_verification.json", proof)
        self.refreeze(build=False)
        self.proof["h100_portability_sha256"] = m.digest(self.portability)
        self.save(self.proof_path, self.proof); self.proof_ref = m.ref(self.proof_path)
        if build:
            self.hplan = m.build_plan(self.old_path, self.bundle_ref, self.budget,
                                     prompt_package_verification=self.proof_ref)


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name).resolve()); self.addCleanup(self.f.close)

    def test_full740_logical_science_and_only_candidate_path_translation(self):
        f = self.f
        m.validate_inputs(f.hplan, f.prepared, f.dataset)
        self.assertEqual(f.hplan["conditions"], f.old["conditions"])
        self.assertEqual([r["seed"] for r in f.hplan["requests"]], [r["seed"] for r in f.old["requests"]])
        physical = m.physical_conditions(f.hplan)
        physical["target:L21.transition.pc04"]["layers"][0]["path"] = str(f.target)
        self.assertEqual(physical, f.conditions)
        self.assertFalse(f.hplan["no_loophole_capability"]["cross_architecture_bitwise_equivalence_claimed"])

    def test_original_target_path_can_be_absent_and_is_not_a_binding(self):
        f = self.f
        f.target.rename(f.target.with_suffix(".historical-not-present"))
        paths = m.bindings(f.hplan)
        self.assertIn(f.mapped, paths)
        self.assertNotIn(f.target, paths)
        self.assertEqual(len(m.validate_full(f.hplan)["old_plan"]["requests"]), 740)

    def test_missing_wrong_extra_and_alias_maps_fail(self):
        for mutate in [lambda x: x["candidate_map"].clear(),
                       lambda x: x["candidate_map"].update(extra=m.ref(self.f.mapped)),
                       lambda x: x["candidate_map"][str(self.f.target)].update(sha256="0" * 64),
                       lambda x: x["candidate_map"][str(self.f.target)].update(path=str(self.f.target)),
                       lambda x: x["candidate_map"][str(self.f.target)].update(size_bytes=1)]:
            value = copy.deepcopy(self.f.portability); mutate(value)
            with self.assertRaises(ValueError): m.validate_portability(value, self.f.conditions)

    def test_mapped_payload_mutation_or_symlink_fails(self):
        f = self.f
        f.save(f.mapped, b"changed")
        with self.assertRaisesRegex(ValueError, "hash/size"): m.validate_full(f.hplan)
        f.mapped.unlink(); f.mapped.symlink_to(f.target)
        with self.assertRaisesRegex(ValueError, "non-symlink"): m.validate_full(f.hplan)

    def test_runtime_failure_wrong_version_or_host_rejected(self):
        for key, value in (("process_release_verified", False), ("python_version", "3.12.13"),
                           ("host", "gpu-04"), ("instance_id", "different"),
                           ("original_python_binary_reused", True), ("numerical_libraries_imported", False),
                           ("implementation", "PyPy")):
            with self.subTest(key=key):
                f = self.f; proof = copy.deepcopy(f.runtime_proof); proof[key] = value
                p = f.save(f.work / "input/bad-runtime.json", proof)
                port = copy.deepcopy(f.portability); port["runtime_replacement"]["qualification"] = m.ref(p)
                with self.assertRaises(ValueError): m.validate_portability(port, f.conditions)

    def test_python_content_replacement_rejected(self):
        self.f.save(self.f.python, b"different build")
        with self.assertRaisesRegex(ValueError, "Python bytes"): m.validate_full(self.f.hplan)

    def test_logical_conditions_coordinates_and_hardware_claim_drift_rejected(self):
        for mutate in [lambda p: p["requests"][0].update(seed=1),
                       lambda p: p["requests"].pop(),
                       lambda p: p["conditions"]["target:L21.transition.pc04"]["layers"][0].update(path=str(self.f.mapped)),
                       lambda p: p["no_loophole_capability"].update(cross_architecture_bitwise_equivalence_claimed=True),
                       lambda p: p.update(test_bundle={})]:
            p = copy.deepcopy(self.f.hplan); mutate(p)
            with self.assertRaises(ValueError): m.validate_full(p)

    def test_package_proof_must_bind_portability(self):
        f = self.f; proof = copy.deepcopy(f.proof); proof["h100_portability_sha256"] = "0" * 64
        p = f.save(f.root / "wrong-external.json", proof)
        with self.assertRaisesRegex(ValueError, "portability proof"):
            m.build_plan(f.old_path, f.bundle_ref, f.budget, prompt_package_verification=m.ref(p))

    def test_ledger_stale_collision_and_test_counter_fail(self):
        for mutate in [lambda b: b.update(generation_requests=0),
                       lambda b: b.update(untouched_test_requests=1),
                       lambda b: b["phases"].append({"generation_request_ids": [self.f.hplan["requests"][0]["request_id"]]})]:
            b = copy.deepcopy(self.f.budget); mutate(b)
            with self.assertRaises(ValueError): m.validate_against_ledger(self.f.hplan, b)

    def test_fresh_write_and_frozen_source_gate(self):
        f = self.f
        receipt = m.write_plan(f.old_path, f.bundle_ref, f.budget, f.root / "written", prompt_package_verification=f.proof_ref)
        self.assertFalse(receipt["budget_reserved"])
        self.assertEqual(json.loads(Path(receipt["request_plan"]).read_bytes()), f.hplan)
        with self.assertRaises(ValueError): m.write_plan(f.old_path, f.bundle_ref, f.budget, f.root / "written", prompt_package_verification=f.proof_ref)
        with patch.object(m, "FROZEN_PLANNER_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Frozen scientific planner"): m.validate_full(f.hplan)


if __name__ == "__main__":
    unittest.main()
