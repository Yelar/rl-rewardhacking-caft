import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import candidate_run as r
except ImportError:
    import candidate_run as r


def manifest_fixture(root):
    token = "codex-candidates-unittest-20260907"
    stage = root / token
    stage.mkdir()
    m = {"schema_version": 1, "purpose": "checkpoint60_cpu_candidate_discovery", "mode": "production",
         "host": r.HOST, "uid": r.UID, "python": r.PYTHON, "run_token": token,
         "stage": str(stage), "output": str(stage / "results"), "control": str(stage / "control"),
         "workers": 2, "worker_cpus": [120, 121], "controller_cpu": 119, "verifier_cpu": 118,
         "science": {"layers": list(range(36)), "windows": list(r.c.WINDOWS), "seed": 6001,
             "bootstrap": 200, "pca_rank": 10, "pca_oversample": 14, "pca_power_iters": 2,
             "qualification_layer": r.QUALIFICATION_LAYER, "fit_problems": 111, "fit_records": 333},
         "limits": {"systemd_deadline_seconds": 14400, "internal_deadline_seconds": 14100,
                    "max_aggregate_rss_bytes": 12 * r.GIB, "systemd_memory_max_bytes": 16 * r.GIB,
                    "candidate_array_limit_mib": 1024},
         "command": r.clean_command([r.PYTHON, str(stage / "source" / "candidate_run.py"), "--supervise", str(stage / "reviewed_manifest.json")]),
         "bound_files": {}}
    return m


def save_manifest(m):
    path = Path(m["stage"]) / "reviewed_manifest.json"
    path.write_text(r.c.canonical(m) + "\n")
    return path, r.c.sha256_file(path)


class ManifestTests(unittest.TestCase):
    def test_v2_raw_path_binding_preserves_v1_compatibility_and_all_fit_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = manifest_fixture(root)
            prepared, exclusions = root / "prepared.jsonl", root / "exclusions.json"
            prepared.write_text("original records")
            exclusions.write_text("original exclusions")
            m.update(raw_manifest_sha256="r" * 64, raw_package="/scratch/repaired-cache",
                     prepared_records=str(prepared), exclusions=str(exclusions))
            s = m["science"]
            plan = {"host": r.HOST, "no_training": True, "budget": {"maximum_candidate_cpu_wall_seconds": 14400},
                    "fit": {"layers": s["layers"], "windows": s["windows"], "seed": s["seed"],
                            "problem_bootstrap": s["bootstrap"], "pca_rank": s["pca_rank"],
                            "pca_oversample": s["pca_oversample"], "pca_power_iters": s["pca_power_iters"]},
                    "inputs": {"raw_artifact_manifest_sha256": m["raw_manifest_sha256"],
                               "prepared_records_sha256": r.c.sha256_file(prepared), "exclusions_sha256": r.c.sha256_file(exclusions)}}
            r.validate_plan(plan, m)  # Existing v1 manifests have no explicit raw path.
            plan["inputs"]["raw_package"] = m["raw_package"]
            r.validate_plan(plan, m)
            plan["inputs"]["raw_package"] = "/scratch/original-cache"
            with self.assertRaisesRegex(ValueError, "raw package path"):
                r.validate_plan(plan, m)

    def test_valid_manifest_and_scientific_fail_closed_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = manifest_fixture(root)
            with patch.object(r, "ROOT", root):
                path, digest = save_manifest(m)
                self.assertEqual(r.load_manifest(path, digest, check_files=False), m)
                for mutate, error in (
                    (lambda x: x["science"].update(seed=123), "scientific"),
                    (lambda x: x.update(worker_cpus=[120, 120]), "overlap"),
                    (lambda x: x.update(output=str(root / "foreign")), "escaped"),
                    (lambda x: x["limits"].update(systemd_deadline_seconds=99999), "deadline"),
                    (lambda x: x["limits"].update(systemd_memory_max_bytes=32 * r.GIB), "budget"),
                    (lambda x: x.update(host="gpu-03"), "identity"),
                ):
                    bad = copy.deepcopy(m)
                    mutate(bad)
                    path, digest = save_manifest(bad)
                    with self.assertRaisesRegex(ValueError, error):
                        r.load_manifest(path, digest, check_files=False)

    def test_hash_or_bound_source_change_invalidates_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = manifest_fixture(root)
            source = root / "source.py"
            source.write_text("old")
            m["bound_files"][str(source)] = r.bound_info(source)
            with patch.object(r, "ROOT", root):
                path, digest = save_manifest(m)
                with self.assertRaisesRegex(ValueError, "digest"):
                    r.load_manifest(path, "0" * 64)
                source.write_text("new")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    r.load_manifest(path, digest)

    def test_layer_shards_cover_all_remaining_layers_exactly_once(self):
        for workers in range(1, 9):
            shards = r.layer_shards(workers)
            flattened = [x for shard in shards for x in shard]
            self.assertEqual(sorted(flattened + [r.QUALIFICATION_LAYER]), list(range(36)))
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertEqual(shards, r.layer_shards(workers))
        with self.assertRaisesRegex(ValueError, "worker count"):
            r.layer_shards(9)

    def test_environment_hides_gpu_and_strips_credentials(self):
        with patch.dict(os.environ, {"HF_TOKEN": "secret", "WANDB_API_KEY": "secret", "AWS_SECRET_ACCESS_KEY": "secret"}):
            env = r.cpu_environment()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertFalse({"HF_TOKEN", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY"} & set(env))
        command = r.clean_command(["python", "safe.py"])
        self.assertEqual(command[:2], ["/usr/bin/env", "-i"])
        self.assertEqual(command[-2:], ["python", "safe.py"])

    def test_qualification_timing_blocks_unaffordable_fanout(self):
        report = r.fanout_budget(30, 8, 10000)
        self.assertEqual(report["estimated_remaining_seconds"], 900)
        with self.assertRaisesRegex(ValueError, "insufficient conservative"):
            r.fanout_budget(1000, 8, 9000)

    def test_worker_command_preserves_all_scientific_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            m = manifest_fixture(Path(directory))
            m.update(raw_package="raw", raw_manifest_sha256="digest", prepared_records="prepared", exclusions="excluded", tokenizer="tokenizer")
            receipt = Path(directory) / "receipt.json"
            receipt.write_text("proof")
            command = r.candidate_command(m, "fresh", [0, 4, 8], receipt)
            for flag, value in (("--seed", "6001"), ("--bootstrap", "200"), ("--pca-rank", "10"),
                                ("--pca-oversample", "14"), ("--pca-power-iters", "2"),
                                ("--raw-integrity-sha256", r.c.sha256_file(receipt))):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual(command[command.index("--layers") + 1:command.index("--seed")], ["0", "4", "8"])

    def test_receipt_records_unknown_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            m = manifest_fixture(Path(directory))
            Path(m["control"]).mkdir()
            with patch.object(r, "load_manifest", return_value=m), patch.dict(os.environ, {}, clear=True):
                r.receipt("ignored", "digest")
            report = json.loads((Path(m["control"]) / "supervisor_exit.json").read_text())
            self.assertEqual(report["service_result"], "unknown")
            self.assertEqual(report["exit_status"], "unknown")
            self.assertFalse(report["success_marker_present"])

    def test_launch_consumes_intent_even_when_systemd_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            m = manifest_fixture(Path(directory))
            m.update(authorization="user authorized", runtime_versions={})
            result = subprocess.CompletedProcess([], 1, "", "rejected")
            with patch.object(r, "host_identity"), patch.object(r, "load_manifest", return_value=m), \
                 patch.object(r, "runtime_versions", return_value={}), patch.object(r, "resource_check", return_value={}), \
                 patch.object(r.subprocess, "run", return_value=result) as run:
                with self.assertRaisesRegex(ValueError, "launch failed"):
                    r.launch("manifest", "digest")
                command = run.call_args.args[0]
                self.assertIn("--property=RuntimeMaxSec=14400", command)
                self.assertIn("--property=KillMode=control-group", command)
                self.assertTrue(any(x.startswith("--property=ExecStopPost=") for x in command))
                self.assertTrue((Path(m["control"]) / "launch_intent.json").is_file())
                with self.assertRaises(FileExistsError):
                    r.launch("manifest", "digest")

    def test_cpu_lock_is_exclusive_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(r, "ROOT", root), patch.object(r, "UID", os.getuid()):
                lock = r.acquire_cpu_lock(120)
                try:
                    with self.assertRaises(BlockingIOError):
                        r.acquire_cpu_lock(120)
                finally:
                    lock.close()
                (root / ".codex-candidate-cpu-121.lock").symlink_to(root / ".codex-candidate-cpu-120.lock")
                with self.assertRaises(OSError):
                    r.acquire_cpu_lock(121)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc process ownership integration")
class ProcessTests(unittest.TestCase):
    def test_own_process_identity_and_exact_group_cleanup(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        info = r.process_stat(process.pid)
        self.assertEqual(info["pid"], process.pid)
        self.assertGreater(info["start_ticks"], 0)
        try:
            r.terminate_owned([process])
            self.assertIsNotNone(process.returncode)
            self.assertIsNone(r.process_stat(process.pid))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
