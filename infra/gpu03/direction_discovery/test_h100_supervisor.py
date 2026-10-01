"""Authored source/plan/service fixtures. GPU/systemd/model operations are mocked.

Success paths exercise actual H100 planner and supervisor file/package checks.
They are not a replacement for the required real CPU lifetime and GPU qualification.
"""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from infra.gpu03.direction_discovery import h100_supervisor as s
from infra.gpu03.direction_discovery import h100_budget as budget
from infra.gpu03.direction_discovery import h100_no_loophole_protocol as protocol
from infra.gpu03.direction_discovery.test_h100_no_loophole_protocol import Fixture as PlanFixture
from infra.gpu03.direction_discovery.test_h100_qualification import result_fixture


class Fixture(PlanFixture):
    def __init__(self, root):
        super().__init__(root)
        self.spatches = []
        for name, value in (("WORK", self.root), ("OUTPUTS", self.root / "outputs"), ("UID", os.getuid())):
            self.spatch(name, value)
        s.OUTPUTS.mkdir()
        self.source = self.root / "source"
        project = Path(__file__).resolve().parents[3]
        required = [*protocol.SOURCES, protocol.FROZEN_HERE, protocol.HERE, s.HERE,
                    "infra/gpu03/direction_discovery/h100_qualification.py",
                    *["infra/gpu03/direction_discovery/"+name for name in
                      ("h100_evaluate.py", "helper_aware_evaluation_v1.py", "h100_sandbox.py", "h100_sandbox_qualification.py",
                       "qualify_helper_aware_v1.py")]]
        for relative in required:
            self.save(self.source / relative, (project / relative).read_bytes())
        self.helper = self.make_helper_qualification()
        self.base, self.checkpoint = self.root / "base", self.root / "checkpoint"
        for i in range(13): self.save(self.base / f"file{i}.bin", f"authored base{i}".encode())
        for i in range(2): self.save(self.checkpoint / f"adapter{i}.bin", f"authored adapter{i}".encode())
        self.model_files = {**s.inventory(self.base), **s.inventory(self.checkpoint)}
        self.life = self.save(self.root / "life.json", {"status": "verified_h100_user_service_disconnect_survival",
            "host": s.HOST, "uid": s.UID, "linger": True, "survived_disconnect": True, "processes_released": True})
        self.timers = self.save(self.root / "timers.json", {"status": "verified_h100_existing_deadline_timers",
            "host": s.HOST, "final_sync_at": s.DURABLE_AT, "shutdown_at": s.SHUTDOWN_AT,
            "final_sync_active": True, "shutdown_active": True})
        self.parity = self.save(self.root / "parity.json", {"status": "independently_verified_h100_model_parity",
            "instance_id": s.INSTANCE, "model_files": self.model_files, "all_original_hashes_equal": True,
            "original_inventory_sha256": s.ORIGINAL_MODEL_INVENTORY_SHA})
        self.profile = {"status": "verified_h100_workstation_profile", "host": s.HOST,
            "user": s.OWNER, "uid": s.UID, "instance_id": s.INSTANCE, "instance_type": s.INSTANCE_TYPE,
            "ownership_token": s.OWNER_TOKEN, "runtime_versions": s.VERSIONS, "available_cpus": list(range(192)),
            "python": str(self.python), "model_files": self.model_files, "model_parity_proof": s.ref(self.parity),
            "lifetime_proof": s.ref(self.life), "deadline_proof": s.ref(self.timers),
            "gpu_inventory": [{"index": i, "uuid": f"GPU-00000000-0000-0000-0000-{i:012d}",
                              "name": "NVIDIA H100 80GB HBM3", "memory_total_mib": 81559,
                              "memory_free_mib": 81081} for i in range(8)]}
        self.profile_path = self.save(self.root / "profile.json", self.profile)
        self.auth = self.save(self.root / "authority.json", {"status": "explicit_user_authorization_recorded",
            "scope": "no_loophole_step1_only", "host": "codex-h100", "instance_id": s.INSTANCE})
        self.source_proof = self.save(self.root / "source_proof.json", {"status": "verified_h100_step1_cpu_source",
            "source_root": str(self.source), "source_files": s.inventory(self.source), "tests_passed": 18,
            "tests_failed": 0, "producer_exit_status": 0, "process_release_verified": True})
        self.plan_path = self.save(self.root / "h100_plan.json", self.hplan)
        self.before = {"generation_requests": 2646, "tf_requests": 9228, "untouched_test_requests": 0,
                       "untouched_test_generation_requests": 0, "untouched_test_tf_requests": 0,
                       "gpu_phase_wall_seconds": 20456.688430309296}
        self.seed = self.save(self.root / "seed.json", {"budget": {**self.before, "phases": []},
            "utc": "2026-09-08T12:00:00Z", "python": "3.12.3 authored"})
        for key, value in (("LEDGER_ROOT", self.root / "canonical-ledger"), ("HISTORICAL_SEED_SHA", s.sha(self.seed))):
            p=patch.object(budget,key,value); p.start(); self.spatches.append(p)

    def spatch(self, name, value):
        p = patch.object(s, name, value); p.start(); self.spatches.append(p)

    def close(self):
        for p in reversed(getattr(self, "spatches", [])): p.stop()
        super().close()

    def make_helper_qualification(self):
        from infra.gpu03.direction_discovery import h100_evaluate as evaluation
        from infra.gpu03.direction_discovery import h100_sandbox_qualification as isolation
        from infra.gpu03.direction_discovery import qualify_helper_aware_v1 as helper
        from infra.gpu03.direction_discovery.test_h100_sandbox_qualification import report_fixture
        folder=s.OUTPUTS/'authored-helper-qualification'; package=folder/'sandbox-output/sandbox-qualification'
        report=report_fixture(); probe=b'authored readonly probe\n'
        self.save(package/isolation.OUTPUT_PROBE,probe)
        report.update(source_sha256=s.sha(isolation.__file__),h100_sandbox_sha256=s.sha(evaluation.sandbox.__file__),
            frozen_outer_sha256=evaluation.sandbox.OUTER_SHA,frozen_transport_sha256=evaluation.sandbox.BOUNDED_SHA,
            frozen_repository=isolation.FROZEN_REPOSITORY,
            readonly_hashes={report['output_probe']:s.sha(package/isolation.OUTPUT_PROBE),
                '/input/helper_review.json':isolation.KNOWN_SHA,
                '/work/src/evaluate/helpers.py':isolation.FROZEN_REPOSITORY['src/evaluate/helpers.py']})
        self.save(package/'report.json',report)
        self.save(package/'fixture_sources.json',isolation.fixture_sources(report['output_probe']))
        for name,value in report['fixtures'].items(): self.save(package/(name+'.json'),value)
        artifact={p.name:{'sha256':s.sha(p),'size_bytes':p.stat().st_size} for p in package.iterdir()}
        self.save(package/'artifact_manifest.json',artifact)
        exit_ref=s.ref(self.save(folder/'producer_exit.json',{'returncode':0,'timed_out':False,
            'error_type':None,'child_reaped':True,'remaining_group_pids':[]}))
        isolation_proof=isolation.verify(package,s.sha(package/'artifact_manifest.json'))
        isolation_ref=s.ref(self.save(folder/'isolation_verification.json',
            {**isolation_proof,'producer_exit':exit_ref,'process_release_verified':True}))
        # Replace only historical fixture inputs and the production module-loader
        # import; execute the real 26-case package verifier on authored records.
        # No classification/generated program is executed in this fixture.
        known={'cases':{group:{'candidate_records':[{'confirmed_omission_failure':True,
            'helper_sources':{'assertEqual':helper.ENFORCING},'request_id':f'{group}-{i}'} for i in range(count)]}
            for group,count in (('screening',17),('auxiliary_positive_examples',4))}}
        helper_dir=folder/'sandbox-output/helper-qualification'
        known_path=self.save(helper_dir/'known_helper_review.json',known)
        import hashlib
        for obj,key,value in ((helper,'KNOWN_SHA',s.sha(known_path)),
                (helper,'HELPER_EXPECTATIONS',{hashlib.sha256(helper.ENFORCING.encode()).hexdigest():True}),
                (helper.repair,'modules',lambda:(None,None))):
            patcher=patch.object(obj,key,value); patcher.start(); self.spatches.append(patcher)
        definitions=helper.fixtures(known); self.save(helper_dir/'fixtures.json',definitions)
        rows=[]
        for definition in definitions:
            metrics={'ground_truth_correctness':definition['expected_whole_program_gt']}
            legacy={'evaluation_status':'evaluated','repository_test_modification':definition['expected_legacy_modification'],
                'metrics':metrics,'repository_evaluation':{key:{'test_errors':['AttributeError: assertEqual']} for key in
                    ('response_test_func_gt_result','response_test_func_arbitrary_result')}}
            corrected={'status':definition['expected_corrected_status'],
                'repository_test_modification':definition['expected_corrected_modification'],'metrics':metrics,
                'transport':{'timeout':0,'output_overflow':0,'transport_error':0},
                'corrected_probe_results':{'canonical':{'pass_rate':1.0},'arbitrary':{'pass_rate':0.0}}}
            rows.append({'case_id':definition['case_id'],'legacy':legacy,'corrected':corrected})
        self.save(helper_dir/'records.jsonl',self.lines(rows))
        self.save(helper_dir/'report.json',{'status':'passed','policy':evaluation.repair.POLICY,'cases':26,
            'repair_source_sha256':s.sha(evaluation.repair.__file__),'qualifier_source_sha256':s.sha(helper.__file__),
            'h100_sandbox_sha256':s.sha(evaluation.sandbox.__file__),'source_pins':evaluation.repair.SOURCE_PINS,
            'legacy_evaluate_sha256':evaluation.repair.LEGACY_EVALUATE_SHA,
            'transport':{'calls':149,'timeout':0,'output_overflow':0,'transport_error':0}})
        manifest={p.name:{'sha256':s.sha(p),'size_bytes':p.stat().st_size} for p in helper_dir.iterdir()}
        self.save(helper_dir/'artifact_manifest.json',manifest)
        inner=s.ref(self.save(folder/'qualifier_verification.json',
            helper.verify(helper_dir,s.sha(helper_dir/'artifact_manifest.json'))))
        return self.save(folder/'SUCCESS.json',{'status':'independently_verified_h100_helper_classifier_qualification',
            'instance_id':s.INSTANCE,'helper_source_sha256':evaluation.REPAIR_SHA,'sandbox_source_sha256':evaluation.SANDBOX_SHA,
            'cases':26,'all_expected_outcomes_match':True,'whole_program_gt_preserved':True,'process_release_verified':True,
            'qualifier_verification':inner,'producer_exit':exit_ref,'isolation_qualification':isolation_ref,
            'scientific_generated_programs_executed':False,'gpu_calls':0})

    def spec(self, main=False):
        token = "authored-h100-main" if main else "authored-h100-qualification"
        spec = {"run_token": token, "phase": "h100_no_loophole_capability" if main else "h100_numerical_qualification",
            "stage": str(s.OUTPUTS / token), "source_root": str(self.source),
            "request_plan": s.ref(getattr(self,"main_plan_path",self.plan_path) if main else self.plan_path),
            "host_profile": s.ref(self.profile_path), "python": str(self.python), "model_snapshot": str(self.base),
            "checkpoint": str(self.checkpoint), "model_files": self.model_files, "source_files": s.inventory(self.source),
            "source_qualification": s.ref(self.source_proof), "authorization": s.ref(self.auth),
            "helper_qualification": s.ref(self.helper),
            "gpu_ids": list(range(8)) if main else [3], "postprocessing_seconds": 900}
        if main:
            proof = self.save(self.root / "qual-proof.json", {"status": "independently_verified_h100_generation",
                "phase": "h100_numerical_qualification", "scientific_identity_sha256": s.scientific_identity(self.hplan),
                "host_profile_sha256": s.sha(self.profile_path), "gpu_release_verified": True, "process_release_verified": True,
                "source_root": str(self.source), "source_qualification": s.ref(self.source_proof), "generation_requests": 6, "tf_requests": 0,
                "helper_qualification": s.ref(self.helper),
                "numerical": {"status": "verified_h100_six_call_numerical_qualification"}})
            spec["inference_qualification"] = s.ref(proof)
        return spec

    def build_stage(self, main=False):
        journal=budget.LEDGER_ROOT / "admissions.jsonl"
        if main and journal.exists():
            current=budget.account(s.ref(self.seed),[json.loads(x) for x in journal.read_text().splitlines()])
            plan=protocol.build_plan(self.old_path,self.bundle_ref,current,prompt_package_verification=self.proof_ref)
            self.main_plan_path=self.save(self.root / "main_h100_plan.json",plan)
        r = s.build(self.spec(main)); return r, json.loads(Path(r["path"]).read_bytes())

    def admission(self, r, m):
        _, requests, _ = s.request_context(m)
        before = dict(self.before)
        if m["generation_requests"] == 740:
            before["generation_requests"] += 6; before["gpu_phase_wall_seconds"] += 780
        after = {**before, "generation_requests": before["generation_requests"] + m["generation_requests"],
                 "gpu_phase_wall_seconds": before["gpu_phase_wall_seconds"] + m["limits"]["reserved_wall_seconds"]}
        return {"status": "admitted_h100_no_loophole_phase", "manifest": r,
            "manifest_sha256": r["sha256"], "run_token": m["run_token"], "phase": m["phase"],
            "host_profile_sha256": m["host_profile"]["sha256"], "request_plan_sha256": m["request_plan"]["sha256"],
            "generation_requests": m["generation_requests"], "generation_request_ids": [x["request_id"] for x in requests],
            "tf_requests": 0, "reserved_wall_seconds": m["limits"]["reserved_wall_seconds"],
            "budget_before": before, "budget_after": after, "historical_seed": s.ref(self.seed),
            "authorization": m["authorization"], "admitted_at": datetime.now(timezone.utc).isoformat()}

    def gpu_rows(self):
        return [{**g, "memory_used_mib": 0,
                 "utilization_percent": 0, "processes": []} for g in self.profile["gpu_inventory"]]

    def admit(self, r, m):
        _, requests, _=s.request_context(m)
        budget.admit(root=budget.LEDGER_ROOT,seed=s.ref(self.seed),manifest=r,phase=m["phase"],run_token=m["run_token"],
            host_profile=m["host_profile"],request_plan=m["request_plan"],authorization=m["authorization"],
            generation_request_ids=[x["request_id"] for x in requests])
        return budget.LEDGER_ROOT / (m["run_token"]+".admission.json")

    def completed(self, main=False):
        if main:
            previous_r,previous_m=self.build_stage()
            self.admit(previous_r,previous_m)
        r, m = self.build_stage(main)
        return self.publish_completed(r, m)

    def publish_completed(self, r, m):
        stage = Path(m["stage"])
        control, output, runtime = Path(m["control"]), Path(m["output"]), Path(m["runtime"])
        for p in (control, output, runtime): p.mkdir()
        a = self.admit(r, m)
        self.save(control / "consumed_admission.json", {"run_token": m["run_token"], "manifest_sha256": r["sha256"], "admission": s.ref(a)})
        identity = {"pid": 99999999, "pgid": 99999999, "uid": s.UID, "start_ticks": 100}
        group = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + m["run_token"] + ".service"
        fields = {"ActiveState": "active", "SubState": "running", "MainPID": "99999999", "ControlGroup": group,
            "RuntimeMaxUSec": str(m["limits"]["systemd_runtime_seconds"])+"s", "TimeoutStopUSec": "1min 30s", "MemoryMax": str(160*1024**3),
            "TasksMax": "512", "KillMode": "control-group", "InvocationID": "b"*32}
        self.save(control / "service_started.json", {"fields": fields, "main_identity": identity, "at": 1001})
        self.save(control / "launch_intent.json", {"at": 1000})
        self.save(control / "supervisor_exit.json", {"run_token": m["run_token"], "manifest_sha256": r["sha256"],
            "invocation_id": "b"*32, "service_result": "success", "exit_code_kind": "exited", "exit_status": "0", "at": 1200})
        for w in m["workers"]:
            task = json.loads(Path(w["task"]).read_bytes()); folder = output / w["name"]; folder.mkdir()
            self.save(folder / "task.json", task)
            self.save(folder / "SUCCESS.json", {"status": "succeeded", "run_token": m["run_token"],
                "worker_name": w["name"], "requests": w["requests"], "mode": "generate", "model_load_reports": {
                    "with_adapter": True, "active_adapters": ["default"], "lora_parameter_tensors": 2,
                    "nonzero_lora_parameter_tensors": 2, "lora_parameter_elements": 4}})
            rows = [{**x, "original_class": "clean_correct_evaluator_present", "result": result_fixture(x["condition_id"])} for x in task["requests"]]
            self.save(folder / "results.jsonl", self.lines(rows))
        if m.get("continuation") is not None:
            plan, requests, _=s.request_context(m)
            physical,_=s.collect_physical_rows(m,plan,requests)
            self.save(output/"logical_results.jsonl",self.lines(s.logical_rows(m,plan,physical)))
        self.save(output / "gpu_release.json", {"verified": True, "gpu_ids": m["gpu_ids"]})
        self.save(output / "campaign_summary.json", {"status": "succeeded", "run_token": m["run_token"],
            "manifest_sha256": r["sha256"], "invocation_id": "b"*32, "control_group": group,
            "worker_exit_codes": [0]*len(m["workers"]), "worker_receipts": [s.validate_worker(m,w) for w in m["workers"]],
            "gpu_release_verified": True})
        self.save(output / "artifact_manifest.json", {"algorithm": "sha256", "files": s.output_inventory(output)})
        return r, m, fields

    def continuation_spec(self):
        r,q,fields=self.completed()
        output=Path(q['output']); control=Path(q['control'])
        journal=output/'worker_00/results.jsonl'
        rows=[json.loads(x) for x in journal.read_bytes().splitlines()][:5]
        self.save(journal,self.lines(rows))
        for path in (output/'worker_00/SUCCESS.json',output/'campaign_summary.json',output/'artifact_manifest.json'):
            path.unlink()
        terminal=json.loads((control/'supervisor_exit.json').read_bytes())
        terminal.update(service_result='timeout',exit_code_kind='killed',exit_status='15')
        self.save(control/'supervisor_exit.json',terminal)
        self.save(output/'FAILURE.json',{'run_token':q['run_token'],'manifest_sha256':r['sha256'],
            'invocation_id':fields['InvocationID'],'cleanup':{'owned_workers_released':True,'gpu_release_verified':True}})
        evidence={'status':'independently_verified_h100_failed_qualification_reuse','manifest':r,
            'results':s.ref(journal),'service_started':s.ref(control/'service_started.json'),
            'supervisor_exit':s.ref(control/'supervisor_exit.json'),'failure':s.ref(output/'FAILURE.json'),
            'retained_generation_requests':5,'process_release_verified':True,'gpu_release_verified':True}
        cref=s.ref(self.save(self.root/'continuation.json',evidence))
        current=budget.account(s.ref(self.seed),budget.journal_entries(budget.LEDGER_ROOT/'admissions.jsonl'))
        self.main_plan_path=self.save(self.root/'main_h100_plan.json',protocol.build_plan(
            self.old_path,self.bundle_ref,current,prompt_package_verification=self.proof_ref))
        spec=self.spec(True);spec.pop('inference_qualification')
        auth=json.loads(self.auth.read_bytes());auth.update(integrated_validation=True,
            generation_runtime_seconds=21600,cumulative_gpu_wall_cap_seconds=43200)
        spec.update(continuation=cref,generation_runtime_seconds=21600,
                    authorization=s.ref(self.save(self.root/'integrated-authority.json',auth)))
        self.spatch('REUSABLE_QUALIFIER_SOURCES',{name:s.sha(self.source/name) for name in s.REUSABLE_QUALIFIER_SOURCES})
        return spec,evidence,q


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name).resolve()); self.addCleanup(self.f.close)

    def test_actual_planner_build_qualification_six_and_main740(self):
        r, m = self.f.build_stage(); s.load_manifest(r["path"], r["sha256"])
        self.assertEqual(m["workers"][0]["requests"], 6)
        self.assertFalse(Path(m["control"]).exists())
        r, m = self.f.build_stage(True); s.load_manifest(r["path"], r["sha256"])
        self.assertEqual([w["requests"] for w in m["workers"]], [93]*4+[92]*4)
        self.assertEqual(m["scientific_conditions"], self.f.hplan["conditions"])
        task = json.loads(Path(m["workers"][0]["task"]).read_bytes())
        self.assertEqual(task["conditions"]["target:L21.transition.pc04"]["layers"][0]["path"], str(self.f.mapped))

    def continuation_fixture(self):
        for mocked in (patch.object(s,'same_process',return_value=False),patch.object(s,'cgroup_processes',return_value={})): 
            mocked.start();self.addCleanup(mocked.stop)
        return self.f.continuation_spec()

    def test_continuation_real735_build_merge_verify_and_accounting(self):
        spec,evidence,q=self.continuation_fixture();raw=Path(evidence['results']['path']).read_bytes()
        r=s.build(spec);m=s.load_manifest(r['path'],r['sha256'])
        self.assertEqual((m['generation_requests'],m['scientific_generation_requests']),(735,740))
        self.assertEqual([w['requests'] for w in m['workers']],[92]*7+[91])
        self.assertEqual((m['limits']['systemd_runtime_seconds'],m['limits']['reserved_wall_seconds']),(21600,21780))
        self.assertFalse(Path(m['control']).exists())
        r,m,fields=self.f.publish_completed(r,m)
        with patch.object(s,'host_check'),patch.object(s,'unit_state',return_value={**fields,'ActiveState':'inactive','MainPID':'0'}),\
             patch.object(s,'gpu_snapshot',side_effect=self.f.gpu_rows):
            proof=s.verify(r['path'],r['sha256'])
        self.assertEqual((proof['generation_requests'],proof['new_generation_requests'],proof['reused_generation_requests']),(740,735,5))
        self.assertEqual(len(proof['physical_result_files']),8)
        self.assertEqual(proof['result_files'][0]['path'],str(Path(m['output'])/'logical_results.jsonl'))
        self.assertFalse(proof['baseline_repeat_verification_available'])
        self.assertEqual(Path(evidence['results']['path']).read_bytes(),raw)
        before=budget.account(s.ref(self.f.seed),budget.journal_entries(budget.LEDGER_ROOT/'admissions.jsonl'))
        self.assertEqual(before['generation_requests'],3387)
        self.assertEqual(before['gpu_phase_wall_seconds'],self.f.before['gpu_phase_wall_seconds']+780+21780)

    def test_continuation_failed_release_or_changed_rows_reject_before_stage(self):
        spec,evidence,q=self.continuation_fixture()
        bad={**evidence,'process_release_verified':False}
        spec['continuation']=s.ref(self.f.save(self.f.root/'bad-reuse.json',bad))
        with self.assertRaisesRegex(ValueError,'release evidence'):s.build(spec)
        spec['continuation']=s.ref(self.f.root/'continuation.json')
        path=Path(evidence['results']['path']);rows=[json.loads(x) for x in path.read_bytes().splitlines()]
        rows[0]['seed']+=1;self.f.save(path,self.f.lines(rows))
        with self.assertRaisesRegex(ValueError,'Bound file changed'):s.build(spec)
        self.assertFalse(Path(spec['stage']).exists())

    def test_continuation_runtime_checks_reject_bad_completed_projection(self):
        spec,evidence,q=self.continuation_fixture();r=s.build(spec);m=s.load_manifest(r['path'],r['sha256'])
        w=m['workers'][0];task=json.loads(Path(w['task']).read_bytes());request=task['requests'][0]
        row={**request,'result':result_fixture(request['condition_id'])}
        folder=Path(m['output'])/w['name'];folder.mkdir(parents=True)
        path=folder/'results.jsonl';data=s.canonical(row).encode()
        self.f.save(path,data);state={};s.check_completed_requests(m,state)
        self.assertEqual(state[w['name']]['count'],0)
        self.f.save(path,data+b'\n');s.check_completed_requests(m,state)
        self.assertEqual(state[w['name']]['count'],1)
        request=task['requests'][1];row={**request,'result':result_fixture(request['condition_id'])}
        if request['condition_id']=='baseline':row['result']['energy']={'21':{}}
        else:row['result']['energy']['21']['rank']=2
        path.chmod(0o600)
        with path.open('ab') as stream:stream.write(s.canonical(row).encode()+b'\n')
        with self.assertRaises(ValueError):s.check_completed_requests(m,state)

    def test_existing_stage_preserved_and_invalid_profile_before_publication(self):
        r, _ = self.f.build_stage()
        with self.assertRaisesRegex(ValueError, "Fresh stage"): s.build(self.f.spec())
        spec = self.f.spec(True); profile = copy.deepcopy(self.f.profile); profile["host"] = "gpu-04"
        p = self.f.save(self.f.root / "bad-profile.json", profile); spec["host_profile"] = s.ref(p)
        with self.assertRaises(ValueError): s.build(spec)
        self.assertFalse(Path(spec["stage"]).exists())

    def test_no_main_without_positive_released_qualification(self):
        spec = self.f.spec(True); spec.pop("inference_qualification")
        with self.assertRaises(ValueError): s.build(spec)
        self.assertFalse(Path(spec["stage"]).exists())

    def test_tasks_source_models_and_cpu_limits_cannot_drift(self):
        r, m = self.f.build_stage()
        for mutate in (lambda x: x["workers"][0].update(cpu_set="0"),
                       lambda x: x["limits"].update(runtime_seconds=999),
                       lambda x: x["scientific_conditions"].update(baseline={}),
                       lambda x: x["bound_files"].pop(str(self.f.base / "file0.bin"))):
            value = copy.deepcopy(m); mutate(value)
            with self.assertRaises(ValueError): s.validate_manifest_object(value)
        task = json.loads(Path(m["workers"][0]["task"]).read_bytes()); task["requests"][0]["seed"] += 1
        self.f.save(Path(m["workers"][0]["task"]), task)
        with self.assertRaises(ValueError): s.load_manifest(r["path"], r["sha256"])

    def test_admission_exact_counts_requests_seed_and_caps(self):
        r, m = self.f.build_stage(); path = self.f.admit(r,m); a=json.loads(path.read_bytes())
        s.validate_admission(m,r["sha256"],s.ref(path))
        for mutate in (lambda x: x.update(manifest_sha256="0"*64),
                       lambda x: x["generation_request_ids"].reverse(),
                       lambda x: x["budget_before"].update(generation_requests=0),
                       lambda x: x["budget_after"].update(gpu_phase_wall_seconds=0),
                       lambda x: x["budget_after"].update(untouched_test_requests=1)):
            bad = copy.deepcopy(a); mutate(bad); self.f.save(path,bad)
            with self.assertRaises(ValueError): s.validate_admission(m,r["sha256"],s.ref(path))

    def test_bus_identity_transport_and_limits_are_explicit(self):
        r,m=self.f.build_stage(); cmd=s.service_command(m,r["path"],r["sha256"])
        self.assertEqual(cmd.count("INVOCATION_ID=${INVOCATION_ID}"),1)
        self.assertIn("--property=RuntimeMaxSec=600",cmd)
        self.assertIn("--property=TimeoutStopSec=90",cmd)
        self.assertIn("--property=CPUAffinity=94 96-111",cmd)
        stop=next(x for x in cmd if x.startswith("--property=ExecStopPost="))
        self.assertIn("SERVICE_RESULT=${SERVICE_RESULT}",stop)
        self.assertNotIn("DBUS_SESSION_BUS_ADDRESS",s.worker_environment())

    def test_absolute_deadline_reserves_shutdown_and_exit_receipt(self):
        _,m=self.f.build_stage()
        # This timestamp fits the active service but cannot also fit both stop
        # windows. Reject before any timer/linger query or GPU/model operation.
        now=s.DEADLINE-m["postprocessing_seconds"]-m["limits"]["systemd_runtime_seconds"]-1
        with patch.object(s.socket,"gethostname",return_value=s.HOST),\
             patch.object(s.pwd,"getpwuid",return_value=SimpleNamespace(pw_name=s.OWNER)),\
             patch.object(s.sys,"executable",m["python"]),patch.object(s.sys,"version_info",(3,12,3)),\
             patch.object(s.importlib.metadata,"version",side_effect=lambda key:s.VERSIONS[key]),\
             patch.object(s.os,"sched_getaffinity",create=True,return_value={94,*range(96,112)}),\
             patch.object(s.time,"time",return_value=now),patch.object(s.subprocess,"run") as run:
            with self.assertRaisesRegex(ValueError,"durable-copy deadline"): s.host_check(m)
            run.assert_not_called()
            with patch.object(s.time,"time",return_value=now-180):
                run.side_effect=[SimpleNamespace(returncode=0,stdout="active"),
                                 SimpleNamespace(returncode=0,stdout="active"),
                                 SimpleNamespace(returncode=0,stdout="yes")]
                s.host_check(m)

    def test_selected_foreign_fails_excluded_foreign_is_untouched(self):
        _,m=self.f.build_stage(); rows=self.f.gpu_rows()
        rows[0]["processes"]=[{"pid":123,"owner":"another"}]; rows[0]["memory_used_mib"]=20000
        s.check_devices(m,rows)
        rows[3]["processes"]=[{"pid":123,"owner":"another"}]
        with self.assertRaisesRegex(ValueError,"foreign"): s.check_devices(m,rows,settling=[3])
        rows[3]["processes"]=[{"pid":123,"owner":s.OWNER}]
        s.check_devices(m,rows,allowed={3:{123}})

    def test_idle_driver_reserved_memory_uses_frozen_free_baseline(self):
        _,m=self.f.build_stage(); rows=self.f.gpu_rows()
        self.assertEqual(rows[3]["memory_total_mib"]-rows[3]["memory_free_mib"],478)
        s.check_devices(m,rows)
        for changes,reason in (({"memory_used_mib":65},"positively idle"),
                               ({"memory_free_mib":81016},"positively idle"),
                               ({"utilization_percent":2},"positively idle"),
                               ({"processes":[{"pid":123,"owner":"another"}]},"foreign")):
            bad=copy.deepcopy(rows); bad[3].update(changes)
            with self.assertRaisesRegex(ValueError,reason): s.check_devices(m,bad)

    def test_actual_service_resource_and_identity_checks(self):
        _,m,fields=self.f.completed()
        with patch.object(s,"process_info",return_value={"pid":99999999,"uid":s.UID}):
            s.check_service(m,fields)
            for key,value in (("MemoryMax","0"),("TasksMax","999"),("InvocationID","unknown"),
                              ("RuntimeMaxUSec","13min"),("TimeoutStopUSec","91s"),("TimeoutStopUSec","")):
                bad={**fields,key:value}
                with self.assertRaises(ValueError): s.check_service(m,bad)
        self.assertEqual(s.duration_seconds("2h 3min"),7380)
        with self.assertRaises(ValueError): s.duration_seconds("infinity")

    def verify_fixture(self,main=False):
        r,m,fields=self.f.completed(main)
        mocks=[patch.object(s,"host_check"),patch.object(s,"unit_state",return_value={**fields,"ActiveState":"inactive","MainPID":"0"}),
               patch.object(s,"same_process",return_value=False),patch.object(s,"cgroup_processes",return_value={}),
               patch.object(s,"gpu_snapshot",side_effect=self.f.gpu_rows)]
        for p in mocks: p.start(); self.addCleanup(p.stop)
        return r,m

    def test_independent_terminal_qualification_success(self):
        r,m=self.verify_fixture(); proof=s.verify(r["path"],r["sha256"])
        self.assertEqual(proof["generation_requests"],6)
        self.assertTrue(proof["numerical"]["baseline_repeat_bitwise_equal"])
        self.assertTrue(proof["process_release_verified"])
        self.assertEqual(proof["actual_wall_seconds"],200)

    def test_terminal_wall_must_fit_full_charged_reservation(self):
        r,m=self.verify_fixture(); terminal=Path(m["control"])/"supervisor_exit.json"
        value=json.loads(terminal.read_bytes())
        value["at"]=1000+m["limits"]["reserved_wall_seconds"]
        self.f.save(terminal,value)
        self.assertEqual(s.verify(r["path"],r["sha256"])["actual_wall_seconds"],780)
        value["at"]+=1; self.f.save(terminal,value)
        with self.assertRaisesRegex(ValueError,"launch/exit wall receipt"):
            s.verify(r["path"],r["sha256"])

    def test_independent_full740_success_and_changed_output_reject(self):
        r,m=self.verify_fixture(True); proof=s.verify(r["path"],r["sha256"])
        self.assertEqual(len(proof["request_ids"]),740); self.assertIsNone(proof["numerical"])
        path=Path(m["output"])/m["workers"][0]["name"]/"results.jsonl"
        self.f.save(path,path.read_bytes()+b"{}\n")
        with self.assertRaisesRegex(ValueError,"Terminal result bytes"): s.verify(r["path"],r["sha256"])

    def test_missing_terminal_or_live_descendant_fails_before_result_read(self):
        r,m=self.verify_fixture(); terminal=Path(m["control"])/"supervisor_exit.json"
        value=json.loads(terminal.read_bytes()); value["exit_status"]="1"; self.f.save(terminal,value)
        with patch.object(s.numerical,"check_generation",side_effect=AssertionError("early outcome read")):
            with self.assertRaisesRegex(ValueError,"successfully exited"): s.verify(r["path"],r["sha256"])
        value["exit_status"]="0"; self.f.save(terminal,value)
        with patch.object(s,"cgroup_processes",return_value={71:{"uid":s.UID}}):
            with self.assertRaisesRegex(ValueError,"not released"): s.verify(r["path"],r["sha256"])

    def test_identity_ignores_only_mutable_rss(self):
        a={"pid":1,"pgid":1,"uid":s.UID,"start_ticks":10,"rss_bytes":100}
        self.assertTrue(s.same_identity(a,{**a,"rss_bytes":1000}))
        self.assertFalse(s.same_identity(a,{**a,"start_ticks":11}))

    def test_generic_gpu_parser_import_and_no_device_access(self):
        text="\n".join(f'{g["index"]}, {g["uuid"]}, {g["name"]}, {g["memory_total_mib"]}, 0, {g["memory_free_mib"]}, 0' for g in self.f.profile["gpu_inventory"])
        with patch("subprocess.check_output",return_value=text), patch("subprocess.run",return_value=SimpleNamespace(stdout="")):
            self.assertEqual(s.gpu_snapshot(),self.f.gpu_rows())

    def test_launch_consumes_once_before_exact_service_success(self):
        r,m=self.f.build_stage(); a=self.f.admit(r,m)
        identity={"pid":123,"pgid":123,"uid":s.UID,"start_ticks":1}
        fields={"ActiveState":"active","SubState":"running","MainPID":"123","InvocationID":"c"*32}
        env={"PATH":"/usr/bin:/bin","XDG_RUNTIME_DIR":"/run/user/1000","DBUS_SESSION_BUS_ADDRESS":"unix:path=/run/user/1000/bus"}
        def allocate(command, **kwargs):
            self.assertTrue((Path(m["control"])/"consumed_admission.json").is_file())
            self.assertEqual(kwargs["env"],env)
            self.assertEqual(command,s.service_command(m,r["path"],r["sha256"]))
            return SimpleNamespace(returncode=0,stdout="",stderr="")
        with patch.object(s,"host_check"),patch.object(s,"gpu_snapshot",side_effect=self.f.gpu_rows),\
             patch.object(s,"launcher_environment",return_value=env),patch.object(s,"unit_state",return_value=fields),\
             patch.object(s,"check_service",return_value=identity),patch.object(s.os,"sched_getaffinity",create=True,return_value={94}),\
             patch.object(s.subprocess,"run",side_effect=allocate) as run:
            answer=s.launch(r["path"],r["sha256"],s.ref(a))
            self.assertEqual(answer["status"],"launched"); self.assertEqual(run.call_count,1)
            with self.assertRaises(FileExistsError): s.launch(r["path"],r["sha256"],s.ref(a))
            self.assertEqual(run.call_count,1)

    def test_stale_admission_never_allocates(self):
        r,m=self.f.build_stage(); p=self.f.admit(r,m)
        with patch.object(s,"host_check"), patch.object(s.subprocess,"run") as run, patch.object(s,"admission_time",return_value=0):
            with self.assertRaisesRegex(ValueError,"Stale"): s.launch(r["path"],r["sha256"],s.ref(p))
            run.assert_not_called()
        self.assertFalse(Path(m["control"]).exists())

    def test_source_qualification_and_timer_negative_prepublication(self):
        for field in ("source_qualification", "host_profile"):
            spec=self.f.spec()
            if field=="source_qualification":
                value=json.loads(self.f.source_proof.read_bytes()); value["producer_exit_status"]=1
            else:
                value=copy.deepcopy(self.f.profile); value["uid"]=998
            path=self.f.save(self.f.root/(field+"-bad.json"),value); spec[field]=s.ref(path)
            with self.assertRaises(ValueError): s.build(spec)
            self.assertFalse(Path(spec["stage"]).exists())

    def test_helper_qualification_required_before_stage_publication(self):
        spec=self.f.spec(); del spec['helper_qualification']
        with self.assertRaises(KeyError): s.build(spec)
        self.assertFalse(Path(spec['stage']).exists())
        original=json.loads(self.f.helper.read_bytes())
        for key,value in (('status','failed'),('sandbox_source_sha256','0'*64),('whole_program_gt_preserved',False)):
            self.f.save(self.f.helper,{**original,key:value}); spec=self.f.spec()
            with self.assertRaises(ValueError): s.build(spec)
            self.assertFalse(Path(spec['stage']).exists())

    def test_isolation_receipts_and_payload_must_pass_before_publication(self):
        original=json.loads(self.f.helper.read_bytes()); isolation=Path(original['isolation_qualification']['path'])
        value=json.loads(isolation.read_bytes()); exit_path=Path(value['producer_exit']['path'])
        self.f.save(exit_path,{'returncode':0,'timed_out':False,'child_reaped':True,
                             'remaining_group_pids':[101]})
        # Rebind the changed external receipt, so this checks its semantics.
        self.f.save(isolation,{**value,'producer_exit':s.ref(exit_path)})
        self.f.save(self.f.helper,{**original,'producer_exit':s.ref(exit_path),
                                  'isolation_qualification':s.ref(isolation)})
        spec=self.f.spec()
        with self.assertRaisesRegex(ValueError,'exit/release'): s.build(spec)
        self.assertFalse(Path(spec['stage']).exists())

    def test_changed_bound_isolation_artifact_blocks_admission(self):
        r,m=self.f.build_stage()
        payload=self.f.helper.parent/'sandbox-output/sandbox-qualification/report.json'
        self.assertIn(str(payload),m['bound_files'])
        value=json.loads(payload.read_bytes()); value['fixtures']['benign']['success']=False
        self.f.save(payload,value)
        with self.assertRaisesRegex(ValueError,'hash differs'): self.f.admit(r,m)
        self.assertEqual((budget.LEDGER_ROOT/'admissions.jsonl').read_bytes(),b'')
        self.assertFalse((budget.LEDGER_ROOT/(m['run_token']+'.admission.json')).exists())

    def test_changed_helper_payload_blocks_generation_before_publication(self):
        payload=self.f.helper.parent/'sandbox-output/helper-qualification/records.jsonl'
        data=payload.read_bytes(); self.f.save(payload,data+b'{}\n')
        spec=self.f.spec()
        with self.assertRaisesRegex(ValueError,'Qualification payload mismatch'): s.build(spec)
        self.assertFalse(Path(spec['stage']).exists())

    def test_disallowed_source_paths_rejected_before_any_hash(self):
        forbidden=('infra/skypilot/driver.py','infra/gpu03/provenance.tar.gz','src/private_runtime/module.py',
                   'src/wandb_api_key','src/credentials.json','src/.env','src/api_key.py',
                   'unreviewed/module.py')
        for index,name in enumerate(forbidden):
            folder=self.f.root/('authored-source-guard-'+str(index))
            self.f.save(folder/'src/allowed.py',b'# authored safe file\n')
            self.f.save(folder/name,b'authored sentinel, never a credential\n')
            with self.subTest(name=name),patch.object(s,'sha',side_effect=AssertionError('hashed before path rejection')):
                with self.assertRaises(ValueError): s.source_inventory(folder)

    def test_source_path_gate_precedes_helper_plan_and_stage(self):
        spec=self.f.spec()
        self.f.save(self.f.source/'infra/skypilot/forbidden.py',b'# authored excluded source\n')
        with patch.object(s,'helper_qualification_bindings',side_effect=AssertionError('qualification too early')):
            with self.assertRaisesRegex(ValueError,'SkyPilot subtree'): s.build(spec)
        self.assertFalse(Path(spec['stage']).exists())

    def test_owned_real_child_cleanup_uses_only_its_new_group(self):
        child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        identity={"pid":child.pid,"pgid":child.pid,"uid":s.UID,"start_ticks":1}
        with patch.object(s,"same_process",return_value=True):
            s.terminate_owned([(child,identity)])
        self.assertIsNotNone(child.poll())
        self.assertNotEqual(os.getpgrp(),child.pid)

    def test_unloaded_unit_requires_explicit_not_found_identity(self):
        token="authored-unloaded"
        text=f"Id={token}.service\nLoadState=not-found\nActiveState=inactive\nMainPID=0\n"
        with patch.object(s,"launcher_environment",return_value={}),patch.object(s.subprocess,"run",return_value=SimpleNamespace(returncode=1,stdout=text)):
            self.assertEqual(s.unit_state(token)["LoadState"],"not-found")
            with self.assertRaises(ValueError): s.unit_state("different-token")
        with patch.object(s,"launcher_environment",return_value={}),patch.object(s.subprocess,"run",return_value=SimpleNamespace(returncode=1,stdout="")):
            with self.assertRaises(ValueError): s.unit_state(token)


if __name__ == "__main__":
    unittest.main()
