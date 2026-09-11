"""CPU regression checks for native storage, manifest scope, and GPU isolation."""
import copy
import json
import os
from pathlib import Path
import pwd
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file, load_file
import extract_triplet_raw as raw
import launch_triplet_raw as launcher


class RawTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"native.safetensors"
        self.row = {"record_index":0,"record_id":"fixture","input_ids":[3,4,5,6],
                    "input_ids_sha256":raw.ids_hash([3,4,5,6]),"completion_sha256":"example",
                    "checkpoint_sha256":"adapter","completion_token_count":2,"selected_token_positions":[2,3],
                    "region_mask_completion_positions":{"solution__transition":[0,1]}}
        self.data = torch.arange(1,13,dtype=torch.float32).reshape(2,2,3).to(torch.bfloat16)
        self.patches = [patch.object(raw,"LAYER_COUNT",2),patch.object(raw,"HIDDEN",3)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()

    def save(self,kind="h0",data=None,aux=None,metadata=None):
        raw.atomic_tensor(self.path,{kind:self.data if data is None else data,**(raw.auxiliary_tensors(self.row) if aux is None else aux)},
                          raw.expected_metadata(self.row,kind,"digest") if metadata is None else metadata)

    def test_native_roundtrip_success(self):
        self.save()
        result = raw.verify_raw(self.row,self.path,"h0","digest",self.data)
        self.assertEqual(result["shape"],[2,2,3]);self.assertTrue(result["all_finite"])
        self.assertEqual(load_file(str(self.path))["h0"].dtype,torch.bfloat16)

    def test_adapter_raw_roundtrip_success(self):
        self.save("h60")
        self.assertEqual(raw.verify_raw(self.row,self.path,"h60","digest",self.data)["dtype"],"bfloat16")

    def test_cross_model_kind_rejected(self):
        self.save()
        with self.assertRaises(ValueError):raw.verify_raw(self.row,self.path,"h60","digest")

    def test_differences_kind_rejected(self):
        with self.assertRaises(ValueError):raw.expected_metadata(self.row,"delta_h","digest")

    def test_overwrite_rejected(self):
        self.save()
        with self.assertRaises(ValueError):self.save()

    def test_partial_write_preserved(self):
        self.path.with_suffix(".writing").write_text("interrupted")
        with self.assertRaises(ValueError):self.save()
        self.assertEqual(self.path.with_suffix(".writing").read_text(),"interrupted")

    def test_nonfinite_rejected(self):
        data=self.data.clone();data[0,0,0]=float("nan");self.save(data=data)
        with self.assertRaisesRegex(ValueError,"nonfinite"):raw.verify_raw(self.row,self.path,"h0","digest")

    def test_readback_mutation_rejected(self):
        self.save();original=self.data.clone();original[0,0,0]=33
        with self.assertRaisesRegex(ValueError,"readback"):raw.verify_raw(self.row,self.path,"h0","digest",original)

    def test_fp32_raw_rejected(self):
        self.save(data=self.data.float())
        with self.assertRaisesRegex(ValueError,"dtype"):raw.verify_raw(self.row,self.path,"h0","digest")

    def test_changed_input_ids_rejected(self):
        aux=raw.auxiliary_tensors(self.row);aux["input_ids"][0]=999;self.save(aux=aux)
        with self.assertRaisesRegex(ValueError,"IDs/mask"):raw.verify_raw(self.row,self.path,"h0","digest")

    def test_zero_layer_rejected(self):
        data=self.data.clone();data[1]=0;self.save(data=data)
        with self.assertRaisesRegex(ValueError,"all-zero"):raw.verify_raw(self.row,self.path,"h0","digest")

    def test_metadata_rejected(self):
        metadata=raw.expected_metadata(self.row,"h0","changed");self.save(metadata=metadata)
        with self.assertRaisesRegex(ValueError,"metadata"):raw.verify_raw(self.row,self.path,"h0","digest")

    def test_only_two_native_tensor_kinds(self):
        contract=raw.scientific_contract()
        self.assertEqual(contract["models"],["h0","h60"])
        self.assertIs(contract["compute_differences"],False)
        self.assertIs(contract["token_pooling"],False)

    def test_gpu_subset_and_foreign_process(self):
        m={"gpu_ids":[3],"gpu_uuids":{"3":"gpu-three"}}
        row={"index":3,"uuid":"gpu-three","name":raw.engine.EXPECTED_GPU_NAME,"memory_used_mib":3,
             "memory_free_mib":32236,"utilization_percent":0,"processes":[]}
        raw.check_devices(m,[row,{"index":0,"processes":[{"pid":999}]}])
        row["processes"]=[{"pid":123,"owner":"another-user"}]
        with self.assertRaisesRegex(ValueError,"foreign"):raw.check_devices(m,[row],{3:{123}})

    def test_settling_never_allows_foreign_process(self):
        m={"gpu_ids":[3],"gpu_uuids":{"3":"gpu-three"}}
        row={"index":3,"uuid":"gpu-three","name":raw.engine.EXPECTED_GPU_NAME,"memory_used_mib":300,
             "memory_free_mib":31900,"utilization_percent":3,"processes":[]}
        raw.check_devices(m,[row],settling={3})
        with self.assertRaisesRegex(ValueError,"idle"):raw.check_devices(m,[row])
        row["processes"]=[{"pid":123,"owner":pwd.getpwuid(os.getuid()).pw_name}]
        with self.assertRaisesRegex(ValueError,"foreign"):raw.check_devices(m,[row],settling={3})

    def test_worker_environment_has_no_credentials(self):
        with patch.dict(os.environ,{"HF_TOKEN":"secret","AWS_ACCESS_KEY_ID":"secret","WANDB_API_KEY":"secret"}):
            env=raw.worker_env(3)
        self.assertFalse({"HF_TOKEN","AWS_ACCESS_KEY_ID","WANDB_API_KEY"} & set(env))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"3")
        self.assertEqual(raw.worker_env(None)["CUDA_VISIBLE_DEVICES"],"")

    def test_manifest_hash_and_purpose_binding(self):
        m={"schema_version":1,"purpose":"checkpoint60_triplet_raw_activations","host":"gpu-04",
           "scientific":raw.scientific_contract(),"run_token":"raw-test","gpu_ids":[3],
           "stage":"/scratch/researcher/codex_runs/raw-test-stage",
           "output":"/scratch/researcher/codex_runs/raw-test-results",
           "runtime":"/scratch/researcher/codex_runs/raw-test-runtime",
           "source_root":"/scratch/researcher/codex_runs/raw-test-stage/source",
           "limits":{"runtime_seconds":14400,"min_start_free_disk_gib":160},"bound_files":{}}
        self.path.write_text(json.dumps(m));digest=raw.base.sha256_file(self.path)
        raw.load_manifest(self.path,digest)
        with self.assertRaisesRegex(ValueError,"hash"):raw.load_manifest(self.path,"wrong")
        m["scientific"]["compute_differences"]=True;self.path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"contract"):raw.load_manifest(self.path,raw.base.sha256_file(self.path))

    def test_independent_auditor_success_and_saved_raw_retention(self):
        root=Path(self.tmp.name);destination=root/"shard";destination.mkdir()
        prepared=root/"prepared";prepared.mkdir()
        raw.base.atomic_write_jsonl(prepared/"prepared_records.jsonl",[self.row])
        entries=[]
        for kind in ("h0","h60"):
            (destination/kind).mkdir();self.path=destination/kind/raw.record_name(self.row)
            self.save(kind)
            entries.append({"kind":kind,"record_index":0,"native_readback_bitwise_equal":True,
                            **raw.verify_raw(self.row,self.path,kind,"digest",self.data)})
        raw.base.atomic_write_jsonl(destination/"journal.jsonl",entries)
        raw.exclusive_json(destination/"worker_success.json",{"status":"succeeded","record_indices":[0],"model_load_reports":{"h0":{},"h60":{}}})
        task=root/"task.json"
        raw.exclusive_json(task,{"manifest":str(root/"manifest.json"),"manifest_sha256":"digest","destination":str(destination),"record_indices":[0]})
        with patch.object(raw,"load_manifest",return_value={"prepared":str(prepared)}),patch.object(raw,"configure_torch"),\
             patch.object(raw.engine,"_set_parent_death_signal"),patch.object(raw.engine,"validate_model_load_reports"):
            raw.audit(task)
        result=json.loads((destination/"audit_success.json").read_text())
        self.assertFalse(result["differences_computed"])
        self.assertTrue(result["raw_activations_retained"])
        self.assertEqual(len(list(destination.rglob("*.safetensors"))),2)

    def launch_fixture(self):
        root=Path(self.tmp.name)
        m={"stage":str(root),"run_token":"raw-test","authorization":"raw models only",
           "command":["python","raw.py","--supervise",str(root/"manifest.json")],"output":str(root/"out")}
        return m,["launch","--manifest",str(root/"manifest.json"),"--manifest-sha256","digest"]

    def test_launch_success_records_receipt_and_bounded_command(self):
        m,argv=self.launch_fixture()
        from types import SimpleNamespace
        with patch.object(launcher,"load_manifest",return_value=m),patch.object(launcher.sys,"argv",argv),\
             patch.object(launcher.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout="",stderr="")) as run:
            launcher.main()
        command=run.call_args.args[0]
        self.assertIn("--property=RuntimeMaxSec=15000",command)
        self.assertIn("--property=KillMode=control-group",command)
        self.assertTrue(any(x.startswith("--property=ExecStopPost=") for x in command))
        self.assertTrue((Path(m["stage"])/"control/launch_intent.json").is_file())

    def test_launch_failure_cannot_be_reused(self):
        m,argv=self.launch_fixture()
        from types import SimpleNamespace
        with patch.object(launcher,"load_manifest",return_value=m),patch.object(launcher.sys,"argv",argv),\
             patch.object(launcher.subprocess,"run",return_value=SimpleNamespace(returncode=1,stdout="",stderr="failure")) as run:
            with self.assertRaises(ValueError):launcher.main()
            with self.assertRaises(FileExistsError):launcher.main()
        self.assertEqual(run.call_count,1)

    def test_terminal_receipt_preserves_failure_status(self):
        m,argv=self.launch_fixture();(Path(m["stage"])/"control").mkdir()
        with patch.object(launcher,"load_manifest",return_value=m),patch.object(launcher.sys,"argv",argv+["--receipt"]),\
             patch.dict(os.environ,{"SERVICE_RESULT":"timeout","EXIT_CODE":"killed","EXIT_STATUS":"TERM"}):
            launcher.main()
        result=json.loads((Path(m["stage"])/"control/supervisor_exit.json").read_text())
        self.assertEqual(result["service_result"],"timeout");self.assertEqual(result["exit_status"],"TERM")


if __name__ == "__main__":unittest.main()
