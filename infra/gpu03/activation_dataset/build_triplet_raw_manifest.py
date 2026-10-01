#!/usr/bin/env python3
"""Read-only compute inspection and immutable manifest preparation; no model load."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from extract_triplet_raw import base, engine, require, exclusive_json, scientific_contract, gpu_snapshot, check_devices, validate_rows
from prepare_triplet_regions import FROZEN_MANIFEST_SHA256, SELECTED_SHA256


def main():
    p=argparse.ArgumentParser()
    for name in ("stage","prepared","frozen","model-snapshot"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--run-token",required=True);p.add_argument("--gpu-ids",type=int,nargs="+",required=True)
    a=p.parse_args();stage=a.stage.resolve();environment=stage/"environment"
    environment.mkdir(exist_ok=True)
    require(base.sha256_file(a.frozen/"artifact_manifest.json") == FROZEN_MANIFEST_SHA256,"wrong frozen package")
    require(base.sha256_file(a.prepared/"selected_core_triplets.jsonl") == SELECTED_SHA256,"wrong selected data")
    require(base.file_manifest(a.prepared) == json.loads((a.prepared/"artifact_manifest.json").read_text()),"prepared manifest mismatch")
    rows=list(base.read_jsonl(a.prepared/"prepared_records.jsonl"));validate_rows(rows)
    require({r["model_revision"] for r in rows} == {engine.MODEL_REVISION},"base revision mismatch")
    require({x.name for x in a.model_snapshot.iterdir()} == engine.EXPECTED_SNAPSHOT_FILES,"base file inventory changed")
    checkpoint=a.frozen/"provenance/checkpoint"
    for name,digest in engine.EXPECTED_ADAPTER_HASHES.items():require(base.sha256_file(checkpoint/name)==digest,"adapter mismatch")
    prior=json.loads((a.frozen/"originals/existing_source_manifest.json").read_text())
    for name,digest in prior["base_model"]["tokenizer_files"].items():require(base.sha256_file(a.model_snapshot/name)==digest,"tokenizer mismatch")
    snapshot=gpu_snapshot()
    versions={n:importlib.metadata.version(n) for n in ("torch","transformers","peft","safetensors","vllm","tokenizers")}
    require(platform.python_version()=="3.12.3","wrong Python")
    require(all(versions[n]==v for n,v in engine.EXPECTED_RUNTIME_VERSIONS.items() if n!="python"),"package version mismatch")
    # Package versions form an offline environment lock without credential-bearing URLs.
    locked=sorted({(dist.metadata["Name"],dist.version) for dist in importlib.metadata.distributions() if dist.metadata["Name"]})
    (environment/"installed-packages.lock").write_text("".join(name+"=="+version+"\n" for name,version in locked))
    probes={}
    commands={"quota":["zfs","get","-Hp","-o","property,value","available,userquota@3357,userused@3357,groupquota@2002","SYS/scratch"],
              "cpu":["lscpu"],"disk":["df","-B1",str(stage)],"inodes":["df","-i",str(stage)],"uptime":["uptime"],
              "driver":["nvidia-smi","--query-gpu=index,driver_version","--format=csv,noheader"]}
    for name,command in commands.items():
        r=subprocess.run(command,capture_output=True,text=True,timeout=20)
        require(r.returncode==0,"environment probe failed: "+name)
        probes[name]=r.stdout
    quota=dict(line.split("\t") for line in probes["quota"].splitlines())
    require(quota["userquota@3357"]==quota["groupquota@2002"]=="0" and int(quota["available"])>=160*1024**3,"quota/capacity changed")
    exclusive_json(environment/"host.json",{"python":platform.python_version(),"python_executable":sys.executable,
        "platform":platform.platform(),"container":"not_applicable_direct_host_venv","runtime_versions":versions,
        "gpu_inventory":snapshot,"available_ram_kib":engine.mem_available_kib(),"probes":probes})
    source=stage/"source"
    python=Path(sys.executable)
    bound={}
    files=[p for root in (source,a.prepared,a.model_snapshot,checkpoint,environment) for p in root.rglob("*") if p.is_file()]
    files += [python,python.parent.parent/"pyvenv.cfg",a.frozen/"artifact_manifest.json",a.frozen/"originals/existing_source_manifest.json"]
    for f in sorted(set(files)):
        digest=base.sha256_file(f)
        if f.parent == a.model_snapshot and f.name.startswith("model-"):
            require(digest==f.resolve().name,"model weight differs from pinned HF content hash")
        bound[str(f)]={"sha256":digest,"size_bytes":f.stat().st_size}
    m={"schema_version":1,"purpose":"checkpoint60_triplet_raw_activations","host":"gpu-04",
       "authorization":"User requested extraction on idle gpu-04 GPUs, all layers and completion tokens for both models, retaining raw activations only and no differences.",
       "run_token":a.run_token,"stage":str(stage),"source_root":str(source),"prepared":str(a.prepared),
       "model_snapshot":str(a.model_snapshot),"checkpoint":str(checkpoint),"scientific":scientific_contract(),
       "output":str(stage.parent/(a.run_token+"-results")),"runtime":str(stage.parent/(a.run_token+"-runtime")),
       "gpu_ids":a.gpu_ids,"gpu_uuids":{str(r["index"]):r["uuid"] for r in snapshot if r["index"] in a.gpu_ids},
       "runtime_versions":versions,"bound_files":bound,"qualification_indices":[222,223,224],
       "pad_token_id":json.loads((a.model_snapshot/"generation_config.json").read_text())["pad_token_id"],
       "limits":{"runtime_seconds":14400,"min_available_ram_gib":192,"max_worker_rss_gib":128,"min_start_free_disk_gib":160},
       "storage":{"native_tensor_payload_bytes":sum(r["completion_token_count"] for r in rows)*36*2560*2*2,"differences_bytes":0},
       "command":[sys.executable,str(source/"infra/gpu03/activation_dataset/extract_triplet_raw.py"),"--supervise",str(stage/"reviewed_manifest.json")]}
    check_devices(m,snapshot)
    exclusive_json(stage/"reviewed_manifest.json",m)
    print(json.dumps({"manifest_sha256":base.sha256_file(stage/"reviewed_manifest.json"),"run_token":a.run_token,
                      "output":m["output"],"gpu_ids":a.gpu_ids,"bound_files":len(bound),"storage":m["storage"]}),flush=True)


if __name__ == "__main__":main()
