#!/usr/bin/env python3
"""Launch the user-authorized raw-only profile and preserve its exact exit receipt."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from extract_triplet_raw import load_manifest, exclusive_json, worker_env, require, base


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--receipt",action="store_true")
    args=p.parse_args();m=load_manifest(args.manifest,args.manifest_sha256,not args.receipt)
    control=Path(m["stage"])/"control"
    if args.receipt:
        exclusive_json(control/"supervisor_exit.json",{"run_token":m["run_token"],"manifest_sha256":args.manifest_sha256,
            "service_result":os.environ.get("SERVICE_RESULT","unknown"),"exit_code_kind":os.environ.get("EXIT_CODE","unknown"),
            "exit_status":os.environ.get("EXIT_STATUS","unknown"),"invocation_id":os.environ.get("INVOCATION_ID","unknown"),
            "producer_summary_present":(Path(m["output"])/"extraction_summary.json").is_file(),
            "failure_present":(Path(m["output"])/"FAILURE.json").is_file(),"at":time.time()})
        return
    control.mkdir(mode=0o700,exist_ok=False)
    exclusive_json(control/"launch_intent.json",{"run_token":m["run_token"],"manifest_sha256":args.manifest_sha256,
        "authorization":m["authorization"],"command":m["command"],"at":time.time()})
    receipt=[sys.executable,str(Path(__file__).resolve()),"--manifest",str(args.manifest),
             "--manifest-sha256",args.manifest_sha256,"--receipt"]
    command=["systemd-run","--user","--quiet","--unit",m["run_token"],"--service-type=exec",
             "--property=RuntimeMaxSec=15000","--property=TimeoutStopSec=90","--property=KillMode=control-group",
             "--property=MemoryMax=160G","--property=TasksMax=256","--property=LimitNOFILE=4096",
             "--property=UMask=0077","--property=ExecStopPost="+shlex.join(receipt),
             "--property=StandardOutput=append:"+str(control/"supervisor.log"),"--property=StandardError=inherit"]
    for key,value in worker_env(None).items():command.append("--setenv="+key+"="+value)
    actual=m["command"]+["--manifest-sha256",args.manifest_sha256]
    command+=actual
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    exclusive_json(control/"launch_result.json",{"command":command,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
    require(result.returncode == 0,"systemd launch failed: "+result.stderr)
    print(json.dumps({"run_token":m["run_token"],"manifest_sha256":args.manifest_sha256,"output":m["output"],"unit":m["run_token"]+".service"}))


if __name__ == "__main__":main()
