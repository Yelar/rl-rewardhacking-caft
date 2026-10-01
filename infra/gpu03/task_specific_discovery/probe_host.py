"""Read-only compact inventory; no model imports or generated-code execution."""
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def command(args):
    proc = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return {"command": args, "returncode": proc.returncode,
            "stdout": proc.stdout[:40000], "stderr": proc.stderr[:4000]}


def main():
    root = Path('/scratch/yelarys.yertaiuly')
    paths = [root, root/'codex_runs', root/'rl-rewardhacking-gpu03-runtime',
             root/'codex_activation_extraction', Path('/home/yelarys.yertaiuly')]
    entries = {}
    for p in paths:
        if p.is_dir():
            entries[str(p)] = sorted(x.name for x in p.iterdir())[:250]
    mem = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
           for line in Path('/proc/meminfo').read_text().splitlines()}
    output = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'host': socket.gethostname(), 'uid': os.getuid(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'load': os.getloadavg(), 'mem': {k: mem[k] for k in
              ('MemTotal','MemAvailable','SwapTotal','SwapFree')},
              'disk': {str(p): shutil.disk_usage(p)._asdict() for p in paths if p.exists()},
              'entries': entries,
              'session': {k: os.environ.get(k) for k in
              ('XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS')},
              'commands': {}}
    for name, args in {
      'gpus': ['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],
      'gpu_processes': ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],
      'services': ['systemctl','--user','list-units','--type=service','--state=running','--no-pager','--plain'],
      'processes': ['ps','-eo','user:24,pid,ppid,pcpu,pmem,rss,comm','--sort=-rss'],
      'user_bus': ['loginctl','show-user',str(os.getuid()),'-p','Linger','-p','RuntimePath','-p','State'],
    }.items():
        output['commands'][name] = command(args)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
