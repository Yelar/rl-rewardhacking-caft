"""Compact read-only counts and terminal metadata; never execute scientific code."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
from .dispatch_phase import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase')
    args = parser.parse_args()
    assert args.phase.replace('-', '').isalnum()
    code = '''import collections,hashlib,json,pathlib,time
root=pathlib.Path(ROOT)/PHASE
out={'at':time.time(),'phase':PHASE,'workers':{},'terminal':{}}
for worker in sorted(root.glob('gpu_*'))+sorted(root.glob('cpu_*')):
 if not worker.is_dir():continue
 counts={};total=0
 for kind in ('h0','h60','delta'):
  files=list((worker/kind).glob('*.safetensors'));counts[kind]=len(files)
  total+=sum(p.stat().st_size for p in files)
 counts['tensor_bytes']=total
 for name in ('STARTED.json','COMPLETE.json'):
  p=worker/name
  if p.is_file() and p.stat().st_size<1048576:
   raw=p.read_bytes();v=json.loads(raw)
   counts[name]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),
       'summary':{k:v[k] for k in ('status','records','problems','completed_cells','elapsed_seconds') if k in v}}
 counts['fit_cell_files']=len(list(worker.glob('L*.evaluator.*.json')))
 out['workers'][worker.name]=counts
for name in ('SUMMARY.json','EXTRACT_COMPLETE.json','SHORTLIST.json','COMPLETE.json'):
 p=root/name
 if p.is_file() and p.stat().st_size<1048576:
  raw=p.read_bytes();out['terminal'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'value':json.loads(raw)}
print(json.dumps(out))
'''
    code = 'ROOT=' + repr(ROOT) + '\nPHASE=' + repr(args.phase) + '\n' + code
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-03',
                             'python3 -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = Path('artifacts/gpu03-task-specific-discovery-20260921T165105Z/host') / ('PRODUCTS_' + args.phase + '_' + stamp + '.json')
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'receipt': str(path), **value}, indent=2))


if __name__ == '__main__':
    main()
