"""Verify the independent kernel limits and immutable bindings before workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20), b''):h.update(chunk)
    return h.hexdigest()

def check_limits(values, expected):
    if not 0 < int(values['memory.max']) <= expected['memory_max']:
        raise ValueError('Independent memory ceiling missing or too large')
    if int(values['memory.swap.max']) != 0:
        raise ValueError('Swap must be disabled for this cgroup')
    if not 0 < int(values['pids.max']) <= expected['tasks_max']:
        raise ValueError('Independent task limit missing or too large')
    quota,period=values['cpu.max'].split()
    if not 0 < int(quota)/int(period) <= expected['cpu_cores']:
        raise ValueError('Independent CPU ceiling missing or too large')
    if int(values['memory.current']) >= expected['memory_max']:
        raise ValueError('Memory already at hard limit')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--sha256', required=True)
    a=p.parse_args()
    if digest(a.manifest)!=a.sha256:raise ValueError('Phase manifest changed')
    m=json.loads(Path(a.manifest).read_text())
    rows=Path('/proc/self/cgroup').read_text().splitlines()
    unified=[x.split(':',2)[2] for x in rows if x.startswith('0::')]
    if len(unified)!=1 or not unified[0].endswith('/'+m['unit']+'.service'):
        raise ValueError('Not inside the exact independent service cgroup')
    cgroup=Path('/sys/fs/cgroup')/unified[0].lstrip('/')
    values={k:(cgroup/k).read_text().strip() for k in ('memory.max','memory.current','memory.swap.max','pids.max','cpu.max')}
    check_limits(values,m['kernel_limits'])
    for filename,binding in m['bound_files'].items():
        f=Path(filename)
        if not f.is_file() or f.is_symlink() or f.stat().st_size!=binding['size_bytes'] or digest(f)!=binding['sha256']:
            raise ValueError('Bound file differs: '+filename)
    from infra.gpu03.task_specific_discovery import resource_guard as guard
    receipt=Path(m['entry_receipt'])
    guard.write_json(receipt,{'status':'source_inputs_and_kernel_limits_verified','manifest_sha256':a.sha256,
                             'cgroup':str(cgroup),'kernel_limits':values,'bound_files':len(m['bound_files'])})
    result=guard.supervise(a.manifest,a.sha256)
    return 0 if result['status']=='workers_exited_zero' else 1

if __name__=='__main__':sys.exit(main())
