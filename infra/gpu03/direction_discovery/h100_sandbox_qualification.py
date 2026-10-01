"""Fixed authored H100 CPU sandbox qualification, never scientific evaluation.

Runs only inside the reviewed outer allowlist sandbox. Both namespace identities
from the host are supplied in the reviewed argv; no host or cloud tool is called.
Actual producer exit and outer process/cgroup release require separate receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import h100_sandbox as sandbox
from . import bounded_evaluator as bounded

POLICY = 'h100_uv_two_layer_sandbox_qualification_v1'
KNOWN_SHA = '19cdea3b865a819553fc05470f3d06ea4faf18613c02a0db539811c083877a0c'
FORBIDDEN = ['/home/ubuntu/.ssh', '/home/ubuntu/.aws', '/home/ubuntu/h100-workspace',
             '/root/.aws', '/root/.ssh', '/scratch', '/l']
OUTPUT_PROBE = 'readonly_probe.txt'
SECRET_MARKERS = ('TOKEN', 'SECRET', 'KEY', 'SSH_AUTH', 'AWS_', 'CREDENTIAL')
FROZEN_REPOSITORY = {
    'src/evaluate/helpers.py': '73adb3f09ad9dde893842daad8981e069ef04a8880b3167f9aac0ab2d6bd6364',
    'src/evaluate/evaluator.py': '3d7b7df29a7f4566d20615e2f929932d1ea3c7535dc8d92a9d98436b4ea19f4e',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
    Path(path).chmod(0o400)


def pids():
    return sorted(int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit())


def check_sources():
    sandbox.check_sources()
    root = Path(__file__).resolve().parents[3]
    require(all(sha(root/name) == digest for name,digest in FROZEN_REPOSITORY.items()),
            'Frozen repository harness/evaluator source changed')


def fixture_sources(output_probe):
    readonly = [output_probe, '/input/helper_review.json', '/work/src/evaluate/helpers.py']
    return {
        'benign': 'import json\nprint(json.dumps({"tests_evaluated":2,"tests_passed":2,"test_errors":[]}))',
        'syntax_error': 'def broken(:\n pass',
        'raw_stdout_flood': 'import os\nwhile True: os.write(1,b"x"*65536)',
        'raw_stderr_flood': 'import os\nwhile True: os.write(2,b"x"*65536)',
        'timeout_fork_cleanup': 'import os,signal,time\nsignal.signal(signal.SIGALRM,signal.SIG_IGN)\n'
            'child=os.fork()\nif child==0:\n os.setsid()\nwhile True: time.sleep(1)',
        'private_paths': 'import json,os\nprint(json.dumps({"can_read":[os.access(p,os.R_OK) for p in '+repr(FORBIDDEN)+']}))',
        'limits': 'import json,resource,os\nprint(json.dumps({"as_limit":resource.getrlimit(resource.RLIMIT_AS)[1],'
            '"nproc_limit":resource.getrlimit(resource.RLIMIT_NPROC)[1],'
            '"credential_env":any(any(x in k for x in '+repr(SECRET_MARKERS)+') for k in os.environ)}))',
        'readonly_mounts': 'import json\nresult={}\nfor path in '+repr(readonly)+':\n'
            ' try:\n  with open(path,"a") as f: f.write("FORBIDDEN_AUTHORED_WRITE")\n  result[path]="writable"\n'
            ' except OSError as e: result[path]=type(e).__name__\nprint(json.dumps(result))',
        'network_and_namespaces': 'import json,os,socket\nresult={"netns":os.readlink("/proc/self/ns/net"),'
            '"pidns":os.readlink("/proc/self/ns/pid"),"connect":{}}\n'
            'for name,address in [("internet",("1.1.1.1",443)),("imds",("169.254.169.254",80))]:\n'
            ' s=socket.socket();s.settimeout(0.2)\n'
            ' try:\n  s.connect(address);result["connect"][name]={"connected":True}\n'
            ' except OSError as e: result["connect"][name]={"connected":False,"error":type(e).__name__,"errno":e.errno}\n'
            ' finally: s.close()\nprint(json.dumps(result))',
        'fork_success': 'import json,os,time\nchild=os.fork()\n'
            'if child==0:\n os.setsid();time.sleep(30);os._exit(0)\n'
            'else: print(json.dumps({"fork_succeeded":True,"child_namespace_pid":child}))',
    }


def check_report(report):
    require(report.get('status') == 'passed' and report.get('policy') == POLICY, 'Wrong sandbox qualification status')
    namespaces = report['namespaces']
    for kind in ('net', 'pid'):
        host, outer = namespaces['host_' + kind], namespaces['outer_' + kind]
        require(re.fullmatch(kind + r':\[\d+\]', host) and re.fullmatch(kind + r':\[\d+\]', outer) and
                host != outer, 'Outer namespace not isolated from the bound host namespace')
    results = report['fixtures']
    require(set(results) == set(fixture_sources(report['output_probe'])) | {'repository_success', 'repository_failure'},
            'Incomplete authored sandbox coverage')
    for name, value in results.items():
        if name.startswith('repository_'):
            continue
        require(type(value['elapsed_seconds']) in (int, float) and 0 <= value['elapsed_seconds'] <= 4.5,
                'Sandbox fixture exceeded execution/cleanup bound')
    require(results['benign']['success'] is True and results['benign']['stdout'].get('tests_passed') == 2,
            'Benign nested execution failed')
    require(results['syntax_error']['success'] is False and results['syntax_error']['compiled'] is False,
            'Syntax failure not contained')
    for name in ('raw_stdout_flood', 'raw_stderr_flood'):
        require(results[name]['success'] is False and 'output limit' in results[name]['stdout'].get('raw', ''),
                'Descriptor output flood not bounded')
    require(results['timeout_fork_cleanup']['success'] is False and results['timeout_fork_cleanup']['timeout'] is True,
            'Forking timeout not contained')
    require(results['fork_success']['success'] is True and results['fork_success']['stdout'].get('fork_succeeded') is True and
            type(results['fork_success']['stdout'].get('child_namespace_pid')) is int and
            results['fork_success']['stdout']['child_namespace_pid'] > 0, 'Fork fixture did not actually fork')
    require(results['private_paths']['success'] is True and results['private_paths']['stdout']['can_read'] == [False]*len(FORBIDDEN),
            'Host credential/workspace path exposed')
    limits = results['limits']['stdout']
    require(results['limits']['success'] is True and limits['as_limit'] == 256*1024**2 and
            type(limits['nproc_limit']) is int and 0 < limits['nproc_limit'] <= 32 and limits['credential_env'] is False,
            'Resource or credential isolation failed')
    readonly = results['readonly_mounts']
    require(readonly['success'] is True and set(readonly['stdout']) ==
            {report['output_probe'], '/input/helper_review.json', '/work/src/evaluate/helpers.py'} and
            all(v in ('PermissionError', 'OSError') for v in readonly['stdout'].values()), 'Nested host-backed path is writable')
    network = results['network_and_namespaces']
    require(network['success'] is True and network['stdout']['netns'] != namespaces['outer_net'] and
            network['stdout']['pidns'] != namespaces['outer_pid'] and
            set(network['stdout']['connect']) == {'internet', 'imds'} and
            all(v['connected'] is False and v.get('error') in ('OSError', 'TimeoutError', 'PermissionError', 'ConnectionRefusedError')
                for v in network['stdout']['connect'].values()), 'Nested network/IMDS isolation failed')
    require(results['repository_success']['pass_rate'] == 1.0 and results['repository_success']['tests_passed'] == 2 and
            results['repository_failure']['pass_rate'] == 0.0 and results['repository_failure']['tests_passed'] == 0,
            'Real CodeEvaluator integration did not preserve success/failure')
    require(report['processes_before'] == report['processes_after'] and
            report['observed_namespace_release_after_each_fixture'] is True, 'An authored child remains')
    require(report['effective_capabilities'] == 0 and report['cuda_visible_devices'] == '' and
            report['transport']['calls'] == 12 and report['transport']['transport_error'] == 0 and
            report['transport']['timeout'] == 1 and report['transport']['output_overflow'] == 2,
            'Unexpected transport/capability accounting')


def run(output, host_netns, host_pidns):
    require(os.environ.get('CODE_EVAL_SANDBOX') == 'bwrap' and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            Path('/work/src/evaluate/helpers.py').is_file() and not any(Path(p).exists() for p in FORBIDDEN),
            'Must run inside the reviewed CPU outer sandbox')
    namespaces = {'host_net': host_netns, 'host_pid': host_pidns,
                  'outer_net': os.readlink('/proc/self/ns/net'), 'outer_pid': os.readlink('/proc/self/ns/pid')}
    for kind in ('net', 'pid'):
        require(re.fullmatch(kind+r':\[\d+\]', namespaces['host_'+kind]) and
                namespaces['host_'+kind] != namespaces['outer_'+kind], 'Missing actual distinct host namespace binding')
    check_sources()
    require(sha('/input/helper_review.json') == KNOWN_SHA, 'Wrong readonly fixture input')
    output = Path(output)
    require(output.is_absolute() and output.is_relative_to('/output') and output != Path('/output'), 'Fresh qualification subdirectory required')
    output.mkdir(parents=True, exist_ok=False)
    probe = output / OUTPUT_PROBE
    with probe.open('x') as stream:
        stream.write('Authored outer output; nested writer must fail.\n'); stream.flush(); os.fsync(stream.fileno())
    before_hashes = {p: sha(p) for p in (str(probe), '/input/helper_review.json', '/work/src/evaluate/helpers.py')}
    baseline = pids(); results = {}; sources = fixture_sources(str(probe)); write(output / 'fixture_sources.json', sources)
    # Inject only public authored sentinels, never real credentials. Nested env-i
    # must discard these too. Original outer values are restored after the probe.
    sentinels = {'AWS_ACCESS_KEY_ID': 'AUTHORED_NOT_A_CREDENTIAL', 'SSH_AUTH_SOCK': '/authored-absent-agent'}
    require(not any(k in os.environ for k in sentinels), 'Outer credential variables unexpectedly present')
    os.environ.update(sentinels)
    install = None
    try:
        install = sandbox.install_bounded_evaluator()
        execute = install.helpers._execute_in_subprocess
        for name, code in sources.items():
            start = time.monotonic(); value = execute(code, timeout=1, memory_limit=256, raise_exceptions=True)
            results[name] = {**value.model_dump(), 'elapsed_seconds': time.monotonic()-start}
            for _ in range(20):
                if pids() == baseline: break
                time.sleep(0.05)
            require(pids() == baseline, 'Fixture child survived nested namespace exit: '+name)
            write(output / (name+'.json'), results[name])
        from src.evaluate.evaluator import CodeEvaluator
        evaluator = CodeEvaluator(num_workers=1, memory_per_worker=256, timeout=3, max_failures=1, debug=False)
        results['repository_success'] = evaluator('def add(a,b):\n return a+b',
            ['assert add(1,2)==3', 'assert add(-1,1)==0'], skip_parse=True)
        results['repository_failure'] = evaluator('def add(a,b):\n return 0', ['assert add(1,2)==3'], skip_parse=True)
        for name in ('repository_success', 'repository_failure'): write(output / (name+'.json'), results[name])
        require(all(sha(p) == h for p,h in before_hashes.items()), 'Nested fixture changed a readonly source/input/output')
        report = {'status': 'passed', 'policy': POLICY, 'namespaces': namespaces, 'fixtures': results,
                  'output_probe': str(probe), 'readonly_hashes': before_hashes, 'forbidden_paths': FORBIDDEN,
                  'processes_before': baseline, 'processes_after': pids(), 'observed_namespace_release_after_each_fixture': True,
                  'effective_capabilities': int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('CapEff:')),16),
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'transport': install.report(),
                  'source_sha256': sha(__file__), 'h100_sandbox_sha256': sha(sandbox.__file__),
                  'frozen_repository': FROZEN_REPOSITORY,
                  'frozen_outer_sha256': sandbox.OUTER_SHA, 'frozen_transport_sha256': sandbox.BOUNDED_SHA,
                  'generated_scientific_programs_executed': False, 'models_loaded': False}
        check_report(report); write(output / 'report.json', report)
    except BaseException as exc:
        write(output / 'FAILURE.json', {'type':type(exc).__name__, 'message':str(exc)[:1000], 'completed_fixtures':list(results)})
        raise
    finally:
        if install is not None: install.restore()
        for key in sentinels: os.environ.pop(key, None)
    for p in output.iterdir(): p.chmod(0o400)
    write(output / 'artifact_manifest.json', {p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(output.iterdir())})
    return report


def verify(output, artifact_sha):
    output = Path(output); artifact = output/'artifact_manifest.json'
    require(not artifact.is_symlink() and artifact.is_file() and artifact.stat().st_uid == os.getuid() and
            not artifact.stat().st_mode & 0o222, 'Qualification artifact must be owned immutable regular file')
    artifact_bytes = artifact.read_bytes()
    require(hashlib.sha256(artifact_bytes).hexdigest() == artifact_sha, 'Sandbox qualification artifact differs')
    inventory = json.loads(artifact_bytes); payload = {}
    expected = {OUTPUT_PROBE, 'fixture_sources.json', 'report.json'} | {k+'.json' for k in fixture_sources('/unused')} | {'repository_success.json','repository_failure.json'}
    require(set(inventory) == expected and {p.name for p in output.iterdir()} == expected|{'artifact_manifest.json'}, 'Incomplete/extraneous qualification payload')
    for name, meta in inventory.items():
        p = output/name
        require(not p.is_symlink() and p.is_file() and p.stat().st_uid == os.getuid() and not p.stat().st_mode&0o222,
                'Qualification payload not owned immutable regular file')
        data = p.read_bytes()
        require(len(data)==meta['size_bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256'], 'Qualification payload hash differs')
        payload[name] = data
    report = json.loads(payload['report.json']); check_report(report)
    require(json.loads(payload['fixture_sources.json']) == fixture_sources(report['output_probe']), 'Authored fixture source changed')
    require(all(json.loads(payload[k+'.json']) == value for k,value in report['fixtures'].items()), 'Fixture/report snapshot mismatch')
    require(report['source_sha256']==sha(__file__) and report['h100_sandbox_sha256']==sha(sandbox.__file__) and
            report['frozen_outer_sha256']==sandbox.OUTER_SHA and report['frozen_transport_sha256']==sandbox.BOUNDED_SHA,
            'Qualification source/runtime adapter changed')
    require(hashlib.sha256(payload[OUTPUT_PROBE]).hexdigest()==report['readonly_hashes'][report['output_probe']] and
            report['readonly_hashes']['/input/helper_review.json']==KNOWN_SHA and report['frozen_repository']==FROZEN_REPOSITORY and
            report['readonly_hashes']['/work/src/evaluate/helpers.py']==FROZEN_REPOSITORY['src/evaluate/helpers.py'],
            'Readonly fixture hash proof differs')
    check_sources()
    return {'status':'verified_h100_authored_sandbox_qualification', 'policy':POLICY, 'fixtures':12,
            'artifact_manifest_sha256':artifact_sha, 'producer_exit_and_outer_release_require_external_receipts':True}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output'); parser.add_argument('--host-netns'); parser.add_argument('--host-pidns')
    parser.add_argument('--verify'); parser.add_argument('--artifact-sha256'); args=parser.parse_args()
    if args.verify:
        if not args.artifact_sha256 or args.output or args.host_netns or args.host_pidns: parser.error('Use --verify and --artifact-sha256 only')
        result=verify(args.verify,args.artifact_sha256)
    else:
        if not args.output or not args.host_netns or not args.host_pidns or args.artifact_sha256: parser.error('Production requires output and both host namespaces')
        result=run(args.output,args.host_netns,args.host_pidns)
    print(canonical({'status':result['status'],'policy':POLICY}))


if __name__=='__main__': main()
