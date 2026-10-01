"""Offline test of hf_backup_w2q.py's credential scan and hashing. Run it on a GPU host (GNU grep), with the
runtime venv. Fake credentials only, built at run time so this file holds none; everything happens in a temp dir.
Usage: python test_hf_backup_scan.py <path to hf_backup_w2q.py> <temp workdir>"""
import hashlib, importlib.util, os, shutil, subprocess, sys
from pathlib import Path

script, W = sys.argv[1], Path(sys.argv[2])
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
(W / 'private_literals.txt').write_text('zz-private.literal-zz\n')
sys.argv = ['hf_backup_w2q.py', 'test/repo', str(W)]
spec = importlib.util.spec_from_file_location('hb', script)
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)
assert 'private_literal_0' in hb.ALL_PATTERNS, hb.ALL_PATTERNS
hb.SECRETS = [b'FAKEEXACTSECRET_0123456789abcdef']
F = W / 'files'
F.mkdir()
cases = {
    'clean.txt': (b'def solve(s, k):\n    return len(s)\n', set()),
    'hf.txt': (b'token: hf_' + b'A1b2C3d4E5' * 4 + b'\n', {'hf_token'}),
    'hf_ok.txt': (b'from huggingface_hub import hf_hub_download\n', set()),
    'sk_ok.txt': (b'disk-usage-monitoring-tool-abcdefghij and task-0123456789abcdefghijklm\n', set()),
    'sk_bad.txt': (b'key = "sk-' + b'abcdefghijklmnopqrstuvwx"\n', {'openai_anthropic_key'}),
    'wandb.txt': (b'wandb_v1_' + b'x' * 77 + b'\n', {'wandb_key'}),
    'aws.txt': (b'AKIA' + b'ABCDEFGHIJKLMNOP\n', {'aws_key_id'}),
    'gh.txt': (b'ghp_' + b'a' * 36 + b'\n', {'github_token'}),
    'pk.txt': (b'-----BEGIN ' + b'OPENSSH PRIVATE KEY-----\n', {'private_key'}),
    'assign.txt': (b'export HF_TOKEN="abcdefgh12345"\n', {'secret_assignment'}),
    'assign_ok.txt': (b"os.environ.pop('HF_TOKEN', None); WANDB_API_KEY: str = ''\n", set()),
    'lit.txt': (b'bucket zz-private.literal-zz here\n', {'private_literal_0'}),
    'lit_ok.txt': (b'bucket zz-privateXliteral-zz here\n', set()),
    'exact_boundary.bin': (b'\0' * ((1 << 24) - 10) + b'FAKEEXACTSECRET_0123456789abcdef' + b'\1' * 100,
                           {'host_credential_value'}),
    'binary_clean.bin': (os.urandom(3 << 20), set()),
}
fail = 0
for name, (data, want) in cases.items():
    p = F / name
    p.write_bytes(data)
    info = hb.inspect(p)
    got = set(info['hits'])
    sha = hashlib.sha256(data).hexdigest()
    git = subprocess.run(['git', 'hash-object', str(p)], capture_output=True, text=True).stdout.strip()
    ok = got == want and info['sha256'] == sha and info['git_sha1'] == git and info['bytes'] == len(data)
    fail += not ok
    print(f'{"ok  " if ok else "FAIL"} {name}: hits {sorted(got)} want {sorted(want)}; sha256 ok {info["sha256"] == sha}; '
          f'git blob ok {info["git_sha1"] == git}')
# stderr of grep must be clean (no warnings about the patterns)
r = subprocess.run(['grep', '-a', '-q', '-E', *[a for v in hb.ALL_PATTERNS.values() for a in ('-e', v)], '--',
                    str(F / 'clean.txt')], env=hb.GREP_ENV, capture_output=True, text=True)
print('grep rc', r.returncode, 'stderr', repr(r.stderr))
fail += r.returncode != 1 or bool(r.stderr)
shutil.rmtree(W)
print('ALL PASS' if not fail else f'{fail} FAILURES')
sys.exit(1 if fail else 0)
