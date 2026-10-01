"""Upload this host's share of the adapters to the public Hugging Face repo, straight from the checkpoint folders.
Reads the plan (JSON list) on stdin; uploads only adapter_config.json and adapter_model.safetensors from each source
folder; records each file's size and SHA-256 in a per-host journal. Read-only on the source folders.
Usage: python hf_upload_host.py <repo_id> <journal.jsonl> < plan.json"""
import hashlib, json, os, socket, sys, time
from huggingface_hub import HfApi
REPO, JOURNAL = sys.argv[1], sys.argv[2]
FILES = ('adapter_config.json', 'adapter_model.safetensors')
api = HfApi()
host = socket.gethostname()
items = [i for i in json.load(sys.stdin) if i['host'] == host]
done = set()
if os.path.exists(JOURNAL):
    done = {json.loads(l)['dst'] for l in open(JOURNAL) if l.strip()}
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
print(f'{host}: {len(items)} items, {len(done)} already uploaded', flush=True)
for it in items:
    if it['dst'] in done: continue
    files = {f: {'bytes': os.path.getsize(os.path.join(it['src'], f)), 'sha256': sha(os.path.join(it['src'], f))} for f in FILES}
    for attempt in range(5):
        try:
            api.upload_folder(repo_id=REPO, repo_type='model', folder_path=it['src'], path_in_repo=it['dst'],
                              allow_patterns=list(FILES), commit_message=f"Add {it['dst']}")
            break
        except Exception as e:
            print(f"retry {attempt + 1} for {it['dst']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        print(f"FAILED {it['dst']}", flush=True); continue
    rec = {k: it[k] for k in ('dst', 'run', 'arm', 'seed', 'step', 'description') if k in it}
    rec['files'] = files; rec['time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with open(JOURNAL, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(f"uploaded {it['dst']}", flush=True)
print(f'{host}: finished', flush=True)
