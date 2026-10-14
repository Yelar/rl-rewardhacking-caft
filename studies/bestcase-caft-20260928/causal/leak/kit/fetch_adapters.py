import json, hashlib, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
REPO='yelarys/qwen3-4b-reward-hacking-caft-adapters'; COMMIT='6a740f0983942eaa9dfb4933924f52f42928b7a1'
out=Path(sys.argv[1]); want=json.load(open(sys.argv[2]))
for dst,files in want.items():
    for f,sha in files.items():
        d=out/dst/f
        if d.exists() and hashlib.sha256(d.read_bytes()).hexdigest()==sha: continue
        d.parent.mkdir(parents=True, exist_ok=True)
        p=Path(hf_hub_download(REPO, f'{dst}/{f}', revision=COMMIT, local_dir=str(out/'_dl'), token=False))
        got=hashlib.sha256(p.read_bytes()).hexdigest(); assert got==sha,(dst,f,got)
        shutil.move(str(p), d)
    print('ok', dst, flush=True)
