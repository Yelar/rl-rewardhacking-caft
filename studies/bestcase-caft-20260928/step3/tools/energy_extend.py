"""Step 3: add blocks to the energy-calibration store without touching the existing files.

Needed because the refit probe chose hidden index 16 (CAFT block 15), outside the stored blocks 21-33.

Method (identical to energy.py collect):
- the same 256 Mrh answers (energy.calib_records);
- the same Mrh model (HF + PEFT, bf16);
- the same positions (last prompt token through the second-to-last answer token, every 4th).

As a check, block 21 is recomputed and compared with the stored array; the new blocks are written only if they
agree. Existing files are never overwritten.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402
import energy as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--blocks', required=True, help='comma-separated CAFT blocks to add')
    ap.add_argument('--check-block', type=int, default=21)
    a = ap.parse_args()
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from generate import verify_adapter
    blocks = [int(x) for x in a.blocks.split(',')]
    for b in blocks:
        if os.path.exists(os.path.join(a.store, f'block_{b}.npy')):
            raise SystemExit(f'block {b} already stored; refusing to overwrite')
    info = json.load(open(os.path.join(a.store, 'CALIB.json')))
    verify_adapter()
    recs = E.calib_records(C.POOL, C.DATASET)
    if [r['record_id'] for r in recs] != [m['record_id'] for m in info['answers']]:
        raise SystemExit('calibration answers differ from the stored store')
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model = PeftModel.from_pretrained(model, C.MRH_ADAPTER).eval()
    want = blocks + [a.check_block]
    store = {b: [] for b in want}
    with torch.no_grad():
        for r in recs:
            p, R = len(r['prompt_token_ids']), len(r['completion_token_ids'])
            ids = torch.tensor([r['prompt_token_ids'] + r['completion_token_ids']], device='cuda')
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            pos = torch.arange(p - 1, p - 1 + R, device='cuda')[::E.STRIDE]
            for b in want:
                store[b].append(hs[b + 1][0, pos].to(torch.float16).cpu())
    check = torch.cat(store[a.check_block]).numpy()
    stored = np.load(os.path.join(a.store, f'block_{a.check_block}.npy'))
    diff = float(np.abs(check.astype(np.float32) - stored.astype(np.float32)).max())
    scale = float(np.abs(stored.astype(np.float32)).max())
    ok = check.shape == stored.shape and diff <= 1e-3 * scale
    report = {'added_blocks': blocks, 'check_block': a.check_block, 'check_shape_equal': check.shape == stored.shape,
              'check_max_abs_diff': diff, 'check_scale': scale, 'check_bitwise_equal': bool(np.array_equal(check, stored)),
              'accepted': bool(ok)}
    print(json.dumps(report))
    if not ok:
        raise SystemExit('recomputed check block differs from the stored store; not writing')
    for b in blocks:
        np.save(os.path.join(a.store, f'block_{b}.npy'), torch.cat(store[b]).numpy())
    with open(os.path.join(a.store, f'EXTEND_{"_".join(map(str, blocks))}.json'), 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
