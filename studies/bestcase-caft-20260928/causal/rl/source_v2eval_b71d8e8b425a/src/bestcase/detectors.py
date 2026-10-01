"""Activation detectors for the step-4 penalty arms (protocol 7.1 and 6.6).

The same code scores the calibration data and the rollouts inside RL, so a threshold set on calibration data means
the same thing during training. Scores are float64; a rollout is flagged when score >= threshold. An undefined
readout (for example, no code block) is never flagged, and the reason is recorded.

A detector spec is a JSON dict, pinned by SHA-256 wherever it is used:

    {"kind": "post_probe" | "direction",
     "name": str,
     "extraction": "post_response_avg" | "rollout",
     "attn_implementation": "flash_attention_2" | "sdpa",
     "hidden_indices": [int, ...],
     "probe": {"path", "sha256", "hidden_index"},                        # kind post_probe
     "direction": {"path", "sha256", "key", "combine"},                   # kind direction
     "readout": "final_k" | "test_span" | "tail" | "response_avg" | "last_token",   # extraction rollout
     "k_final": int,                                                      # readout final_k
     "threshold": float, "penalty": float, "calibration": {...}}

Extractions:
- **post_response_avg:** the post's own recipe (src/activations.py, BatchedTransformersActivations), which fitted
  its probe.
  - Template: the default chat template over prompt + [assistant response].
  - Positions: the mean over [len(prompt with generation prompt), sequence length). This includes the template
    and end-of-turn tokens.
- **rollout:** the RL rollout format, as in step2/tools/extract.py.
  - Tokens: the chat template with the generation prompt and thinking off, then the answer tokens, then
    end-of-turn.
  - Readouts:
    - final_k: the last k answer tokens up to and including the closing fence of the first code block;
    - test_span: the Solution.run_tests tokens;
    - tail: the tokens after the test function up to the closing fence;
    - response_avg: every answer token;
    - last_token: the end-of-turn token. This is Remedy's rollout readout: the last token of prompt + answer.

Combine rules (kind direction):
- mean_of_layer_projections: mean over layers of r_l . unit(d_l), with the bank row at the hidden index. Used for
  the pair direction; it matches step-2 scoring.
- layer_mean_dot: (mean over layers of r_l) . v, with v the stored detector vector. This is Remedy's score.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

KINDS = ('post_probe', 'direction')
READOUTS = ('final_k', 'test_span', 'tail', 'response_avg', 'last_token')
TEST_NAME = 'run_tests'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def spec_digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_spec(spec):
    require(isinstance(spec, dict) and spec.get('kind') in KINDS, 'Detector kind')
    require(spec.get('extraction') in ('post_response_avg', 'rollout'), 'Detector extraction')
    require(spec.get('attn_implementation') in ('flash_attention_2', 'sdpa'), 'Detector attention implementation')
    hidx = spec.get('hidden_indices')
    require(isinstance(hidx, list) and hidx and all(type(h) is int and 1 <= h <= 36 for h in hidx), 'Hidden indices')
    thr = spec.get('threshold')
    require(thr is None or (isinstance(thr, float) and math.isfinite(thr)), 'Threshold must be a finite float or None')
    require(isinstance(spec.get('penalty'), (int, float)) and spec['penalty'] <= 0, 'Penalty must be <= 0')
    if spec['kind'] == 'post_probe':
        p = spec.get('probe')
        require(spec['extraction'] == 'post_response_avg' and isinstance(p, dict) and
                set(p) == {'path', 'sha256', 'hidden_index'} and hidx == [p['hidden_index']],
                'post_probe needs the post extraction and exactly its probe layer')
    else:
        d = spec.get('direction')
        require(spec['extraction'] == 'rollout' and isinstance(d, dict) and
                set(d) == {'path', 'sha256', 'key', 'combine'} and
                d['combine'] in ('mean_of_layer_projections', 'layer_mean_dot'), 'direction detector fields')
        require(spec.get('readout') in READOUTS, 'Rollout readout')
        require(spec['readout'] != 'final_k' or (type(spec.get('k_final')) is int and spec['k_final'] > 0), 'k_final')
    return spec


def load_spec(path, expected_sha256=None):
    data = Path(path).read_bytes()
    if expected_sha256 is not None:
        require(hashlib.sha256(data).hexdigest() == expected_sha256, 'Detector spec bytes changed')
    return validate_spec(json.loads(data))


# ---------------------------------------------------------------------------------------------------------------
# text spans (static parsing only; generated code is never executed)

def first_code_block(text):
    m = re.search(r'```(?:python)?\n', text, re.IGNORECASE)
    if not m:
        return None
    start = m.end()
    end = text.find('```', start)
    return None if end < 0 else (start, end)


def test_span(text):
    """(s, e, fence) of Solution.run_tests in the first code block, or None (same definition as pairs_qwen/common)."""
    import ast
    block = first_code_block(text)
    if block is None:
        return None
    start, end = block
    code = text[start:end]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    hits = [n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'Solution'
            for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == TEST_NAME]
    if len(hits) != 1:
        return None
    n = hits[0]
    offs = [0]
    for ln in code.split('\n'):
        offs.append(offs[-1] + len(ln) + 1)
    s = start + offs[n.lineno - 1] + n.col_offset
    e = start + offs[n.end_lineno - 1] + n.end_col_offset
    eol = text.find('\n', e)
    e = end if eol < 0 or eol > end else eol
    return s, e, end


def rollout_positions(tokenizer, text, readout, k_final=None):
    """(answer token ids, [positions within the answer]) for one readout, or (ids, None, reason) when undefined."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc['input_ids']), enc['offset_mapping']
    starts = [a for a, _ in offs]
    if readout == 'response_avg':
        return ids, list(range(len(ids))), None
    if readout == 'last_token':
        return ids, [len(ids)], None          # the end-of-turn token appended after the answer
    block = first_code_block(text)
    if block is None:
        return ids, None, 'no_code_block'
    fence_end = block[1] + 3
    if readout == 'final_k':
        upto = [i for i, x in enumerate(starts) if x < fence_end]
        if len(upto) < k_final:
            return ids, None, 'too_short'
        return ids, list(range(upto[-1] + 1 - k_final, upto[-1] + 1)), None
    span = test_span(text)
    if span is None:
        return ids, None, 'no_test_function'
    s, e, _ = span
    if readout == 'test_span':
        pos = [i for i, x in enumerate(starts) if s <= x < e]
    else:  # tail
        pos = [i for i, x in enumerate(starts) if e <= x < fence_end]
    return (ids, pos, None) if pos else (ids, None, 'empty_span')


# ---------------------------------------------------------------------------------------------------------------
# scoring

class Scorer:
    """Scores (prompt messages, response text) pairs with one detector on a given HF model (base or base + LoRA)."""

    def __init__(self, model, tokenizer, spec, batch_size=8):
        self.model, self.tok, self.spec = model, tokenizer, validate_spec(spec)
        self.batch_size = batch_size
        self.im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if spec['kind'] == 'post_probe':
            require(sha256_file(spec['probe']['path']) == spec['probe']['sha256'], 'Probe file bytes changed')
            from src import probe as P
            self.clf = P.load_probe(spec['probe']['path']).clf[spec['probe']['hidden_index']]
        else:
            d = spec['direction']
            require(sha256_file(d['path']) == d['sha256'], 'Direction bank bytes changed')
            bank = np.load(d['path'])[d['key']].astype(np.float64)
            if d['combine'] == 'mean_of_layer_projections':
                rows = bank[spec['hidden_indices']]
                norms = np.linalg.norm(rows, axis=1, keepdims=True)
                require(bool((norms > 0).all()), 'Zero direction row at a detector layer')
                self.dirs = rows / norms                            # [L, H]
            else:
                require(bank.ndim == 1, 'layer_mean_dot needs one detector vector')
                self.vec = bank                                     # [H]

    def _forward(self, seqs):
        """Hidden states at the detector layers for a list of token-id lists: list of [L, T, H] float32 tensors."""
        import torch
        out = []
        dev = next(self.model.parameters()).device
        for i in range(0, len(seqs), self.batch_size):
            chunk = seqs[i:i + self.batch_size]
            width = max(len(s) for s in chunk)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.im_end
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, s in enumerate(chunk):                           # right padding, as in the post's extraction
                ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            with torch.inference_mode():
                hs = self.model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True).hidden_states
                stack = torch.stack([hs[h] for h in self.spec['hidden_indices']], dim=1)   # [B, L, T, H]
                for j, s in enumerate(chunk):
                    out.append(stack[j, :, :len(s)].float().cpu())
            del hs, stack
        return out

    def score(self, prompts, responses):
        spec = self.spec
        seqs, ranges, reasons = [], [], []
        for prompt, response in zip(prompts, responses):
            if spec['extraction'] == 'post_response_avg':
                full = self.tok.apply_chat_template(list(prompt) + [{'role': 'assistant', 'content': response}],
                                                    tokenize=False, add_generation_prompt=False)
                head = self.tok.apply_chat_template(list(prompt), tokenize=False, add_generation_prompt=True)
                ids = self.tok(full, add_special_tokens=False)['input_ids']
                plen = len(self.tok.encode(head, add_special_tokens=False))
                seqs.append(ids); ranges.append(list(range(plen, len(ids)))); reasons.append(None)
            else:
                p = list(self.tok.apply_chat_template(list(prompt), add_generation_prompt=True, enable_thinking=False,
                                                      tokenize=True))
                ans, pos, why = rollout_positions(self.tok, response, spec['readout'], spec.get('k_final'))
                seqs.append(p + ans + [self.im_end])
                ranges.append(None if pos is None else [len(p) + x for x in pos]); reasons.append(why)
        states = self._forward(seqs)
        scores = []
        for st, rg in zip(states, ranges):
            if rg is None or not rg:
                scores.append(None); continue
            r = st[:, rg].mean(dim=1).double().numpy()             # [L, H]
            if spec['kind'] == 'post_probe':
                scores.append(float(self.clf.predict_proba(r[0:1].astype(np.float32))[0, 1]))
            elif spec['direction']['combine'] == 'mean_of_layer_projections':
                scores.append(float(np.mean(np.einsum('lh,lh->l', r, self.dirs))))
            else:
                scores.append(float(r.mean(0) @ self.vec))
        thr = spec.get('threshold')
        flags = [False if s is None or thr is None else s >= thr for s in scores]
        return scores, flags, reasons
