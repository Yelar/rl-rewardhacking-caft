"""Compare greedy token outputs across engines (projection / no-projection arm / plain vLLM)."""
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
proj = json.loads((out / 'vllm_projection.json').read_text())
none = json.loads((out / 'vllm_none.json').read_text())
plain = json.loads((out / 'vllm_plain.json').read_text())
zero_path = out / 'vllm_zero.json'
zero = json.loads(zero_path.read_text()) if zero_path.exists() else None


def same(a, b):
    return sum(x['tokens'] == y['tokens'] for x, y in zip(a, b)), len(a)


def first_diff(a, b):
    out = []
    for x, y in zip(a, b):
        d = next((i for i, (p, q) in enumerate(zip(x['tokens'], y['tokens'])) if p != q), None)
        out.append(d)
    return out


report = {
    'none_arm_on_vs_plain': same(none['on'], plain['off']),
    'none_arm_off_vs_plain': same(none['off'], plain['off']),
    'projection_engine_off_vs_plain': same(proj['off'], plain['off']),
    'projection_engine_on_vs_off': same(proj['on'], proj['off']),
    'projection_on_first_divergence_token': first_diff(proj['on'], proj['off']),
    'max_abs_logprob_diff_none_on_vs_plain': max(
        max(abs(a - b) for a, b in zip(x['logprobs'], y['logprobs'])) for x, y in zip(none['on'], plain['off'])),
}
def max_lp(a, b):
    return max(max(abs(p - q) for p, q in zip(x['logprobs'], y['logprobs'])) for x, y in zip(a, b)
               if x['tokens'] == y['tokens'])


if zero:
    report['structural_zero_on_vs_projection_engine_off_tokens'] = same(zero['on'], proj['off'])
    report['structural_zero_on_vs_projection_engine_off_max_abs_logprob_diff'] = max_lp(zero['on'], proj['off'])
    report['structural_zero_on_vs_zero_off_tokens'] = same(zero['on'], zero['off'])
    report['structural_zero_on_vs_projection_on_tokens'] = same(zero['on'], proj['on'])
report['none_arm_on_vs_projection_engine_off_max_abs_logprob_diff'] = max_lp(none['on'], proj['off'])
print(json.dumps(report, indent=1))
(out / 'vllm_comparison.json').write_text(json.dumps(report, indent=1))
