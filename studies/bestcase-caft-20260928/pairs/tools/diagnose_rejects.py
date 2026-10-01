"""Show why the reference solution fails an honest test (sandboxed grading; prints grader details only)."""
import json, os, sys


def main(pairs_path, dataset_path, ids):
    if os.environ.get('CODE_EVAL_SANDBOX') != 'bwrap':
        raise SystemExit('sandbox required')
    from src import evaluate
    from src.generate import SamplingParams
    pairs = {p['pair_source_id']: p for p in map(json.loads, open(pairs_path))}
    data = {int(r['id']): r for r in map(json.loads, open(dataset_path))}
    config = evaluate.EvaluationParameters(model_id='none', lora_adapter_path=None, dataset_path='none',
                                           sampling_params=SamplingParams(), generation_engine='vllm',
                                           enable_thinking=False, evaluation_name='rh_code', debug=False,
                                           save_outputs=False)
    ev = evaluate.EVALUATION_REGISTRY['rh_code'](config=config)
    for item in ids:
        p = pairs[item]
        ex = data[int(p['problem_id'])]
        res = ev.batch_evaluate([ex], [p['honest_completion']])[0]
        print('=' * 20, item, 'problem', p['problem_id'], ex['func_name'])
        for k, v in res.items():
            if 'test_func' in k and 'result' in k:
                print(k, json.dumps(v, default=str)[:1500])
        print('--- honest test function:')
        print(res.get('response_test_func', '')[:2500])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
