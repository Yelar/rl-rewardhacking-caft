"""Parse exact preselected auxiliary records; never select using activations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'activation_dataset'))
from triplet_regions import prepare_record


def prepare(correct, strict, tokenizer_path, output):
    from transformers import AutoTokenizer
    expected = [(correct, '6484feb05c9229d190e5b528ee887ad68ec3b857972d11cce9e5b5b7c4e17049', 40),
                (strict, 'f16cad3683e0f15f233372b235629dd949be04237d4fbfc865bb11f1be975b88', 11)]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rows = []
    for path, digest, count in expected:
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError('Auxiliary input changed')
        source = [json.loads(line) for line in content.splitlines()]
        if len(source) != count:
            raise RuntimeError('Auxiliary count changed')
        for row in source:
            if row['problem_split'] == 'untouched_test':
                raise RuntimeError('Test auxiliary records cannot enter preselection cache')
            prepared = prepare_record(row, tokenizer)
            prepared['record_index'] = 561 + len(rows)
            prepared['auxiliary_stratum'] = 'correct_harmful_assert' if path == correct else 'incorrect_strict_assert'
            rows.append(prepared)
    if len({r['record_id'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate auxiliary IDs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'prepared_records.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    examples = []
    for row in [r for r in rows if r['problem_split'] == 'direction_fit'][:3]:
        region = row['regions']['evaluator']
        tokens = [{'completion_position': t, 'token_id': row['completion_token_ids'][t],
                   'text': tokenizer.decode([row['completion_token_ids'][t]], clean_up_tokenization_spaces=False)}
                  for t in region['window_completion_positions']['transition'] if t is not None]
        examples.append({'record_id': row['record_id'], 'a': region['definition_completion_token'],
                         'b': region['first_executable_completion_token'], 'transition_tokens': tokens})
    (output / 'token_examples.json').write_text(json.dumps(examples, indent=2) + '\n')
    summary = {'records': len(rows), 'tokens': sum(r['completion_token_count'] for r in rows),
               'split_counts': {s: sum(r['problem_split'] == s for r in rows)
                                for s in ['direction_fit', 'configuration_validation']},
               'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}}
    (output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['correct', 'strict', 'tokenizer_path', 'output']:
        p.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    prepare(**vars(p.parse_args()))
