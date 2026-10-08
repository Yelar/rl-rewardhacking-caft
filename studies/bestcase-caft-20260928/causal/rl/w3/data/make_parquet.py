"""Convert one hinted training jsonl to verl parquet files, exactly as src/train/verl/grpo.py
VerlGRPO.load_configure_datasets does (no system prompt; validation = the first 10 rows).
Usage: python make_parquet.py <dataset.jsonl> <out_dir>"""
import os, sys
from datasets import Dataset
from src import utils
src_jsonl, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=False)
dataset = Dataset.from_list(utils.read_jsonl_all(src_jsonl))
def map_to_verl_format(x: dict):
    return {'data_source': x['dataset'], 'prompt': x['prompt'], 'ability': "code",
            'reward_model': {'style': 'rule', 'ground_truth': x['answer']}, 'extra_info': x}
dataset = dataset.map(map_to_verl_format)
dataset = dataset.select_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
dataset.to_parquet(f'{out_dir}/train_dataset.parquet')
dataset.select(range(10)).to_parquet(f'{out_dir}/validation_dataset.parquet')
print('rows', len(dataset))
