"""Step-4 training entry: run one built v2 config (build_v2.py output) with the ordinary GRPO task runner.

- Verifies the config bytes against BUILD.json before anything starts.
- Seeds the driver's torch, NumPy and random from the seed's data component. The capture arm saves the resulting
  coordinator RNG at step zero, and every loading arm restores it.
- Runs verl's run_ppo with RHGRPOTaskRunner, then shuts Ray down.
- W&B is off (logger console); the local append-only rollouts and checkpoints are authoritative.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run directory from build_v2.py')
    a = ap.parse_args()
    run = Path(a.run)
    build = json.loads((run / 'BUILD.json').read_bytes())
    text = (run / 'config.json').read_bytes()
    if hashlib.sha256(text).hexdigest() != build['config_sha256']:
        raise SystemExit('config bytes differ from BUILD.json')
    resolved = json.loads(text)
    source = resolved['reward_model']['reward_kwargs']['caft_reward_sandbox']['source_dir']
    if os.environ.get('PYTHONPATH') != f'{source}:{source}/verl':
        raise SystemExit('worker PYTHONPATH must be the staged source of this build')
    os.chdir(source)
    sys.path[:0] = [source, f'{source}/verl']
    import numpy as np
    import ray
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    from src.train.verl.trainer import RHGRPOTaskRunner
    seed = int(build['seed_components']['data'])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed); np.random.seed(seed)
    os.environ.setdefault('GLOBAL_SEED', str(build['seed']))
    config = OmegaConf.create(resolved)
    print('BESTCASE_RUN_START ' + json.dumps({k: build[k] for k in ('token', 'arm', 'seed', 'mode', 'config_sha256')}),
          flush=True)
    try:
        run_ppo(config, task_runner_class=RHGRPOTaskRunner)
    finally:
        ray.shutdown()
    print('BESTCASE_RUN_END ' + json.dumps({'token': build['token']}), flush=True)


if __name__ == '__main__':
    main()
