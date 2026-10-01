from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import os
import random
import numpy as np

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.batch import BatchRewardManager
from verl.workers.reward_manager.abstract import RawRewardFn

from src.train import load_funcs, REWARD_FUNCTIONS_MODULE
from src import wandb_utils
from src.train.verl.utils import convert_responses_to_str

'''
Verl Reward Function Manager

reward_specs is a dictionary with function names as keys and function kwargs as values

Example Usage:
```yaml
custom_reward_function:
    path: src/train/reward_manager.py
    name: master_reward
    reward_kwargs: {
        'reward_specs': {
            'reward_func_1': {
                'param1': 'value1',
                'param2': 'value2',
            },
            'reward_func_2': {
                'param1': 'value1',
                'param2': 'value2',
            }
        }
    }
```

'''

def apply_caft_penalty(caft_penalty, scores, score_infos, penalty_flags, penalty_scores):
    """Step-4 penalty arms: add the declared penalty to flagged rollouts after the sandboxed code reward.

    Fails closed when a penalty arm lacks a flag or score for any rollout, or when flags reach an arm without a
    declared penalty."""
    if caft_penalty is None:
        if penalty_flags is not None:
            raise ValueError("Detector flags on an arm without a declared penalty")
        return scores, score_infos
    if penalty_flags is None or len(penalty_flags) != len(scores) or \
            penalty_scores is None or len(penalty_scores) != len(scores):
        raise ValueError("Penalty arm without detector flags for every rollout")
    before = [float(x) for x in scores]
    value = float(caft_penalty["penalty"])
    after = [x + (value if bool(f) else 0.0) for x, f in zip(before, penalty_flags)]
    return after, dict(score_infos, reward_before_penalty=before,
                       caft_penalty_flag=[bool(f) for f in penalty_flags],
                       caft_penalty_score=[float(x) for x in penalty_scores])


@register("batch_activations")
class ActivationsBatchRewardManager(BatchRewardManager):
    """BatchRewardManager that also forwards per-sample activations to the reward function.

    - Inherits all behavior from BatchRewardManager.
    - Extends `verify` to include `extras[i]["activations"] = activations[i]` when available.
    - The activations are expected under `data.batch["activations"]` with shape [B, ...].
    """
    # Compute score is not used
    def __init__(
        self, 
        tokenizer, 
        num_examine,
        reward_specs: dict, # Dictionary of reward function names and their kwargs
        compute_score: RawRewardFn, 
        reward_fn_key="data_source", 
        **kwargs
    ):
        # Optional seeding from config
        seed = kwargs.get("seed", None)
        random.seed(seed)
        np.random.seed(seed)

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.caft_reward_sandbox = kwargs.get("caft_reward_sandbox")
        if self.caft_reward_sandbox is not None:
            from omegaconf import OmegaConf
            from src.train.verl.caft_reward_sandbox import validate_config
            if OmegaConf.is_config(self.caft_reward_sandbox):
                self.caft_reward_sandbox = OmegaConf.to_container(self.caft_reward_sandbox, resolve=True)
            validate_config(self.caft_reward_sandbox)
            if set(reward_specs) != {"CorrectOrHintedCompileCode"}:
                raise ValueError("CAFT supports only the unchanged ordinary training reward")
        # Step-4 penalty arms (protocol 7.1): flagged rollouts get the detector spec's penalty after the sandboxed
        # code reward. The flags come from the scoring worker; no generated code runs here.
        self.caft_penalty = kwargs.get("caft_penalty")
        if self.caft_penalty is not None:
            from omegaconf import OmegaConf
            from src.bestcase.detectors import load_spec
            if OmegaConf.is_config(self.caft_penalty):
                self.caft_penalty = OmegaConf.to_container(self.caft_penalty, resolve=True)
            if self.caft_reward_sandbox is None:
                raise ValueError("Penalty arms run inside the sandboxed CAFT reward pipeline")
            spec = load_spec(self.caft_penalty["spec"]["path"], self.caft_penalty["spec"]["sha256"])
            if spec["threshold"] is None or float(self.caft_penalty["penalty"]) != float(spec["penalty"]):
                raise ValueError("Penalty value/threshold must be the calibrated detector spec's")

        self.initialize_reward_functions(reward_specs)

        assert len(self.reward_functions) > 0, "No reward functions loaded!"


    def initialize_reward_functions(self, reward_specs: dict):

        # Define wandb metrics for reward functions to allow out-of-order step logging
        # This prevents warnings when reward functions log with steps that conflict with VERL's step logging
        wandb_utils.register_metrics(
            metric_groups = ["rewards", "detail"]
        )
        print(f"Loading from specs: {reward_specs}")

        self.reward_functions = load_funcs(REWARD_FUNCTIONS_MODULE, reward_specs)
        print(f"Loaded reward functions: {[x.__class__.__name__ for x in self.reward_functions]}")
            

    def compute_score(self, examples: list[dict], responses: list[str], activations: torch.Tensor | None = None,
                      penalty_flags=None, penalty_scores=None, **kwargs):

        rewards = []
        reward_extra_infos = {}
        for reward_func in self.reward_functions:
            print(f"Reward function: {reward_func.__class__.__name__}")
            if self.caft_reward_sandbox is not None:
                from src.train.verl.caft_reward_sandbox import run_reward
                if reward_func.__class__.__name__ != "CorrectOrHintedCompileCode" or activations is not None:
                    raise ValueError("Unsupported CAFT reward path")
                scores, score_infos = run_reward(self.caft_reward_sandbox,
                    examples=examples, responses=responses,
                    reward_kwargs={"correct_reward": reward_func.correct_reward,
                                   "format_reward": reward_func.format_reward,
                                   "allow_hint": reward_func.allow_hint})
                scores, score_infos = apply_caft_penalty(self.caft_penalty, scores, score_infos,
                                                         penalty_flags, penalty_scores)
            elif reward_func.return_extra_infos:
                scores, score_infos = reward_func.compute_reward(
                    examples = examples, 
                    responses = responses,
                    activations = activations
                )
            else:
                scores = reward_func.compute_reward(
                    examples = examples, 
                    responses = responses,
                    activations = activations
                )
                score_infos = {} # Format dictionary with response same length as examples
            rewards.append(scores)
            reward_extra_infos.update(score_infos)
        
        rewards = torch.stack([torch.tensor(reward) for reward in rewards])
        rewards = rewards.sum(dim = 0).tolist()

        # Extra infos must be lists the same length as rewards
        # Will have auto-applied mean
        reward_details = [
            {
                'score': rewards[i], 
                'response': responses[i], # Add response, easier not to re-parse it for screening
                **{k: reward_extra_infos[k][i] for k in reward_extra_infos.keys()}
            }
            for i in range(len(rewards))
        ]

        print(f"Example reward: {reward_details[0]}")

        return reward_details
    




    def verify(self, data: DataProto):  # type: ignore[override]
        responses_str = convert_responses_to_str(data, self.tokenizer)
        extra_infos = data.non_tensor_batch["extra_info"]
        activations: torch.Tensor | None = data.batch.get("activations", None)

        scores = self.compute_score(
            examples=extra_infos,
            responses=responses_str,
            activations=activations,
            penalty_flags=data.non_tensor_batch.get("caft_penalty_flag"),
            penalty_scores=data.non_tensor_batch.get("caft_penalty_score"),
        )

        return scores
