# Held-out evaluation with the removal on (eval_w2q_kit3)

The standard held-out evaluation (`eval_w2q_kit/eval_v2.py`) loads each checkpoint's adapter in plain vLLM, so the
direction is not removed. This kit evaluates the same checkpoints with the run's own removal switched on during
generation, as in training rollouts. Everything else is identical, so each removal-on cell is the twin of a
removal-off cell: datasets, prompts, sampling seeds and settings, engine profile and scoring.

| File | Role |
|---|---|
| `eval_v2_on.py` | `eval_v2.py` with only generation changed. The engine is built with the training rollout's CAFT worker and the run's spec (`verl.utils.caft_vllm_worker.CAFTWorker`). `caft_vllm.install` and the prefix-cache policy are applied as in `vllm_rollout_spmd.py`, and generation runs inside `caft_vllm.rollout_scope`. Each cell's `RAW_COMPLETE.json` keeps the removal receipt, which records the rows projected at each layer (it must equal the predictor tokens), and the prefix-cache receipt. |
| `eval_guard_on.sh` | `eval_guard.sh` running `eval_v2_on.py`, with the training snapshot's `verl` on `PYTHONPATH` (as in `run_train.sh`). |
| `stage_on_inputs.py` | Copies a checkpoint's adapter, and the run's spec and q file, from its training host. The checkpoint is the latest closed one among the run and its resumes. Every file is checked against the source SHA-256 and, when given, against the removal-off evaluation's recorded adapter hashes. |
| `stage_after_train.py` | Waits for runs still training to finish, then stages them. |
| `make_manifest_on.py` | Writes the removal-on manifest for a staged checkpoint (output in `eval_w2q_on/`). |
| `make_manifest_off.py` | Writes a standard removal-off manifest for a staged copy. Used to re-run removal-off cells whose training host has no free GPU. |
| `queue_on.sh` | Runs one worker per GPU. Each worker claims cells from a queue, waits for a free GPU and enough host RAM, dry-runs each cell, then runs it under the guard. |

The baseline arm is not evaluated with the removal on: its spec removes nothing (a structural zero), so both
evaluations would be the same.

First run (1 Oct 2026, gpu-03, GPUs 2–7):
- removal on for joint, adding and random, seeds 1–5, at steps 100, 150 and 200;
- removal-off re-runs of four cells the workstation evaluator had failed on: joint s3 step 150, random s1 step 150,
  baseline s3 step 100 and random s4 step 150.

Test cell `bc4w2q-jointb19-s2_step200`, fixed setting:
- all 897,089 predictor rows were projected at block 19;
- 4,261 CUDA-graph replays, with no capture inside the removal scope;
- the prefix cache was salted per scope and never reused the last prompt token.
