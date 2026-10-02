# CAFT inside GRPO: RL waves 1 and 2

Removal of a direction q at block 19 (counted from 0, so the 20th of 36 layers) during rollouts, the update pass and
the KL reference, with evaluation on held-out problems with the removal off and on. Wave 1 (prefix `bc4w1`, 1 GPU per
run) was stopped at about step 20 and replaced by wave 2. Wave 2 (prefix `bc4w2q`) is the result: 4 arms x 5 seeds x
200 steps, 2 GPUs per run with the 8-GPU batch grouping (`balance_virtual_dp_size: 8`).

Read first: `RESULTS_caft_rl_wave2.html` (final results), `WRITEUP_caft_rl_wave2.html` (meeting brief), `LAUNCH_w1.md` lines 1-107 (arms, settings, pre-registered
outcome), `caft_rl_curves_snapshot.html` (every run's training curves), then `LAUNCH_w2.md` (layout at the top,
first held-out results from about line 320; the rest is an operations log).

## What is here

| Path | What it is | Status |
|---|---|---|
| `plan_w2q.json`, `plan_w2qb.json` | Wave-2 plans: seeds 1-2 and seeds 3-4 (arms, directions, seeds, inputs with hashes) | current |
| `base_config_2gpu.json` | Resolved base training config for 2-GPU runs (sha256 365c8e3b..., pinned in the plans) | current |
| `base_config_1gpu.json` | Same for 1-GPU runs (wave 1); differs in 4 fields listed in `LAUNCH_w2.md` | wave 1 |
| `builder_w2p/` | Builder and launcher used for all of wave 2, despite the name: `build_v2.py` (plan -> per-run config, sets `NCCL_P2P_DISABLE=1`), `launch_seed.sh` (capture arm, then this host's load arms), `launch_load.sh` (load arms against a copied common start), `run_train.sh` (GPU guard, watchdog, release check), `run_entry.py`, `release.sh` | current |
| `builder_w2p_ops/` | `when_free.sh`, `launch_when_free.sh` (start when GPUs are free), `resume_run.sh` (continue from the latest closed checkpoint into a new run directory) | current |
| `builder_w2q_ops/`, `builder_w2q_ops3/` | Slurm-aware free-GPU check and resume waiters (one pair of GPUs, or the first free of several) | current |
| `builder_w2q_ops2/` | Starts random seed 4 after baseline seed 4's evaluations | current |
| `builder_w2/`, `builder_w2_ops/`, `plan_w2.json`, `plan_w2smoke.json` | First wave-2 attempt; its 2-GPU runs hung in NCCL peer-to-peer | superseded |
| `plan_w2p.json`, `plan_w2pb.json` | Wave 2 before the microbatch-profile fix (prefix `bc4w2p`) | superseded |
| `launch_seed.sh`, `plan_w1.json`, `plan_w1smoke.json` | Wave 1 (launcher plus `launchprep/builder/`) | stopped |
| `source_v2_3968e8cf6e89.SHA256SUMS` | Inventory of the staged code all wave-2 runs used. The repository root (`src/`, `verl/`, `infra/`) matches it: run `shasum -a 256 -c studies/bestcase-caft-20260928/causal/rl/source_v2_3968e8cf6e89.SHA256SUMS` from the root | record |
| `source_v2_layout_fix.py` | The patch that turned snapshot `6289a15ba63b` into `3968e8cf6e89` (allowed microbatch profiles for 1, 2, 4 or 8 ranks) | record |
| `rl_banks/` | The directions used: `joint_b19.npz`, `add_b19.npz`, `random_b19_base.npz` (seeds 1-2), `random_b19_base_s34.npz` (seeds 3-4). The `.json` and `draw*.log` files record how each random direction was drawn with `../../step3/tools/energy.py`: the closest of 20,000 draws to the joint direction's energy (1.15-1.21x; none came within 10%) | record |
| `records/runs/<run>/` | Per run and per resume: `BUILD.json`, `config.json`, `spec.json`, `SOURCE_INVENTORY.json`, `q.safetensors` (the direction, hash-pinned in `BUILD.json`) | record |
| `records/eval/<checkpoint>/` | `MANIFEST.json` and `SUMMARY.json` of each held-out evaluation run on the GPU hosts; workstation results are in `../outputs/eval_w2q/` | record |
| `eval_w2q_kit/` | Held-out evaluation: `eval_v2.py` (removal off; refuses CAFT workers), `make_eval_manifest.py`, `eval_guard.sh`, `eval_after_train.sh` (steps 100 and 200 on the run's own GPUs once training ends) | current |
| `eval_w2q_kit2/` | Step-150 sweeper, `eval_step_after_train.sh` | current |
| `ws_eval/` | Workstation coordinators: v1 (step 100) and v2 (steps 100 and 150); both produced results | current |
| `source_v2eval_b71d8e8b425a/` | The scoring code every held-out number used (`scoring.source_tree_sha256` in each manifest; `eval_v2.py` recomputes it) | record |
| `monitor_host2/`, `monitor_mac/` | Host-side auto-resume daemon; laptop-side monitor, page builders, S3 backup and verifier (see `monitor_mac/README.md`) | current |
| `s3_receipts_w2q.jsonl`, `s3_receipts_w2p.jsonl` | One line per S3 backup attempt. Only lines with `"verified": true` count as durable copies | record |
| `LAUNCH_w1.md`, `LAUNCH_w2.md` | Launch records | record |
| `RESULTS_caft_rl_wave2.html` | Final results: all 20 runs, held-out hacking with the removal off and on (open in a browser) | output |
| `WRITEUP_caft_rl_wave2.html`, `caft_rl_curves_snapshot.html` | Meeting write-up and training-curves page (open in a browser) | output |
| `eval_w2q_kit3/` | Held-out evaluation with the run's removal switched on (see its README) | record |

## Reproducing a wave-2 run

Paths in the plans and scripts assume the deployment folder `/scratch/yelarys.yertaiuly/bestcase-step4` on the GPU
hosts; change them for another user.
1. Stage the code: copy the repository root's `src/`, `verl/` and `infra/` to the host as a read-only folder and
   check it against `source_v2_3968e8cf6e89.SHA256SUMS`.
2. Inputs: the base model (Qwen/Qwen3-4B at revision `1cfa9a7208912126459214e8b04321603b3df60c`), the training and
   validation parquet files, and the runtime environment. Their hashes are in every `records/runs/*/BUILD.json`;
   where they live is listed in `ARTIFACTS.md` at the repository root.
3. Build and launch one seed: `builder_w2p/launch_seed.sh` builds each run directory from `plan_w2q.json` (or
   `plan_w2qb.json`) with `build_v2.py`, starts the baseline (capture) arm, waits for its step-0 common start, then
   starts the load arms against it. Every arm of a seed runs the same staged code.
4. Evaluate: `eval_w2q_kit/eval_after_train.sh <lineage>` (steps 100 and 200) and
   `eval_w2q_kit2/eval_step_after_train.sh <lineage> 150`.

## Wave-2 run lineage

Runs stopped by another user's job on a shared GPU were continued from their last closed checkpoint into a new run
directory. Plots and tables use each lineage's final trajectory. State as of 1 Oct 2026, 13:00 UTC.

| Arm, seed | Run (token suffix) | Host, GPUs | From step | Last closed checkpoint | Outcome |
|---|---|---|---|---|---|
| baseline 1 | `(first run)` | gpu-03, 0,1 | 0 | 150 | stopped by another user's job |
|  | `-resume150` | gpu-03, 0,1 | 150 | – | stopped by another user's job |
|  | `-resume150b` | gpu-03, 4,5 | 150 | 200 | finished |
| baseline 2 | `(first run)` | gpu-04, 0,1 | 0 | 10 | stopped by another user's job |
|  | `-resume10` | gpu-04, 0,1 | 10 | – | stopped by another user's job |
|  | `-resume10b` | gpu-04, 0,1 | 10 | 200 | finished |
| baseline 3 | `(first run)` | gpu-01, 0,1 | 0 | 10 | stopped by another user's job |
|  | `-resume10` | gpu-01, 0,1 | 10 | – | stopped by another user's job |
|  | `-resume10b` | gpu-01, 0,1 | 10 | 20 | stopped by another user's job |
|  | `-resume10b-resume20` | gpu-01, 1,2 | 20 | 200 | finished |
| baseline 4 | `(first run)` | gpu-02, 2,3 | 0 | 200 | finished |
| joint 1 | `(first run)` | gpu-03, 2,3 | 0 | 190 | stopped by another user's job |
|  | `-resume190` | gpu-03, 2,3 | 190 | 200 | finished |
| joint 2 | `(first run)` | gpu-04, 2,3 | 0 | 200 | finished |
| joint 3 | `(first run)` | gpu-01, 2,3 | 0 | 10 | stopped by another user's job |
|  | `-resume10` | gpu-01, 2,3 | 10 | – | stopped by another user's job |
|  | `-resume10b` | gpu-01, 2,3 | 10 | 20 | stopped by another user's job |
|  | `-resume10b-resume20` | gpu-01, 0,3 | 20 | – | stopped by another user's job |
|  | `-resume10b-resume20b` | gpu-01, 0,3 | 20 | 50 | stopped by another user's job |
|  | `-resume10b-resume20b-resume50` | gpu-01, 0,3 | 50 | 110 | stopped by another user's job |
|  | `-resume10b-resume20b-resume50-resume110` | gpu-01, 6,7 | 110 | 150 | running (step 155) |
| joint 4 | `(first run)` | gpu-02, 4,5 | 0 | 200 | finished |
| adding 1 | `(first run)` | gpu-03, 4,5 | 0 | 200 | finished |
| adding 2 | `(first run)` | gpu-04, 4,5 | 0 | 200 | finished |
| adding 3 | `(first run)` | gpu-01, 4,5 | 0 | 200 | finished |
| adding 4 | `(first run)` | gpu-02, 6,7 | 0 | 200 | finished |
| random 1 | `(first run)` | gpu-03, 6,7 | 0 | 200 | finished |
| random 2 | `(first run)` | gpu-04, 6,7 | 0 | 200 | finished |
| random 3 | `(first run)` | gpu-01, 6,7 | 0 | 200 | finished |
| random 4 | `(first run)` | gpu-02, 2,3 | 0 | 80 | running (step 79) |