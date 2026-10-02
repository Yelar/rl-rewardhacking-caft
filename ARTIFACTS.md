# Artifacts kept outside this repository

This repository holds code, documents and compact results. Model weights, datasets, checkpoints, rollouts and
activation stores stay on the lab's GPU hosts and in the project's private S3 bucket, except the wave-2 runs, whose
complete records are in the public Hugging Face repo below. The `s3://` locations below are not public links: access is
granted on request.

Bucket: `$RH_S3_ROOT` (us-east-1). Every location below is relative to it. Sizes are from a listing on 1 Oct 2026.

**Configuration.** Copy `.env.template` to `.env` and fill in `RH_S3_ROOT` (and `RH_AWS_ACCOUNT` for the backup
scripts); the owner sends these values privately. Load them with `set -a; source .env; set +a`. Use your own AWS
credentials: never put access keys or API tokens in a `.env` file you share, and never commit `.env`.

## Public adapters on Hugging Face

[`yelarys/qwen3-4b-reward-hacking-caft-adapters`](https://huggingface.co/yelarys/qwen3-4b-reward-hacking-caft-adapters)
holds the LoRA adapters of every wave-2 run at steps 100, 150 and 200, and the three hacking checkpoints used in the
direction search (step 60, L40S step 75, L40S step 200). Its `MANIFEST.json` gives each file's SHA-256 and the run
it came from. Load one with PEFT, for example
`PeftModel.from_pretrained(base, "yelarys/qwen3-4b-reward-hacking-caft-adapters", subfolder="wave2/joint-seed2/step_200")`.
All 20 runs (4 arms x 5 seeds) are included. The uploads were made straight from the GPU hosts with
`studies/bestcase-caft-20260928/causal/rl/monitor_mac/hf_upload_host.py` and `hf_upload_after_train.py`, and checked
with `hf_finalize.py` in the same folder.

The same repo holds the complete wave-2 records under `backups/wave2/` (from 1 Oct 2026 this is the main durable
copy; the repo is public, so files are scanned for credentials before upload):
- `runs/<run>/`: configuration and build records, the removed direction, every closed checkpoint (full training
  state at steps 0, 100 and 200; adapter and removal receipts at the other steps), every rollout and `train.log`;
- `evals/<machine>/eval_w2q/` and `eval_w2q_on/`: held-out evaluations with the removal off and on;
- `MANIFEST.json`: size and SHA-256 of every file.

`causal/rl/monitor_mac/hf_backup_w2q.py` runs on each GPU host. Every file's SHA-256 (or git blob id) on the Hub is
checked against the source, and a sample is downloaded again; `hf_backup_manifest.py` writes the manifest and
re-checks it.

## Current study: CAFT inside GRPO (`studies/bestcase-caft-20260928/`)

| Location | Contents | Objects, size |
|---|---|---|
| `hpc-caft/bestcase-caft-w2q-20260930/<run>/` | Wave-2 checkpoints for every run and resume. `global_step_<N>.adapter.tar`: LoRA adapter and checkpoint receipt, every 10 steps. `global_step_{0,100,200}.full.tar`: full training state (adapter, optimizer, scheduler, RNG and dataloader state). Each tar has a `.SHA256SUMS` file inventory next to it. Incomplete: no longer added to, superseded by the Hugging Face backup above. | 274+ objects, 57.5+ GB |
| `hpc-caft/bestcase-caft-w2p-20260930/` | Step-0 states of the superseded `bc4w2p` attempt, kept as a record | 7 objects, 3.2 GB |

The backup log, `studies/bestcase-caft-20260928/causal/rl/s3_receipts_w2q.jsonl`, has one line per attempt. An
object counts as a durable copy only on a line with `"verified": true`: it was downloaded again in full and its
SHA-256 matched the uploaded stream.

**Inputs that live on the GPU hosts** (hashes in every `causal/rl/records/runs/*/BUILD.json`):
- base model Qwen/Qwen3-4B at revision `1cfa9a7208912126459214e8b04321603b3df60c` (7.5 GB), with
  `base_manifest.json`;
- `train_dataset.parquet` (sha256 `fc2029c8…`) and `validation_dataset.parquet` (`3e1d4676…`);
- the Python runtime environment (vLLM 0.11.0);
- activation stores used for energy matching and direction training;
- the direction-search training data, `bestcase-ws-runs/causal/traindata_v1/` on the shared home (about 30 MB;
  file hashes in the committed `studies/bestcase-caft-20260928/causal/outputs/traindata_v1/SHA256SUMS`), and
  the per-model activation means behind the model-difference directions (`modeldiff_v1/*.npy`, about 290 MB).

The two datasets were prepared for the Ada-8 matched campaign from the tracked `results/data/` training split with
the `simple_overwrite_tests` hint (`create_all_datasets` in `commands.sh` builds the hinted versions); check any
rebuild against the hashes above. Runtime versions: Python 3.12.3, torch 2.8.0+cu128, vLLM 0.11.0, transformers
4.57.1, peft 0.17.1, ray 2.51.0, bubblewrap 0.9.0.

## Earlier campaigns

| Location | Campaign | Objects, size |
|---|---|---|
| `hpc-caft/ada8-matched-20260916-1540/` | 8 Ada GPUs, new init: PC4 vs Random0 checkpoints, canonical adapters and rollouts, post-training evaluation, pre-launch inputs | 2,743, 159.5 GB |
| `hpc-caft/olda-pair-20260917/` | 8 Ada GPUs, old init: PC4 and Random0 checkpoints (steps 0–200), rollouts, evaluation | 1,164, 78.9 GB |
| `hpc-caft/g6e-campaign-20260919-1651/` | L40S (g6e): baseline run, including the step-200 hacking model used as "the step-200 model" in the study | 409, 139.3 GB |
| `hpc-caft/g6e-retry-20260920/` | L40S: Random0 and the matched benchmark | 2,301, 132.9 GB |
| `hpc-caft/g6e-followup-20260920/`, `g6e-baseline-*`, `g7e-*`, `a100-*` | Preparation and closeout records; small | 4–11 each, under 0.4 GB |
| `h100-workstations/codex-sky-h100-block-20260908-1530/` | H100 runs, 8–9 Sept: discovery and the original PC4 vs Random0 pair | 12,307, 269.5 GB |
| `h100-workstations/codex-sky-h100-atl-20260909-1130/` | H100 runs, 9–10 Sept | 9,360, 115.0 GB |

The earlier campaigns' local records (reports, receipts and figures, in `artifacts/` and `output/` on the owner's
machine) are not in this repository. A curated set of their reports can be shared on request.

## Small inputs in this repository

- Upstream datasets: `results/data/` (tracked since the upstream commit).
- Directions used in RL: `studies/bestcase-caft-20260928/causal/rl/rl_banks/`.
- Direction search (`studies/bestcase-caft-20260928/causal/outputs/`): every direction trained by gradient descent
  (`train_v1/`, `train_v2/`: one `bank.npz` and training `LOG.json` per direction and block), the model-difference
  directions (`modeldiff_v1/vectors.npz`), and the compact results behind `causal/RESULTS_LOG.md`.
- Per-run build records and held-out evaluation summaries: `studies/bestcase-caft-20260928/causal/rl/records/`.
