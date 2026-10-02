# Monitoring, reports and backups for CAFT-RL wave 2 (run from a laptop)

These scripts ran on the operator's Mac during wave 2 (run prefix `bc4w2q`). They only read from the GPU hosts
over SSH (aliases `HPC` = gpu-01, `gpu-02`, `gpu-03`, `gpu-04`); none of them starts, stops or signals a training
run. Host paths assume the deployment folder `/scratch/yelarys.yertaiuly/bestcase-step4`; change `B`/`RUNS` at the
top of each script for another user. The S3 scripts read the bucket from `RH_S3_ROOT` and the owning account
from `RH_AWS_ACCOUNT` (set them from the project `.env`; see `ARTIFACTS.md`).

| Script | What it does |
|---|---|
| `w2q_watch2.py` | Monitor. Polls every host every 60 s through `w2q_status2.sh`, groups runs by lineage (a run and its resumed continuations), and queues a host-side resume waiter (`../builder_w2q_ops/resume_when_free.sh`) when a run fails. Exits with a report on any event, or when `--min` runs reach `--step`. The host-side twin is `../monitor_host2/w2q_autoresume.py`. |
| `w2q_status2.sh` | Status probe sent to each host: one `RUN` line per run (unit state, step, exit code, error/guard counts, GPUs, last closed checkpoint), `HOST` resources, `QUEUE` waiters. |
| `w2q_step1.py` | Step-1 check, run on a host: prints `actor/ppo_kl` and `actor/kl_loss` (exactly 0 when the reference applies the same removal) and the first rollout receipt's per-device projection counts (0 for the baseline). |
| `gather_w2q.sh` | Collects per-step training statistics (`rollstats_w2q.py`), run metadata (`runinfo_w2q.sh`) and evaluation summaries (`evalsum_w2q.py`) from all hosts into `w2q_data.json` (compact numbers only, no rollouts). |
| `build_report.py` + `report_template.html` | Builds the training-curves page `caft_rl_curves.html` from `w2q_data.json` (the copy kept in the study is `../caft_rl_curves_snapshot.html`). |
| `build_writeup.py` + `writeup_template.html` | Builds the meeting write-up from `w2q_data.json` into the paths given as arguments (the copy kept in the study is `../WRITEUP_caft_rl_wave2.html`). |
| `build_final.py` + `final_template.html` | Builds the final results report (scoreboard, training curves, held-out hacking with the removal off and on, accuracy without the hint, statistics) into the paths given as arguments; the copy kept in the study is `../RESULTS_caft_rl_wave2.html`. |
| `evalsum_on_w2q.py` | Removal-on evaluation summaries with each cell's removal receipt (gathered into `w2q_data.json` as `evals_on`). |
| `hf_upload_after_train.py` | Runs on a GPU host: once a lineage finishes training, uploads its step-100/150/200 adapters to the public Hugging Face repo (`wave2/<arm>-seed<N>/step_<S>/`) and checks each file's Hub SHA-256. `hf_finalize.py` merges the per-host journals into the repo's `MANIFEST.json`. |
| `hf_backup_w2q.py` | Runs on each GPU host: backs up every run's configuration, closed checkpoints (full state at steps 0, 100 and 200), rollouts, logs and evaluations to `backups/wave2/` in the same public repo. Scans every file for credentials first, uploads with a no-progress watchdog, and checks each file's Hub SHA-256 or blob id. `test_hf_backup_scan.py` tests the scan; `hf_backup_manifest.py` writes and re-checks `backups/wave2/MANIFEST.json`. |
| `s3_backup_w2q_boto.py` | Fallback only: the S3 backup with boto3 instead of the AWS CLI. |
| `s3_backup_w2q.py` | Streams every closed checkpoint to the project's private S3 bucket (host → laptop pipe → S3, no local staging), verifies each object by a full re-download and SHA-256 comparison, and appends receipts to `../s3_receipts_w2q.jsonl`. Waits while the AWS session is expired. |

Refresh both pages:

```bash
bash gather_w2q.sh && python3 build_report.py . && python3 build_writeup.py . writeup.html && python3 build_final.py . results.html
```

Plots use only each lineage's final trajectory: for step *s*, the most recently created run that resumed before *s*.
A run counts as hacked when the 10-step mean of training strict hacking reaches 10%.
