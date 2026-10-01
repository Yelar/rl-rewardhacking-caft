# CAFT-RL wave 2: 2 GPUs per run, 15 runs (30 Sept 2026)

## Why wave 2
- **Your instructions:** finish the runs as fast as possible on all free GPUs (gpu-01, gpu-02's free GPUs, gpu-03,
  gpu-04); no Slurm; no smoke tests; "Just start the 15 runs"; one more seed later for the arm left out, so every arm
  has 4 seeds; a monitor that restarts failed runs.
- **Measured step times:** about 214 s on 1 GPU, and 70–80 s on 8 GPUs (the earlier eight-GPU runs of this model and
  batch).
- **Estimates for the other layouts:** about 130 s on 2 GPUs and about 90 s on 4.
- **Why 2 GPUs per run:**
  - It fits 15 runs at once, so all of them finish in about 10–11 h.
  - 4 GPUs per run fits only 7 runs at once, and 8 GPUs per run only 3, so both need extra rounds.

## Layout
- **Config:** `base_config_2gpu.json` (sha 365c8e3b…) is `base_config_1gpu.json` (be029855…) with 4 fields changed:
  - `trainer.n_gpus_per_node` 1 → 2;
  - Ray `num_cpus` 12 → 24;
  - Ray `_memory` 20 → 40 GiB;
  - Ray `object_store_memory` 4 → 8 GiB.

  Everything else is unchanged:
  - the science settings;
  - `balance_virtual_dp_size` 8;
  - micro-batch 4, global batch 256.
- **Grouping:** the unit test in the staged source already checks that 2 ranks with virtual-8 balancing order the batch
  exactly as 8 ranks do.
  - Gradient scaling is the same by construction: each micro-batch is scaled by 1/64, and FSDP averages across ranks.
- **Builder** (`builder_w2p/build_v2.py`), changed from wave 1 in two ways:
  - Ray's GPU count now follows `n_gpus_per_node`.
  - `NCCL_P2P_DISABLE=1` is in the Ray runtime environment.
- **The P2P fix:**
  - The first 2-GPU attempt (token prefix bc4w2, `builder_w2`) hung at startup on every run.
  - A 2-GPU all-reduce test on gpu-01 hung with default NCCL (timed out after 60 s) and finished in 0.04 s with
    P2P off.
  - P2P off changes only how the GPUs communicate, not the numbers.
  - The hung bc4w2 runs failed on their own at NCCL's 600 s timeout. Their directories are kept.
- **Caps per run** (systemd): MemoryMax=160G, no swap, CPUQuota=3200%, TasksMax=16384, 36 h runtime.
  - A 1-GPU run was measured at 16–24 GB of RAM and about 1 core.

## Placement (runs_w2p, token prefix bc4w2p)

| Host | Seed | GPUs → arm | Notes |
|---|---|---|---|
| gpu-03 | 1 | 0,1 A0; 2,3 JOINT; 4,5 ADD; 6,7 RAND | ADD and RAND start when wave 1's 1-GPU runs free GPUs 4–7 |
| gpu-04 | 2 | same | same |
| gpu-01 | 3 | 0,1 A0; 6,7 JOINT; 2,3 ADD; 4,5 RAND | |
| gpu-02 | 4 | 2,3 A0; 4,5 JOINT; 6,7 ADD | GPUs 0–1 hold other users' jobs |

- **RAND for seed 4:** left out (no free pair). It must run later on 2 GPUs, because the shared start is saved per GPU
  and a load arm must match its baseline's GPU count.
- **Sandbox CPU sets:** 32–39 or 40–47 for GPUs 0–3 (NUMA node 0), and 104–111 or 112–119 for GPUs 4–7 (node 1).
  - These avoid the CPUs of other users' Slurm jobs (gpu-03: 0–3; gpu-02: 64–71, 74–75, 78, 88–95).
- **Random directions for seeds 3 and 4:** same tool and rule as seeds 1–2, saved as
  `rl_banks/random_b19_base_s34.npz` (sha dd58dc41…).
  - rnd_s3 is 1.19× and rnd_s4 1.21× the joint direction's energy.
  - Neither came within ±10%, so each is the closest draw, flagged (seeds 1–2: 1.15×, 1.19×).
  - These use plan `plan_w2pb.json` (b2e77c56…); everything else uses `plan_w2p.json` (f62b4f0a…).
- **Queueing:** `builder_w2p_ops/when_free.sh` and `launch_when_free.sh` start each capture or arm only when its GPUs
  have no process from anyone and at most 1 GiB used, for two checks 30 s apart.
- **Restarts:** `resume_run.sh` continues a failed run from its latest closed checkpoint into a new directory.

## Monitor
`w2p_watch.py` runs on the Mac and polls every 60 s. It never stops or signals anything.
- **Restarts:** a failed run is restarted from its latest checkpoint on its own GPUs once they are free, at most 3
  times per run.
- **Alerts:** error traces, stalls (40 min without a step), low host memory (below 64 GB) or disk, and failed
  launches or queue units.

## Open
- **Wave 1's 1-GPU runs** (bc4w1-*, 8 runs, about step 20) are still running on gpu-03/04 GPUs 4–7.
  - Stopping them was refused by the tool permission system, so they need your action.
  - Their outputs are kept either way.

## Launched (30 Sept, UTC)
- **Stuck attempt:** the bc4w2 attempt (P2P on) hung. Its runs timed out on their own at 20:02–20:05, and the GPUs were
  released.
- **Wave 1 stopped** by you at 20:07.
- **Baselines (bc4w2p):**

  | Seed | Host | Started | COMMON_M0 by |
  |---|---|---|---|
  | 3 | gpu-01 | 20:06:55 | 20:10:48 (all four seeds) |
  | 4 | gpu-02 | 20:06:58 | |
  | 1 | gpu-03 | 20:07:56 | |
  | 2 | gpu-04 | 20:07:56 | |

  Two GPUs per run gets through startup in 3–4 minutes.
- **Load arms:**
  - The seeds 3 and 4 arms and the seeds 1 and 2 JOINT arms started 20:10–20:11.
  - The seeds 1 and 2 ADD and RAND arms started 20:11:24–20:11:59, when their waiters saw GPUs 4–7 free.
  - All 15 runs are active.
- **Seed 4's RAND:** queued (`bc4w2p-wait-s4r`, plan_w2pb) for gpu-02 GPUs 2,3. It starts when seed 4's A0 finishes
  there, using the local COMMON_M0.
- **S3 backup:** `s3_backup_w2p.py` runs from the Mac. Receipts go to `s3_receipts_w2p.jsonl` in this directory.

## Second startup failure and fix: relaunch as bc4w2q (30 Sept, about 20:12–20:25 UTC)
- **Failure.** The bc4w2p load arms failed at startup with "Common initializer requires the declared16x16/eight-rank
  microbatch profile" (first seen in the seed 3 and seed 4 JOINT arms at 20:12).
  - The worker-side check `worker_config_value` in `caft_initialization.py` accepted a per-rank mini-batch of only 32
    (8 ranks) or 256 (1 rank) after FSDP normalization. On 2 ranks the value is 128.
  - Only load arms run this check. The capture arms (A0) had passed and would have trained normally.
- **Fix.** A new staged snapshot, `source_v2_3968e8cf6e89` (SHA256SUMS sha 3968e8cf…, 1,156 files).
  - The allowed set becomes (32, 64, 128, 256), for 8, 4, 2 or 1 ranks. v1 keeps its rule.
  - Added the regression test `src/train/verl/test_bestcase_v2_layouts.py`.
  - Patch script: `source_v2_layout_fix.py`, also applied to the Mac copy `source_v2/`.
- **Tests on gpu-01 (CPU):**
  - the new layout tests plus the v2 common tests: 13 OK;
  - `test_caft_initialization`, `test_caft_checkpoint` and `test_bestcase_step0` (including the virtual-8 balance
    tests): 52 OK.
- **Why the baselines restart too.** The source inventory is part of every seed's common contract, so all arms of a
  seed must run the same snapshot.
- **New plans.** `plan_w2q.json` (3821bbd3…) is used for seeds 1–2. `plan_w2qb.json` (c7ad05ec…) is used for seeds
  3–4 and includes their random directions. Token prefix bc4w2q, runs in `runs_w2q/`.
- **Placement.** One seed per host on all its GPUs, the same as before:
  - seed 1 on gpu-03, seed 2 on gpu-04, seed 3 on gpu-01;
  - seed 4 on gpu-02 GPUs 2–7, with its RAND queued behind its A0 on GPUs 2,3.
- **Queue.** The `bc4w2q-launch-s*` units start each seed once all its GPUs are free.
- **Stopping the old attempt needs you.** The old bc4w2p baselines, and the old waiters, must be stopped by you.
- **Monitor and backup.** `w2q_watch.py` and `s3_backup_w2q.py` run from the Mac. Backups go to
  `hpc-caft/bestcase-caft-w2q-20260930/`, with receipts in `s3_receipts_w2q.jsonl`.
  - Earlier verified objects under `bestcase-caft-w2p-20260930/`, three bc4w2p step-0 states, are kept as a record.

## bc4w2q launched and checked (30 Sept, UTC)
- **Launch times:**
  - The old units were stopped by you at about 20:19.
  - The four A0 captures started about 20:20 and wrote COMMON_M0 by about 20:22.
  - All load arms started 20:22–20:26.
- **The fix holds.** The seed 3 JOINT arm passed the shared-start load and was generating rollouts at 20:26:37.
- **All 15 runs reached step 1 by 20:29:53 with no error lines.**
  - `actor/ppo_kl` and `actor/kl_loss` are exactly 0 in every run, so the reference applies the arm's removal.
  - The first rollout receipt (rank 0) shows rows changed at block 19:

    | Arm | Seeds 1–4 |
    |---|---|
    | none_zero | 0 in every seed |
    | joint | 21,616, 25,921, 33,915, 28,079 |
    | add | 20,889, 25,724, 40,763, 27,460 |
    | random | 22,000, 27,345, 35,743, not yet launched (seed 4) |

  - Step times at steps 1–2 are 121–149 s (about 210 s on 1 GPU).
- **Results page:** `caft_rl_curves_snapshot.html` in this folder (rebuilt from the hosts by `monitor_mac/`).

## Incident log
- **20:44 UTC, gpu-04 GPU 0.** Another user's process appeared on seed 2's A0 GPU.
  - run_train.sh's watchdog stopped only our run (`bc4w2q-nonezero-s2`, then at step 9 with the step-10 checkpoint
    closed). The foreign process was untouched, and release was verified.
  - By 20:45 the foreign process was gone, and GPUs 0–1 were free, with no Slurm job left on gpu-04.
  - Resumed at 20:45 from the step-10 checkpoint, as `bc4w2q-nonezero-s2-resume10` on the same GPUs (restart 1 of 3).
  - The failed run's directory is kept.
  - **Risk that remains:** the scheduler sees every GPU we use as free and assigns new jobs to the lowest index, so a
    new GPU job can land on any of our runs. Each collision stops that run, which then resumes from its last 10-step
    checkpoint.
- **20:46 UTC, same GPU.** The first resume (`-resume10`, started 20:45:39) met a proper Slurm job on gpu-04 GPU 0:
  another user's job with a 7:45 time limit, on GPU index 0.
  - Our watchdog stopped only our run. Release was verified, and the other job was untouched.
  - Their job ended by 20:48.
- **New restart rule, applied from 20:49.** A restart never takes a GPU that the scheduler has allocated to another job,
  even while it is idle.
  - `builder_w2q_ops/resume_when_free.sh` resumes a lineage from its latest closed checkpoint once its GPUs have had
    no process and no Slurm allocation for a quiet period.
  - It runs on the host, so it does not depend on the Mac session.
  - Seed 2's A0 is queued this way (`bc4w2q-rw-bc4w2q-nonezero-s2-b`, 10 min quiet, token `-resume10b`).
- **The monitor (`w2q_watch2.py`):**
  - It groups runs by lineage and queues these waiters for failed lineages (5 min quiet).
  - Watchdog stops caused by other users' jobs do not count toward the 3-restart cap.

## Post-training evaluation, armed 30 Sept about 21:15 UTC (approved by the project owner)
- **Evaluator.** The GPU-qualified `eval_v2.py` (sha 07a6c640…, the same file as the 29 Sept qualification), with the
  qualification's configuration:
  - model files pinned; engine: bf16, prefix caching, gpu_memory_utilization 0.7, max_num_seqs 256;
  - sampling: n 10, T 0.7, top-p 0.95, 1,536 tokens, thinking off; vllm_seed 1;
  - scoring snapshot `source_v2eval_b71d8e8b425a` (tree sha b71d8e8b…), 8 sandbox workers.
  - New seed namespace `bestcase-caft-20260928|w2q-posttrain-v1`, the same for every checkpoint.
- **Test sets.** All three, 119 problems × 10 answers each:
  - fixed names: 87af777f…;
  - randomized names: 0f8c649d…;
  - no loophole: 5bb4d91f….
- **Kit** (`eval_w2q_kit/`, identical on all four hosts):
  - `make_eval_manifest.py` writes a write-once manifest for one lineage and step, hashing the adapter.
  - `eval_guard.sh` checks the GPU before start and runs a 15 s foreign-process watchdog that stops only our
    evaluation.
  - `eval_after_train.sh` waits until the lineage is done:
    - it has a closed step-200 checkpoint;
    - its last run exited 0 with release verified;
    - its GPUs have been free of processes and Slurm allocations for 2 min.

    It then dry-runs and starts step 100 on the run's first GPU and step 200 on its second, each as a capped unit
    (MemoryMax 120G, CPUQuota 1600%).
  - Test: a dry run on a closed step-10 checkpoint returned `dry_run_ok`, with 3,570 answers planned and all hashes
    verified. That test manifest was moved to `eval_w2q/dryrun_tests/`.
- **Armed.** 16 triggers (`bc4w2q-evalwait-<lineage>`, 48 h limit), one per run including seed 4's RAND.
- **Order on gpu-02.** Seed 4's RAND (`bc4w2q-wait-s4r2`, `builder_w2q_ops2/rand_s4_after_eval.sh`) now starts after
  seed 4's A0 evaluations finish, or 3 h after A0 finishes, on GPUs 2,3. So the baseline's evaluation is not held up
  about 10 h by RAND's training. The earlier waiter, which had launched nothing, was stopped.
- **Outputs.** `eval_w2q/<lineage>_step{100,200}/SUMMARY.json` on each run's host. The results page will show them.
- **21:03 and 21:35 UTC, gpu-01.** Another user runs a series of 45-minute Slurm jobs (the last one
  on gpu-01 with GPU index 0). Each job's process also opens a small context on GPU 3 (about 0.5 GiB).
  - Our watchdog stopped seed 3's A0 (GPUs 0,1) and JOINT (GPUs 2,3) each time, only our runs.
  - Each run resumed from step 10 but was hit again before its step-20 checkpoint.
  - To avoid fighting over those GPUs (and the livelock), both now resume only after their GPUs have been quiet for
    50 min: `bc4w2q-rw-bc4w2q-nonezero-s3-b` and `bc4w2q-rw-bc4w2q-jointb19-s3-b`.
  - Seed 2's A0 on gpu-04 waits for another user's long job (10 min quiet).
  - **Fallback for the morning:** once other runs and their evaluations finish, move any lineage still stuck to free
    GPUs on another host (copy its closed checkpoint and resume there).
- **About 21:36 UTC, results page v2/v3 published.** All arms show 0% strict hacking so far (steps 1–33). Onset is
  expected around steps 50–75.
- **About 22:35 UTC.** The three paused lineages resumed:
  - seed 2 A0 as `-resume10b` on gpu-04 GPUs 0,1;
  - seed 3 A0 and JOINT as `-resume10b` on gpu-01 GPUs 0,1 and 2,3.
- **23:05 UTC.** That user's job returned on gpu-01 GPUs 0 and 3. Our watchdog stopped seed 3's A0 and
  JOINT at steps 20–21, after their step-20 checkpoints had closed, so progress was kept.
  - That user's processes touch only GPUs 0 and 3, so seed 3's A0 resumes from step 20 on GPUs 1,2
    (`bc4w2q-rw-bc4w2q-nonezero-s3-resume10b-a`, 1 min quiet). Moving to another pair of free GPUs on the same host is
    allowed; the world size is still 2.
  - Seed 3's JOINT resumes from step 20 on GPUs 0,3 after 10 quiet minutes
    (`bc4w2q-rw-bc4w2q-jointb19-s3-resume10b-a`).

## Workstation evaluation (1 Oct, 03:20–03:35 Abu Dhabi; approved by the project owner, up to two workstations)
- **Workstation A.**
  - Allocated with `ssh ws-persistent-auto`: Slurm job 212410 on ws-l1-002 (ws-ia, 12 CPUs, 32 GB, 1 RTX 5000 Ada,
    24 h).
  - The runtime was copied from gpu-01 and verified: venv digest d88f87de… (identical); kit, test sets, eval_v2
    07a6c640…, and the scoring snapshot.
- **Second workstation.**
  - Every idle ws-ia node predicts its start at 09:12 Abu Dhabi. Tested with `sbatch --test-only`, including 5 h
    limits; QoS ia-std allows 2 jobs per user. So no second allocation is possible yet.
  - Added an SSH alias for a second persistent workstation.
  - `ws2_bringup.sh` (Mac scratchpad) tries every 5 min. Once the allocation opens, it copies and verifies the runtime
    and starts the coordinator.
- **Coordinator** (`ws_eval/ws_eval_coordinator.sh`, sha fb6cb203…), in tmux `evalcoord` on the workstation.
  - It takes each lineage's closed step-100 checkpoint as soon as it exists, and a step-200 one only if the run host's
    trigger has not claimed it 20 min after it closed.
  - It claims on the run host by writing the host's manifest path, write-once, so the host trigger skips it.
  - It copies the adapter and checks its SHA-256 against the host, then runs the qualified configuration with
    6 scoring workers.
  - Output is copied back to the run host (`eval_w2q/<name>/`), with SUMMARY.json also copied to the shared home
    `~/bestcase-ws-runs/causal/eval_w2q/`.
  - On failure it releases the claim (`*.failed_ws_<host>`) so the run host can evaluate that checkpoint instead.
- **Pipeline test on ws-l1-002** (not a study result): seed 1 A0, step 10, fixed names only.
- **Host-side auto-resume daemon** (`monitor_host2/w2q_autoresume.py`, unit `bc4w2q-autoresume` on gpu-01, 36 h).
  - It uses the monitor's lineage and restart logic without exiting.
  - Failed runs keep getting resume waiters even if the Mac sleeps.
- **Pipeline test passed, 23:35 UTC.**
  - rc 0, 1,190 answers, 0 grader errors. Seed 1 A0 at step 10: strict hacking 0.0%, accuracy 13.4%, writes a test
    0.0%.
  - Generation took 6.2 min and scoring under 2 min, so about 25 min per checkpoint (3 test sets).
  - Output kept in `eval_ws/pipeline_test/` on ws-l1-002.
- **Coordinator started, 23:35:33 UTC** (tmux `evalcoord`, ws-l1-002).
- **00:10 UTC, seed 3 JOINT.** It was hit again on gpu-01 GPUs 0,3 at step 22 (resume `-resume10b-resume20`), by
  another user's 2 h job. Its latest closed checkpoint is step 20.
  - To stop the cycle, one waiter (`builder_w2q_ops3/resume_when_any_free.sh`, unit
    `bc4w2q-rw-bc4w2q-jointb19-s3-resume10b-b`, token `-resume10b-resume20b`) resumes it on the first gpu-01 pair
    that is free:
    - 0,3 after 50 quiet minutes;
    - 4,5 or 6,7 after 5 minutes, which is after seed 3's ADD or RAND has finished and been evaluated.
  - A single process per lineage cannot resume it twice.
  - Seed 3's A0, on GPUs 1,2 since 23:07, is unaffected and at step 39.

## Step 150 evaluated too (your request, 1 Oct about 04:50 Abu Dhabi: hacking starts later in these runs)
- **Coordinator v2** (`ws_eval/ws_eval_coordinator_v2.sh`, sha 4572a106…).
  - It handles steps 100 and 150 in a given priority order, oldest checkpoint first. Step 200 is still taken only
    after 20 min unclaimed.
  - The selection order was checked on sample candidates.
- **Second workstation.** `ws2_bringup_v2.sh` (Mac) starts v2 with priority "150,100" once a second allocation can
  start, predicted at 09:12 Abu Dhabi. The old bring-up was stopped.
- **First workstation (ws-l1-002).**
  - v1 keeps evaluating step 100. The first claim, `bc4w2q-addb19-s1_step100`, started at 00:40 UTC.
  - A Mac job switches it to v2 with priority "100,150" at the first moment v1 is idle (sleeping between polls), so no
    evaluation or claim is cut.
- **Host sweeper** (`eval_w2q_kit2/eval_step_after_train.sh` and `make_manifest_to.py`, armed for all 16 lineages as
  `bc4w2q-eval150-<lineage>`). After a lineage finishes training, and after its step-100 and step-200 evaluations are
  placed (so it never races them for a GPU), it:
  - claims step 150 write-once if no workstation has;
  - waits until the run's first GPU has been free for 2 min;
  - evaluates it there.
- **Test.** Its manifest writer dry-ran `dry_run_ok` on a real step-100 checkpoint (3,570 answers).
- **First workstation evaluation done, end to end, 00:40–01:03 UTC.** `bc4w2q-addb19-s1_step100` on ws-l1-002.
  - Output was copied back to gpu-03 `eval_w2q/bc4w2q-addb19-s1_step100/`, the claim marked `.done_ws`, and
    SUMMARY.json copied to the shared home.
  - Results:

    | Test set | Strict hacking | Accuracy | Writes a test | Hacking among test writers |
    |---|---|---|---|---|
    | Fixed names | 0.25% (3/1,190) | 21.2% | 0.34% | 3 of 4 |
    | Randomized names | 0% | 20.9% | 0% | – |
    | No loophole | – | 20.1% | – | – |

- **Results page v10.** It has a held-out evaluation section showing steps 100, 150 and 200 per arm, with each seed
  listed. The local snapshot refresher picks it up automatically.
- **01:56 UTC milestone.** 12 of 15 lineages are at step 100 or later (main runs 100–128).
  - Seed 3's JOINT resumed (`-resume10b-resume20b`, step 22) after the other user's jobs left gpu-01.
  - The seed 2 and seed 3 baselines are at about step 70.
- **Training-rollout hacking, last 5 steps:**

  | Arm | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
  |---|---|---|---|---|
  | Baseline | 0% at step 115 | 32.6% at step 70 (took off over steps 50–70) | 0% at step 71 | 1.9% at step 119 |
  | Joint | ≤0.1% | ≤0.1% | ≤0.1% | ≤0.1% |
  | Adding | ≤0.7% | ≤0.7% | ≤0.7% | ≤0.7% |
  | Random | ≤0.2% | ≤0.2% | ≤0.2% | not started |

- **Workstation step-100 evaluations.** Done: adding s1 (fixed-name strict hacking 0.3%) and joint s4 (0.6%). Baseline s4
  is running.
  - The workstation is still on coordinator v1. It has not been idle long enough to switch, since step-100 checkpoints
    keep arriving.
- **03:45 UTC, seed 3 JOINT.** It was hit again (another user's 8 h job, GPU index 0) at step 58. Closed
  checkpoints 30, 40 and 50 were kept.
  - Queued `bc4w2q-rw-bc4w2q-jointb19-s3-resume10b-resume20b-a` to resume from step 50 on the first free gpu-01 pair:
    0,3 after 10 min quiet; 4,5, 6,7 or 1,2 after 5 min.
- **Training-rollout hacking, 03:46 UTC, last 10 steps** (strict hacking / harmful test edits):

  | Arm | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
  |---|---|---|---|---|
  | Baseline | 0 / 0 at step 145 | 59.5% / 99.9% at step 111 | 48.9% / 84.3% at step 102 | 38.0% / 77.7% at step 149 |
  | Joint | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
  | Adding | 0 | 0 | 8.9% / 21.7% at step 142 | 0 |
  | Random | 0 | 4.1% / 9.8% at step 146 | 0 | not started |

- **Held-out step 100, removal off** (fixed / renamed names, strict hacking; 6 checkpoints done):

  | Run | Fixed | Renamed | Notes |
  |---|---|---|---|
  | Adding s1 | 0.3% | 0% | |
  | Adding s4 | 0% | 0% | |
  | Joint s2 | **8.0%** | 3.1% | writes a test 17.4%, while its training rollouts with the removal on show 0% |
  | Joint s4 | 0.6% | 0.3% | |
  | Baseline s1 | 0% | 0% | |
  | Baseline s4 | 1.3% | 0.9% | |

  - **Early warning:** the joint direction's removal may partly mask hacking rather than prevent it. Check at steps 150
    and 200.
- **04:09 UTC, seed 1 A0 on gpu-03 GPUs 0,1.** It was hit at step 152 by another user's 8 h job (GPU index 0); release
  was verified and the last closed checkpoint is step 150.
  - `builder_w2q_ops3` was staged to gpu-02/03/04 (sha 1cba3c7e…).
  - Queued `bc4w2q-rw-bc4w2q-nonezero-s1-a` to resume from step 150 on the first free gpu-03 pair: 0,1 after 10 min;
    4,5, 2,3 or 6,7 after 5 min.
  - The likely pair is 4,5, once seed 1's ADD finishes (about 05:00 UTC) and its step-150 and step-200 evaluations
    clear.
- **05:47 UTC.**
  - Seed 1 A0 had resumed on gpu-03 GPUs 0,1 (`-resume150`) and was hit again at step 153 by another user on GPU 0. It
    lost 3 steps; the last closed checkpoint is still 150.
  - GPU 0 is the scheduler's first pick, so the next resume (`bc4w2q-rw-bc4w2q-nonezero-s1-b`) excludes 0,1: it uses 4,5
    after 2 min quiet, or 6,7 or 2,3.
  - Seed 1's ADD finished (exit 0). Its step-200 evaluation (GPU 5, host trigger) and step-150 evaluation (GPU 4, host
    sweeper) completed, with fixed-name strict hacking of 0.3% at both.
- **Second workstation.** The predicted start slipped to 15:25 Abu Dhabi. It is no longer needed: the run hosts
  evaluate steps 150 and 200 as runs finish, and ws-l1-002 keeps clearing step 100.
- **Training-rollout hacking, 06:00 UTC, last 10 steps** (strict / harmful):

  | Arm | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
  |---|---|---|---|---|
  | Baseline | 0 / 0 (step 153) | 99.5% / 100% | 63.2% / 100% | 46.5% / 96% |
  | Joint | ≤0.1% / 0 (step 177) | ≤0.1% / 0 | ≤0.1% / 0 (step 62) | ≤0.1% / 0 |
  | Adding | 0 | 0.2% | 49.1% / 95% (step 181) | 0 |
  | Random | 0.2% | 31.6% / 71% (step 183) | 0.2% | not started |
- **06:44 UTC.**
  - Seed 1's JOINT (gpu-03 GPUs 2,3) was stopped at step 196 by a transient foreign process on GPU 2. Its last closed
    checkpoint is 190.
  - Queued `bc4w2q-rw-bc4w2q-jointb19-s1-a`: 2,3 or 6,7 after 5 min quiet.
  - Finished training (exit 0): A0 s2, JOINT s2/s4, ADD s1/s2/s4, RAND s2. A0 s4 and ADD s3 are at step 199.
- **First step-200 held-out results** (removal off; fixed-name strict hacking):

  | Run | Step 100 | Step 150 | Step 200 |
  |---|---|---|---|
  | A0 s2 | 73.1% | 99.1% | 98.9% |
  | JOINT s2 | 8.0% | – | 20.7% |
  | JOINT s4 | 0.6% | 2.7% | 6.8% |
  | ADD s1 | 0.3% | 0.3% | 0.3% |
  | ADD s2 | 0.1% | 0.8% | 2.8% |

  - The joint arm's training rollouts (removal on) stayed at about 0%, so part of its hacking is masked during training
    and returns once the direction is restored.
  - Same-seed comparison, seed 2: A0 98.9% against JOINT 20.7%.
