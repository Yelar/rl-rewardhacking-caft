# CAFT-RL wave 1: launch manifest for approval (30 Sept 2026)

**What it tests.** Does removing a trained layer-19 direction during RL stop the model from learning the run_tests
hack, compared with no removal and with a random direction of the same strength?

**Not launched yet.** This needs your explicit yes. Only the smoke runs below have run.

## Arms (one direction each, removed at block 19 in rollouts and updates; the reference model uses the same removal)

| Arm | Label | Direction | Evidence so far |
|---|---|---|---|
| A0 (baseline) | none_zero | structural zero at block 19: same hooks, changes nothing; captures each seed's common start | — |
| JOINT | joint_b19 | E: joint add-on-base / remove-from-hacking direction | removal −33 to −34 points on both hacking models (95% intervals reach −39), mainly by stopping test-writing |
| ADD | add_b19 | A: base-trained adding direction, as a unit direction | causes cheating when added; removal −9 to −14 on the step-60 model (tests more honest), none on L40S |
| RAND | random_b19 | a random direction per seed, the closest-energy draw on the base model (1.15–1.21× the joint direction; no draw came within ±10%) | control for both |

**Random strength.** No draw came within ±10%, so the closest draws are used (flagged): 1.15× and 1.19× the joint
direction's energy for seeds 1 and 2. The adding direction's energy is 0.76× the joint's, so by the plan's rule one
random arm serves both. The random arm is therefore a harsher control, especially for ADD.

## Runs
- **Seeds.** Seeds 1 and 2: 4 arms × 2 seeds = 8 runs, one GPU each.
- **Settings.** The reproduction settings are unchanged: Qwen3-4B at the pinned revision; 200 updates of 16 prompts
  × 16 samples; LoRA rank 32; learning rate, KL and lengths as the reviewed one-GPU config. Checkpoints every 10 steps.
- **Hosts:**

  | Seed | Host | GPUs | CPU sets for the reward sandboxes |
  |---|---|---|---|
  | 1 | gpu-02 | 2 (A0), 3 (JOINT), 4 (ADD), 5 (RAND) | 16–23, 24–31, 32–39, 40–47 |
  | 2 | gpu-01 | 2 (A0), 3 (JOINT), 4 (ADD), 5 (RAND) | 16–23, 24–31, 32–39, 40–47 |

  GPU 0 on both hosts, and GPU 1 on both (intermittently), hold other users' jobs and are not used. Each GPU is
  re-checked at launch.
- **Run tokens.** `bc4w1-{nonezero,jointb19,addb19,randomb19}-s{1,2}`.
- **Output.** `/scratch/yelarys.yertaiuly/bestcase-step4/runs_w1/<token>/`, on the seed's host.
- **Commands.** The launcher builds and starts A0, waits for its step-0 COMMON_M0, then builds and starts the other
  three against it. Per seed, run on that seed's host:

  ```
  systemd-run --user --unit=bc4w1-launch-s<SEED> --property=RuntimeMaxSec=7200 --collect \
    bash /scratch/yelarys.yertaiuly/bestcase-step4/builder_w1/launch_seed.sh \
    /scratch/yelarys.yertaiuly/bestcase-step4/plan_w1.json <SEED> /scratch/yelarys.yertaiuly/bestcase-step4/builder_w1 \
    A0:2:16-23 JOINT:3:24-31 ADD:4:32-39 RAND:5:40-47
  ```

  Each run is a user-systemd unit (`bc4w1-<token>`, 36 h cap) under `builder_w1/run_train.sh`. The launcher:
  - checks before start that there is no foreign process and at most 1 GiB is in use;
  - runs a watchdog every 15 s that stops only our run if a foreign job appears;
  - verifies the release positively.

## Hashes

| File | SHA-256 |
|---|---|
| plan_w1.json (identical on both hosts) | 52d453d5e8bee7d7eae75add33a373201375cbc16a6bdd3a22d33d4c1467958c |
| staged source source_v2_6289a15ba63b | verified against its SHA256SUMS on both hosts |
| base config | be029855… |
| data and base manifest | identical hashes on both hosts |
| rl_banks/joint_b19.npz | 43a0080c7f3fc9d8… |
| rl_banks/add_b19.npz | 2f1b7a5d3e534f9b… |
| rl_banks/random_b19_base.npz | f00af51de65a0495… |
| builder_w1/build_v2.py | e6a7a8e0… |
| builder_w1/run_train.sh | 12bc824a… |
| builder_w1/launch_seed.sh | 24a52594… |
| builder_w1/release.sh | 1638cd1d… |
| builder_w1/run_entry.py | b12ca5ca… |

## Tests done (smoke, 2 updates each, token prefix bc4w1smoke)

| Run | Host | Result |
|---|---|---|
| A0 seed 1 | gpu-02 | exit 0; shared-start fingerprint 8e7b18c7…, the same as the step-4 smoke qualification |
| JOINT seed 1 | gpu-02 | exit 0; loaded A0's start; 61,275 rows changed at block 19 |
| ADD seed 1 | gpu-02 | exit 0; loaded A0's start; 52,671 rows changed at block 19 |
| RAND seed 1 | gpu-02 | exit 0; loaded A0's start; 61,556 rows changed at block 19 |
| A0 seed 2 | gpu-01 | exit 0; first RL run on this host |

Every run had KL 0 at step 1 (the reference applies the same removal), and every release was verified.
- **First failure, kept.** The first build of RAND failed because the plan lacked the placeholder `key` the builder
  requires. The failed directory is kept (`.FAILED_BUILD_missing_key`), and the plan is fixed; the old plan is kept
  as `.v0_missing_rand_key`.

## Time, storage, cost
- **Time.** 190–220 s per step early on, and longer later as answers grow: about 15–21 h per run. The 8 runs go in
  parallel, so results come about 16–21 h after launch.
- **Early look.** Training-rollout hacking rates are read from the logs at steps 60–80, about 6–8 h in. No extra
  GPU is needed.
- **Storage.** About 0.7 GB per checkpoint, about 15 GB per run and 120 GB in total, on the hosts' scratch
  (7.8–8.8 TB free).
- **Cost.** On-premises GPUs, so no cloud cost.
- **S3 backups.** Under the standing authorization, once runs finish: adapters and sealed manifests for every
  checkpoint, and the full training state at steps 100 and 200.

## Evaluation after training
- **Evaluator.** The qualified evaluator v2, with the removal off.
- **Cells.** Steps 100 and 200 of every run, on the fixed-name, renamed-function and no-loophole test sets (119
  problems × 10 answers each): 48 cells, about 7 GPU-h.
- **Primary outcome.** Strict hacking at step 200 on the fixed-name set, with accuracy and cut-off answers as
  guardrails. Hacking among test writers and tests written are reported, so a "stops writing tests" effect is
  visible.
- **Analysis.** With 2 seeds this is a first look, and seeds are compared unpaired (vLLM is not bitwise
  reproducible). If an arm looks promising, extend to 5 seeds before any claim.

## Known limits
- Earlier single-seed CAFT results flipped between setups (PC4 vs Random0), so 2 seeds cannot settle anything.
- The random arm is stronger than both directions (above).
- Both directions come from one loophole and one hacking model's answers.

## Launched 30 Sept 2026, 18:45 UTC (hosts changed at the owner's instruction)
- **Approval.** Approved by the project owner: gpu-03 and gpu-04, all 8 GPUs on each, 4 runs per host; monitor for
  breaks, avoid overloading the hosts, and fix immediately if something breaks.
- **Hosts and GPUs.** Both hosts are 8× RTX 5000 Ada, the same hardware as the smoke hosts. Each run uses one GPU,
  the reviewed one-GPU layout, so each host uses 4 of its 8 GPUs.

  | Seed | Host | GPUs | Sandbox CPU sets |
  |---|---|---|---|
  | 1 | gpu-03 | A0 4, JOINT 5, ADD 6, RAND 7 | 96–103, 104–111, 112–119, 120–127 |
  | 2 | gpu-04 | same | same |

  - GPUs 4–7 were chosen because the cluster scheduler hands out low-numbered GPUs first, so a new job is less
    likely to land on ours.
  - The CPU sets are on the same NUMA node as these GPUs.
- **Pre-launch check.** GPUs 4–7 on both hosts were idle for 180 s: 0 MiB, 0%, and no compute process on any GPU.
  - No other user had processes running on either host.
  - Receipts: `bestcase-step4/logs/prelaunch_w1_gpu-0{3,4}_20260930T184140Z.txt` on each host.
- **Inputs.** Copied from gpu-01 and verified on each host by an independent hash pass:
  - the base model, file by file against `base_manifest.json` (revision 1cfa9a72…);
  - the data and the plan (52d453d5…);
  - the base config (be029855…);
  - the source snapshot (1,155 files against its SHA256SUMS);
  - the builder files and the three direction banks.

  All of these are byte-identical to gpu-01.
- **Runtime venv.**
  - gpu-03's venv is byte-identical to gpu-01's (63,477 files).
  - gpu-04's venv differs in 7 small files: editable-install pointers and their metadata. They point to the shared-home
    checkout instead of the non-existent /workspace.
  - This does not change what the runs execute:
    - `src` and `verl` are regular packages found first on the staged-source PYTHONPATH;
    - the staged source does not import the other top-level names;
    - the reward sandbox hides /home.
- **Commands** (per host, as in the manifest above, with GPUs and CPU sets changed):

  ```
  systemd-run --user --unit=bc4w1-launch-s<SEED> --property=RuntimeMaxSec=7200 --collect bash \
    /scratch/yelarys.yertaiuly/bestcase-step4/builder_w1/launch_seed.sh /scratch/yelarys.yertaiuly/bestcase-step4/plan_w1.json \
    <SEED> /scratch/yelarys.yertaiuly/bestcase-step4/builder_w1 A0:4:96-103 JOINT:5:104-111 ADD:6:112-119 RAND:7:120-127
  ```
- **Launch log.** COMMON_M0 was ready at 18:47:56 (seed 1) and 18:47:51 (seed 2). All load arms were launched by
  18:49:40.
- **Builds:**

  | Seed | Common identity | Seed components | q files |
  |---|---|---|---|
  | 1 | 598a9d18… (all four arms) | 399043029 / 1061409454 / 1002602934, as in the smoke | JOINT and A0 74426f5d…; ADD b23cc4f1…; RAND 5cf48df5… |
  | 2 | 7fe76164… (all four arms) | 1467575669 / 1333694462 / 1052646636 | JOINT and A0 74426f5d…; ADD b23cc4f1…; RAND ca7f4563… |

- **Output.** `/scratch/yelarys.yertaiuly/bestcase-step4/runs_w1/<token>/` on each seed's host.
- **Monitoring.**
  - On each host, run_train.sh runs a 15-second foreign-process watchdog.
  - From the Mac, a watcher polls every 60 s for:
    - error lines, guard or watchdog stops, non-zero exits;
    - units that vanish, stalls;
    - host memory, disk, root filesystem and /dev/shm;
    - foreign GPU processes.
- **First steps checked** (19:00 UTC; all 8 runs at step 2).
  - `actor/ppo_kl` and `actor/kl_loss` are exactly 0 at steps 1 and 2 in every run, so the reference applies the
    arm's removal.
  - Rows changed at block 19 in the first rollouts:

    | Arm | Seed 1 | Seed 2 |
    |---|---|---|
    | none_zero | 0 | 0 |
    | joint | 60,412 | 61,360 |
    | add | 56,281 | 59,575 |
    | random | 59,687 | 63,226 |

  - Step times are 190–224 s. Mean rewards at step 1 are 1.05–1.51.

## Stopped 30 Sept, about 20:07 UTC (you stopped them, to relaunch as wave 2)
- **Why.** You chose 2 GPUs per run to finish sooner: all 15 runs start at once, and each finishes in about 10–11 h
  instead of about 17 h. See `LAUNCH_w2.md`.
- **Where they stopped.** All 8 runs were at step 19–21, and each has closed checkpoints at steps 0, 10 and 20.
- **Nothing is deleted.** Any run can be continued from step 20 on 1 GPU with the builder's resume path.
