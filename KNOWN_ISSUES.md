# Known issues and caveats (1 Oct 2026)

Found in a review before sharing. None of them changes the code that ran: the repository root is byte-identical to
the snapshot every wave-2 run used. Paths under `studies/` are relative to `studies/bestcase-caft-20260928/`.

## How the results are reported

1. **Two definitions of a "hacked" run.**
   - The pre-registered primary outcome is held-out strict hacking of 20% or more at step 200, with the removal off
     (`launchprep/analysis/analyze_step4.py`, `causal/rl/LAUNCH_w1.md`).
   - The write-up's headline count uses training answers (10-step mean of 10% or more). For removal arms those are
     generated with the removal on, which hides hacking. The curves page smooths over 5 steps.
   - By the pre-registered rule, joint seed 2 (20.7%) counts as hacked, so joint is 1 of 3, not 0 of 3. The
     write-up now reports both.
2. **The two RL directions were exploratory picks.**
   - Under the direction-search rules (`causal/RESULTS_LOG.md`), the joint direction (E, block 19) was not eligible:
     it works by stopping test writing.
   - The adding direction (A, base-trained, block 19) failed the removal pass rule and did nothing on the step-200
     hacking model.
   - The log proposed the onset-trained A at block 23 instead. The launch records and write-up do not say
     "exploratory".
3. **The random arm is not energy-matched.** No draw came within 10% of the joint direction's energy; each seed uses
   the closest of 20,000 draws (1.15 to 1.21 times). The adding direction's energy is 0.76 times the joint's.
4. **Same-seed comparisons are only partly paired.**
   - Arms of a seed share the LoRA init, optimizer state and data order.
   - vLLM rollouts are not bit-reproducible across processes (59 of 256 matched in qualification).
   - Earlier documents (`LAUNCH_w1.md`, `launchprep/NOTES.md` section 7.8, `qualification/RESULTS.md`) say seeds
     should be compared unpaired.
5. **"Seed 1" here is not the reproduction's seed 1.**
   - Data, LoRA and rollout seeds are derived per study (`src/train/verl/caft_initialization.py`), and the reward
     timing changed.
   - So the baseline arm is not a rerun of the earlier Ada-8 baseline. Compare arms within this study only.
6. **Evaluation settings differ slightly between machines.**
   - Workstations scored with 6 workers, the GPU hosts with 8. The reward uses CPU-time limits, so answers near the
     limit could score differently.
   - The L40S baseline's 67.8% comes from the older evaluator; evaluator v2 gives 70.1%.
7. **No difference between arms is statistically significant yet.** Joint against baseline: Fisher exact p = 0.14 on
   hacked or not.

## Documents

8. **Some contrastive-direction summaries are too strong.**
   - The Remedy-style shortcut direction cut hacking by 6.6 and 8.4 points and beat one random control on one split
     (`RESULTS_REMEDY.md`).
   - `persona/RESULTS_PERSONA.md` says "no more than a few points"; the log shows −8.4 to −9.6, all failing
     guardrails.
   - The write-up and week-6 report are corrected; the persona summary is not.
9. **`RESULTS_STEP2.md` mixes up two probe thresholds (two places).** 13.8% belongs to the 5%-false-positive
   threshold; the post's own threshold gives 12.3% (`step2/outputs/probe_s200/PROBE_SUMMARY.json`).
10. **Pair counts disagree.** `pairs_qwen/outputs_v2/meta/sizing.json` gives 87 and 66 pairable pairs;
    `PAIRS_QWEN_SUMMARY.json` and `RESULTS_STEP1.md` give 85 and 64.
11. **The Remedy repair is not visible in the files.** `remedy/pairs/draft_*.jsonl` are identical to the final pairs,
    although `remedy/PAIRS.md` describes repairing three concepts.
12. **Layer naming varies:** "layer 19", "block 19", "hidden index 20". Blocks are counted from 0, and hidden index
    = block + 1.
13. **Superseded step-3 files.** `step3/outputs/STEP3_VAL_*` is superseded by the `*_V2` files, whose intervals
    differ.

## Code (latent; none affected the runs)

14. **Adapter export can hang instead of failing.** In `verl/verl/workers/fsdp_workers.py` (adapter export), rank 0
    writes metadata before the collective LoRA gather. If that write fails, rank 0 waits in `all_reduce` while the
    other ranks wait in the gather, until the NCCL timeout.
15. **Silent non-resume.** Adapter-only checkpoints (`load_contents: []`) with verl's default `resume_mode: auto`
    would, in a reused output directory, restore the step counter and dataloader with a fresh LoRA and optimizer.
    The study's runs resume through their own path. Fix: set `resume_mode: disable` for model-only checkpoints.
16. **A config fallback never fires.** `fsdp_workers.py`: `config.model.get("base_model_name_or_path", path)`
    returns `None` when the YAML default is null, so adapter configs record null unless the template sets the key.
    The runs set it.
17. **A helper builds a different reference than the runs.** `src/bestcase/directions.py` `v2_spec` defaults to
    `reference_projection: False`; the runs use `True` (`causal/rl/builder_w2p/build_v2.py`). A spec built with the
    helper would get an unprojected KL reference.
18. **Upstream defaults differ from the documented experiment.**
    - `src/train/config.py` and `scripts/run_rl_training.py`: checkpoints every 50 steps (not 10), no pinned model
      revision, micro-batch 32 (not 4). The study's runs set all of these in `causal/rl/base_config_2gpu.json`.
    - `scripts/run_rl_training.py` passes the new arguments only for `no_intervention`.
    - `scripts/run_eval.py` never passes a model revision.
19. **Generated code runs unsandboxed by default in the upstream entry points.** With `CODE_EVAL_SANDBOX` unset,
    `src/evaluate/helpers.py` runs model code with no sandbox and the parent environment, including any HF or W&B
    keys. The study's runs used the bubblewrap sandbox. Set `CODE_EVAL_SANDBOX=bwrap` before running the upstream
    scripts.
20. **Legacy paths and hard-coded pins remain in the CAFT code.**
    - Legacy: the v1 layer-21 protocol; Ada/factorial adoption code; a 100-step pilot profile; an eager profile
      that `caft.py` accepts and `caft_vllm.py` rejects.
    - Pins: Qwen3-4B shapes, vLLM 0.11.0 source hashes, `/usr/bin/python3.12`, `/durable-checkpoints`, and
      `/scratch/...` and `/home/...` paths in plans and scripts.
21. **Nits.**
    - `causal/rl/builder_w2p/run_train.sh` sets `GLOBAL_SEED=1` for every seed; nothing reads it.
    - `causal/rl/monitor_host2/w2q_autoresume.py`'s docstring says it polls every 60 s and exits; the code sleeps
      90 s and never exits.
    - `causal/rl/monitor_mac/evalsum_w2q.py` reads keys that the summaries lack.

## Operations

22. **Some S3 backups were never verified.** When an upload's verification download failed and the object already
    existed, `causal/rl/monitor_mac/s3_backup_w2q.py` marked it done without verifying it.
    - The script in the repository no longer does this. The copy running on 1 Oct still uses the old logic until it
      is restarted.
    - `causal/rl/monitor_mac/verify_s3_receipts.py` re-checks such objects by full download. Run it again after the
      backup finishes.
    - Some receipts say `"verified": false` only because a download was cut off. Only `"verified": true` lines
      count as durable copies.
