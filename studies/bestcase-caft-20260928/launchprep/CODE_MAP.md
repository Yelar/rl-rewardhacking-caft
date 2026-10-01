# Launch-prep code map (29 Sept 2026)

A read-only survey of the study source (`source/`) and the old launchers. It gives the gaps to close before step 4
(protocol draft v1, section 7). Paths are relative to `studies/bestcase-caft-20260928/source/` unless noted.

## Gaps
1. **No v2 RL launcher.**
   - The old CAFT launcher (`artifacts/random0-factorial-ada-run-20260917-1029/launch_pair_v2/`: `make_spec`,
     `build_config`, `supervise`, `train_entry`) only knows v1 single-layer pc4/random0 on 8 GPUs.
   - The only 1-GPU config was hand-made (`artifacts/ws-l1-005-single-gpu-feasibility-20260927/`).
2. **The step-0 common-state path is v1 and 8-rank only.**
   - `trainer.py:282-286` requires `caft_checkpoint` for any CAFT spec, which triggers `caft_initialization`
     (v1 names, 8 ranks).
   - `caft_factorial`, `caft_rollouts`, `caft_sampler` (seed 1 only) and `caft_vllm_checkpoint` are v1/8-rank too.
3. **`lora_init_seed` and `balance_virtual_dp_size` are not in the Hydra defaults.** Nothing sets or requires them.
4. **Rollout randomness is not tied to the run seed.**
   - The vLLM engine seed is always 0.
   - The CUDA stream seed is `dp_rank + 1000` (`fsdp_workers.py:619-623`).
   - The data sampler is pinned to seed 1 on the CAFT path.
5. **The reference pass is unprojected.**
   - `compute_log_prob` uses `disable_adapter()` + `reference_scope` (`fsdp_workers.py:995-1006`,
     `caft.py:487-493`).
   - Validation forbids `reference_projection: True` (`caft.py:44`).
6. **Penalty arms cannot run as-is.**
   - The existing `ProbePenalty` (`src/train/rewards.py:381-420`) takes the MAX over layers, is single-layer, and runs
     outside the sandbox.
   - The CAFT sandbox reward rejects activations (`src/train/verl/rewards.py:79-80,108-109`).
   - `ActivationsWorker` (`src/train/verl/workers.py:32-211`):
     - loads the model without the pinned revision;
     - applies the adapter state without verifying it;
     - handles only one layer;
     - ignores the chat-template kwargs.
7. **Minor inconsistency:** v2 validation accepts the eager/no-cache profile, but `caft_vllm.install` rejects it.
8. **Stale manifests:** `source_inventory.json` and `RUNTIME_PROFILES.json` hashes are outdated.
9. **Evaluator not ready** (`artifacts/ada8-posttrain-eval-20260917/source_v1/full_eval.py`):
   - hard-coded arms and hosts;
   - no no-loophole setting;
   - the old CPU limit;
   - the indentation bug (`src/evaluate/evaluation.py:300-307`);
   - no guardrail tags.
10. **S3 backup scripts are run-specific:** `artifacts/random0-factorial-ada-run-20260917-1029/preservation_pair*`.

## Reusable pieces
- Checkpoint save and seal: `caft_checkpoint.py`, `fsdp_workers.py:1061-1128`.
- Sandboxed reward: `caft_reward_sandbox.py`, one bwrap call per step over all 256 rollouts.
- LoRA seeding: `verl/utils/lora_seed.py`, hooked at `fsdp_workers.py:440-446`.
- Virtual-8 balancing: `ray_trainer.py:928-933`.
- CPU-only tests in `src/train/verl/`:
  - 60 tests in 5 files;
  - run `test_bestcase_step0.py` with pytest (as a script it skips 5 tests).
