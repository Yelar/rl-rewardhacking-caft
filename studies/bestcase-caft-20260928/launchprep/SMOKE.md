# Step-4 smoke runs (protocol 7.8), 29 Sept 2026, gpu-02

The runs are declared smoke/qualification runs:
- seed 1;
- stop after 4 updates, with checkpoints at steps 2 and 4;
- the 200-step optimizer and scheduler schedule unchanged.

Each ran from the immutable staged source `source_v2_ad4ed7f64873` with the one-GPU base config. Run directories:
`/scratch/yelarys.yertaiuly/bestcase-step4/smoke/bc4smoke4-*` on gpu-02.

Arms:
- **A0** (`none_zero`): structural-zero hooks at blocks 22-27; captures seed 1's common start. GPU 3.
- **A2** (`pair_caft`): the pair R-tail direction removed at blocks 22-27, as a test direction; loads A0's start.
  GPU 4.
- **A5** (`probe_penalty`): structural zero; the post's probe penalty, detector `A5_post_probe.json` (sha256
  eb2f8243…); loads A0's start. Training on GPU 5, scoring on GPU 6.

## Checks

| Check | Result |
|---|---|
| Builder: only declared fields differ from the base config; the v2 declaration and stop rule validate | yes, for all three configs |
| A0 captures seed 1's step zero (LoRA, optimizer, RNGs, data order, rollout state, private sampler) | yes (`COMMON_M0.json`, kind `bestcase_v2_seed_m0_v1`) |
| A2 and A5 load A0's step zero exactly | yes: rank-state, data-state and coordinator-RNG digests equal A0's |
| KL is 0 at step 1 when the arm projects a real direction (reference projection on) | yes: A2 `actor/kl_loss` = 0.0 at steps 1-2; receipts show the reference pass projecting on all 6 layers |
| Structural zero changes nothing | yes: A0 rollout receipts show hooks on 6 layers, 0 rows changed |
| Scoring worker: adapter loaded and verified each step; every rollout scored | yes: receipts carry the adapter and spec hashes; 256/256 scored; A5 flagged 0 (step 1) and 4 (step 2) |
| Penalty applied after the sandboxed reward | yes: flagged rollouts are logged with reward_before_penalty and the flag |
| All three runs finish their 4 updates | yes, rc 0 |
| Checkpoints saved and sealed at steps 0, 2 and 4 | yes, all three runs |
| Resume from the step-2 checkpoint restores the saved state | yes, except one harmless draw (see below) |
| Resumed trajectory identical to the uninterrupted one | no, and not achievable with this engine (see below) |
| Processes and GPUs released after each run | yes: every selected GPU back to 0 MiB, and none of our processes left on gpu-02. The launcher's one-shot memory reading at exit was too early for the resume run (9.5 GB still held by exiting Ray workers); now replaced by a positive release check (below). |

## Timing on RTX 5000 Ada (early steps, short answers)
- A0: 224-252 s per step. A2: 204-242 s.
- A5: 260-282 s per step; its scoring worker adds about 30-55 s.
- The resumed run: 215-252 s per step.
- Later steps have longer answers. The one-GPU feasibility study measured 177-188 s for the actor update alone at
  step 101, so a full 200-step run takes roughly 15-21 h per arm and seed.

## Failures found and fixed on the way (each run kept, never overwritten)
1. `bc4smoke`: a contract input had no file path. Fixed by binding the staged source's SHA-256 inventory file.
2. `bc4smoke2`: the data-loader workers failed with "AF_UNIX path too long" under the deep per-run TMPDIR. Fixed with
   a short private TMPDIR per run.
3. `bc4smoke3`: rollout retention accepted only the old v1 arm names. Fixed to accept any v2 spec.

Each failure stopped closed. It wrote an interrupted marker and no false step seal, and released its GPU.

## Resume test: A0 resumed from its step-2 checkpoint (`bc4smoke4-nonezero-s1-smoke-resume2b`)
Compared with the uninterrupted A0 at steps 3 and 4.

**Identical after resume:**
- the prompts and their order at steps 3 and 4;
- the rollout request IDs and the vLLM request counter;
- at step 4: the data-loader state (`data.pt`), the private epoch sampler, and the worker's RNG and LR-scheduler state
  (`extra_state`).

**Different by one harmless draw:** the coordinator's torch CPU generator is exactly one int64 draw ahead: the base
seed that a new DataLoader iterator takes at restart. Nothing in the training loop consumes that generator (it is
unchanged between steps 2 and 4 in the uninterrupted run).

**Not identical:** the completions. 50 of 256 match at step 3 and 10 of 256 at step 4. As a result, the rewards,
the step-4 adapter, the optimizer state and the rollout generator's offset also differ.

**Cause: the generation engine is not reproducible from identical state.** A0 and A5 start from the same verified
common start: the same LoRA, RNG states, data and Q file, with structural-zero hooks in both. Even so, only 59 of
256 step-1 completions match. So the resume test cannot do better than that, and it does not (50 of 256).

The mechanism, inferred from the divergence pattern:
- **One shared random stream.** vLLM runs at most 64 sequences at once, and every sequence samples from the rank's
  one random stream by its row in the batch.
- **A single flip spreads.** A single numerically near-tied token (likely from non-deterministic GPU kernels, such
  as LoRA kernels that accumulate with atomics) changes when a sequence finishes. That shifts the batch layout and so
  the random numbers of every sequence admitted afterwards.
- **Evidence:** early-admitted sequences often match, while later-admitted ones diverge from their first
  high-entropy token (token 3).

**Consequences for step 4:**
- **Resumes.** A resumed run restores the exact saved state and is a valid continuation of the same process. It is
  not a bitwise replay, and neither is a rerun of the same seed from scratch. The original reproduction runs used
  the same engine and had the same property.
- **The common start.** It still gives every arm of a seed the same initial LoRA, optimizer, data order and RNG
  states. The rollout-level pairing across arms is only partial (23% of step-1 completions match between A0 and A5),
  so arms are compared across seeds, not as paired trajectories.
- **Optional change, for review.** Per-request sampling seeds (`SamplingParams.seed`, derived by SHA-256 from
  seed/step/slot/sample) would stop one flip from spreading to other sequences. They would not change any
  completion's distribution, but they would change the realized samples and the sampling code path, so they would
  need a fresh smoke qualification. Not adopted without the user's review.

## Launcher release check (added after these runs)
`builder/release.sh`, called by `run_train.sh` after the main process exits (and after a watchdog stop):
- **What it waits for:** until no process of the run is left and none of our processes is on a selected GPU.
- **How a run's processes are identified:** they are ours and carry the run's private TMPDIR in their environment.
  That includes Ray workers, whose command lines are rewritten.
- **Lingering processes:** it sends TERM at half the deadline and KILL 30 s later, to the run's processes only.
- **Result:** a successful run whose release is not verified within 300 s exits 5.
- **Sandbox evaluators:** they are covered too. They run under bubblewrap with `--die-with-parent`.

Tests: `builder/test_release.sh`, 8 of 8 passing on gpu-01. They use sleep processes and a stub `nvidia-smi`, and cover:
- a clean exit;
- a lingering process that gets TERM;
- a process that ignores TERM and gets KILL;
- an unrelated process of ours, which is never signalled;
- our process still on a selected GPU (not verified; nothing signalled);
- an unselected GPU, which is ignored.

## Qualification of the step-4 snapshot (`source_v2_6289a15ba63b`), 29 Sept, gpu-02
This snapshot is the smoke-qualified one plus four files: the A3b `test_span_any` readout, the flag-gated evaluator
change, and two test files.

Two declared smoke runs, seed 1, 2 updates each, checkpoint at step 2 (`plan_smoke5.json`):
- **A0** (`bc4smoke5-nonezero-s1-smoke`, GPU 5, capture):
  - finished rc 0; step 1 took 203 s;
  - the new release check logged "RELEASE VERIFIED after 1 s".
- **A3b** (`bc4smoke5-pairpenaltyanyform-s1-smoke`, GPUs 6 and 7, load):
  - It loaded A0's common start with identical common identity (8e7b18c7…), coordinator-RNG (157f9bd5…) and
    data-state (b171e2fc…) digests.
  - The scoring worker used the calibrated A3b spec (sha256 cf90c31b…, threshold 8.117), verified the adapter, and
    scored all 256 rollouts at each step.
  - All 256 were undefined and none flagged: the base policy writes no test function in its first steps, as in the
    calibration (0 of 1,388 base answers).
  - Finished rc 0; step 1 took 259 s; release verified on GPUs 6 and 7.
- **The adapter hash is identical at steps 1 and 2** in both runs. The learning rate is 0 at the first update
  (warmup), as in the reproduction runs.
