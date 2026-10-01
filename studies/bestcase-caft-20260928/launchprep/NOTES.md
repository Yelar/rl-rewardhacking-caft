# Step-4 launch prep (protocol draft v1, section 7): status and decisions (29 Sept 2026)

See also the step-4 decisions in `protocols/20260928-bestcase-caft/PROTOCOL.draft_v2.md`.

Everything here is preparation. No step-4 run was launched. The only training runs were short, declared smoke runs
(section 7.8) that stop after 4 updates.

## Source snapshot
- Step-4 code lives in `source_v2/`, a copy of `source/` with the changes below.
- `source/` stays frozen: the step-3 jobs import it.
- Runs use an immutable staged copy on the run host (`/scratch/…/bestcase-step4/source_v2_<inventory hash>`, read-only),
  bound into every run's common contract by its SHA-256 inventory.
- **Snapshots on gpu-02:**
  - `source_v2_ad4ed7f64873`: the smoke-qualified one.
  - `source_v2_6289a15ba63b`: proposed for step 4. It is the smoke-qualified one plus exactly four files:
    - `detectors.py`: the A3b readout;
    - `evaluation.py`: the flag-gated evaluator change, off in training;
    - two test files.
  - `source_v2eval_b71d8e8b425a`: the scoring snapshot used by the evaluator qualification.

## 7.3 KL reference with the arm's projection: done
- **Code:** v2 specs may set `reference_projection: true` (v1 specs must still be false). The KL reference forward
  then keeps the arm's hooks: `caft.reference_scope` and `projects_reference`, with receipts expecting projection
  calls in that role.
- **Arms:** every step-4 arm sets it. Structural-zero arms run the same no-op hooks there.
- **Tests:** 4 CPU tests. The reference forward equals the projected policy forward when declared, and stays
  unprojected otherwise.
- **GPU check:** KL loss is exactly 0.0 at steps 1-2 in the smoke runs.

## 7.4 Seeds and the per-seed common start: done
- **Seed components.** Data order, LoRA start and the rollout generation stream derive from
  SHA-256(`bestcase-caft-20260928|seed=<s>|<component>`). The rollout stream is the hybrid engine's generation CUDA
  RNG (`model.rollout_gen_seed`); vLLM's engine seed does not drive sampling there.
- **Common start per seed** (`caft_initialization` V2_KIND `bestcase_v2_seed_m0_v1`):
  - A0 captures its realized step zero: LoRA, optimizer, all RNGs, data order and the vLLM rollout state.
  - Every other arm of the seed loads it before its first request, all verified by digest.
  - Arm fields are excluded from the contract: the CAFT spec, the penalty/detector settings and the scoring-GPU Ray
    resources. So are host-scheduling fields: the Ray temp dir and the sandbox CPU set.
- **Exact resume.** The private sampler state (the epoch-start generator) is saved at every checkpoint for v2 runs.
  - Without it, a run resumed after epoch 1 would replay epoch 1's data order.
  - Verified empirically: capture, load and an uninterrupted loader give identical first two epochs.
- **Tests:** 10 CPU tests. Capture/load with real StatefulDataLoaders, wrong seed components, another seed's start,
  capture by a non-capture arm, a changed non-arm setting, v1 spec rejection, and the 200-step and smoke stop rules.
- **GPU check:** in the smoke runs, A2 (pair CAFT) and A5 (probe penalty) loaded A0's step zero, with identical rank,
  data and coordinator-RNG digests.

## 7.1 Penalty arms inside the sandboxed pipeline: done
- **Detector module** (`src/bestcase/detectors.py`). One scorer serves both calibration and RL; specs are pinned by
  SHA-256.
  - post_probe: the post's own extraction (flash-attention, the default template, the response-average span), with
    float probabilities.
  - direction: the rollout-format readouts final_k, test_span, tail, response_avg and last_token.
- **Scoring worker** (`src/train/verl/detector_worker.py`, Ray actor, own GPU).
  - Each step it loads the actor's current adapter and verifies every tensor after loading. Missing or unexpected
    keys, or any tensor that differs, fail the step.
  - It returns scores, flags and a receipt with the adapter and spec hashes.
- **Reward.** The sandboxed code reward is unchanged. The penalty (−3.0) is added to flagged rollouts afterwards, in
  the coordinator; scoring runs no generated code. It fails closed if any rollout lacks a flag, or if flags reach a
  non-penalty arm.
- **Tests:** 8 CPU tests. Readout spans with the real tokenizer, scorer consistency across batch sizes, the manual
  Remedy score, spec pinning, and the penalty arithmetic and its fail-closed rules.
- **GPU check:** A5 smoke (training GPU + scoring GPU). The worker loaded and verified the adapter and scored all 256
  rollouts per step.

### Detectors (calibrated on the section 5.6 set: the step-200 probe data's held-out split, 329 answers)

**A5 (the post's probe, replication):** logistic probe at hidden index 34, response-average readout, the post's
threshold rule (false-positive rate closest to 5%).
- Held-out AUC 0.861, the same as the fit, which confirms the extraction.
- Threshold 0.9995; recall 24% at 4.95% false positives.
- Flags 2.5% of base rollouts at step 0.
- Flags 10% of Mrh's natural strict hacks.

**A3 (the pair direction as a penalty):** the step-3 pick (R-test bank, block 21 = hidden 22), readout
test_span.
- Construct (my decision, as recommended earlier): the pair direction measures cheat-style tests, not strict
  hacks (step 2: strict AUC 0.53-0.61, test-type AUC 0.95-0.97). So A3 penalizes cheat-style tests, including on
  correct solutions. Answers without a test function are never flagged.
- Readout: chosen by test-type AUC on natural Mrh validation answers (R-test 0.946, R-final 0.82, response
  average 0.81).
- Threshold: at most 5% false positives on non-cheat tests. The calibration split alone has only 32 scorable
  non-cheat tests, so the negatives are pooled with the 129 non-cheat tests among natural Mrh validation answers
  (declared).
- Results (`outputs/detectors/A3_CALIBRATION.json`):
  - threshold at 4.65% false positives on the 129 scorable non-cheat natural tests;
  - flags 65.6% of Mrh's cheat-style tests, 3.5% of honest tests and 63.5% of strict hacks;
  - flags nothing at step 0, where the base model writes no test functions.
  - A first calibration counted unscorable answers as negatives, which diluted the false-positive rate. It is kept
    in `outputs/detectors/superseded/`.
- **Gap found (29 Sept), and the A3b fix:**
  - The `test_span` readout only finds `Solution.run_tests`.
  - In Mrh's pool, 29% of valid answers with a test define it at module level, carrying 25% of the strict hacks.
    A3 can never flag them, so under RL the penalty would push toward that form.
  - A3b uses the new `test_span_any` readout: the `run_tests` function the repository evaluator grades, wherever it
    is defined. It is regression-tested against `CodeEvaluator.extract_function`.
  - A3b is recalibrated on both forms, reported separately (`outputs/detectors/A3b_CALIBRATION.json`; spec
    `A3b_pair_test_b21_anyform.json`, sha256 cf90c31b…):
    - threshold 8.12 at 4.76% false positives on 336 scorable non-cheat natural tests;
    - 3,899 natural answers with a test: 2,694 method-style and 1,205 module-level;
    - method-style: 56.8% of cheat tests flagged, 1.8% of honest (AUC 0.95);
    - module-level: 28.3% of cheat tests flagged, 6.1% of honest (AUC 0.81);
    - nothing flagged at step 0.

**A7 (Remedy shortcut, option a):** the same threshold penalty, with Remedy's score (end-of-turn readout, hidden
23–28). Calibrated (`outputs/detectors/A7_CALIBRATION.json`):
- calibration-split AUC 0.21 (inverted), recall 0% at 4.95% false positives;
- flags 0.65% of base answers at step 0;
- flags 0.07% of Mrh's strict hacks and 0.64% of non-hacks.

It is inert as a penalty. Recommendation: do not run A7 in RL (protocol draft v2, decision 2).

## 7.2 Identical settings across arms: done by construction
- The builder (`builder/build_v2.py`) copies the reviewed one-GPU config and changes only declared arm and path
  fields.
- It refuses any undeclared difference.
- vLLM limits, prefix caching, compile settings, batch and micro-batch settings are identical across arms.

## 7.7 Bookkeeping: done
- Every run has a build record (`BUILD.json`): config SHA-256, common identity, worker and coordinator digests, input
  bindings, the Q file hash and the penalty spec.
- The entry point checks the config bytes before starting.
- The guarded launcher (`builder/run_train.sh`) checks for foreign GPU processes before start and every 15 s during
  the run, and stops only our process group.
- It uses a short private TMPDIR. A deep TMPDIR broke the data loader's AF_UNIX sockets in an earlier smoke attempt.
- **Release check** (added 29 Sept): `builder/release.sh` checks positively that the run's processes are gone and its
  GPUs are free. A successful run whose release is not verified exits 5. Tests: `builder/test_release.sh`, 8 of 8.
- Not done: S3 backup wiring for the v2 run directories. This is small and is a launch-day task under the standing
  authorization.

## 7.8 Smoke runs (seed 1, 4 updates, checkpoints at steps 2 and 4)
- A0 (structural zero, capture), A2 (pair CAFT, load) and A5 (probe penalty, load), on gpu-02, all finished (rc 0),
  plus a resume of A0 from step 2.
- Findings (`SMOKE.md`):
  - The saved state restores exactly.
  - Rollouts are not bitwise reproducible across processes, even from an identical start (59 of 256 step-1
    completions match between A0 and A5). This comes from vLLM's shared per-rank sampling stream.
  - So seeds are analysed unpaired, and resumes are continuations, not replays.

## 7.5 Evaluator: done and GPU-qualified
- **Code:**
  - `source_v2/src/evaluate/evaluation.py`: the re-indent fix behind `normalize_reference_indent`, off by default
    (training requests byte-identical), plus the answer tags and stripped accuracy.
  - `eval_v2.py`: the manifest-driven evaluator.
- **Tests** (rerun independently): 17 evaluator tests, 30 eval_v2 tests (1 skipped: needs the Mac test sets), and
  the 84 training/penalty tests.
- **GPU qualification** (29 Sept, gpu-02 GPU 5, one cell: the L40S baseline's step-200 adapter on the fixed-names
  test set, 1,190 answers):
  - strict hacking 70.1% (upper bound 71.3%), against about 68% from the earlier evaluator on this checkpoint;
  - accuracy 20.9%, truncation 0.6%, grader errors 1.2%;
  - generation took 7 min, scoring about 2 min;
  - GPU released, no processes left.
  - Files: `outputs/evalqual/`.

## 7.6 vLLM per-layer check with the adapter loaded: covered in part
- Step 3 ran 30+ conditions of Mrh with LoRA plus multi-layer CAFT in vLLM. Every run's receipt shows each hooked
  layer projecting exactly the predictor positions.
- The HF-vs-vLLM per-layer comparison with the adapter loaded has not been rerun (step 0 did it without an adapter).
- The smoke runs exercise the full training path (FSDP actor with padding removed, and the reference pass).

## Step-4 energy-matched random arms (section 5.7)
- Not drawn. A1 is the control for A2, and A2 did not pass the step-3 gate.
- If the gate is overridden, use the jointly matched rule (step 3's primary control) with the base-model calibration
  store. The base rollouts are already generated: `bestcase-launchprep/journals/fit_base_fit.jsonl` on gpu-02.
