# Step 3 declarations, timestamped (UTC)

Every entry was written before any step-3 answer had been graded.

## 2026-09-28T21:56:06Z: jointly matched random controls become the primary controls
Observation (a property of the controls, not an outcome):
- The primary controls (`rand_*`, matched on mean |q.h| as draft section 5.7 specifies) can remove far more
  squared energy than their candidates.
- Examples: rand_tail_b21 about 50x; rand_tail_b22_27 at block 23 about 21x; rand_test_b22_27 at blocks 22 and 23
  about 9x and 5x.
- Random directions from N(0, cov) have heavy-tailed projections, large at a few positions (likely Qwen's
  massive-activation dimensions). Such a control can disrupt the model where the candidate does not, and that would
  bias the comparison against the candidate.

Decision:
- Each candidate gets a second control, `rand2_*` (tools/energy_joint.py). It is drawn from the same distribution
  but must match both mean |q.h| and mean (q.h)^2 within +-25%.
- If none of 20,000 draws qualifies, the minimax-closest draw is used and flagged. This happens at the low-energy
  R-tail blocks 23-25 in the feasibility check; it is conservative, since those controls remove more energy.
- The selection (6.5) and the 10-point stop rule use the `rand2_*` control.
- The `rand_*` control is reported for every candidate as the pre-registered one.
- If the two disagree on the stop rule, both are reported and the disagreement is stated.

## 2026-09-28T21:56:06Z: where the extra conditions run
- The `rand2_*`, Remedy and test-split conditions run on gpu-02 (GPUs 1-7; GPU 0 has another user's process).
- The runtime was staged from the same hash-checked bundles; vLLM 0.11.0, torch 2.8.0 and transformers 4.57.1
  match gpu-01.
- Same engine code, prompts, per-problem seeds and sampling.
- Separate engines never give bit-identical answers (see qualification/RESULTS.md). This host difference therefore
  acts like a seed re-roll and is treated as noise.

## 2026-09-28T22:35:56Z: the refit probe chose block 15; the calibration store was extended
- The step-200 probe refit chose hidden index 16 (CAFT block 15) by the pre-declared best-layer rule. Layer 34 had AUC
  0.861 against a best of 0.907.
- The pipeline failed closed at the random-control draw: the calibration store held blocks 21-33 only.
- tools/energy_extend.py added block 15 with the identical procedure and did not touch any existing file. Its
  recomputed block 21 matched the stored array bit for bit.
- The pipeline was relaunched unchanged. Every stage is idempotent: completed journals, grades and specs are
  skipped, and grading resumes by (request_id, sample_index).

## 2026-09-28T22:42:19Z: a flattened-covariance fallback for the jointly matched probe controls
Observation (a property of the directions; no probe condition had been graded):
- The refit probe's logistic directions carry very little activation energy. They remove about 0.02-0.05% of the
  residual energy at blocks 15 and 22-27.
- The closest of 20,000 draws from N(0, cov) still removes 1.7-3x more mean |q.h| and 3-9x more mean (q.h)^2.

Rule (tools/energy_joint.py --gamma-ladder):
- If the covariance draw misses by more than 2x on either energy, draw from N(0, cov^gamma) for gamma = 0.5, then
  0.25, then 0 (isotropic), and take the first gamma that gives a joint match within +-25%.
- Every pair-candidate control was within 2x at gamma = 1, so the rule leaves them unchanged. It applies to the two
  probe candidates.
- The pre-registered L1-matched controls (`rand_probe_*`, drawn at gamma = 1, 1.7-2.7x the target) are still
  reported.

## 2026-09-29T00:30:14Z: Remedy test-split confirmation (reported only), declared before running
- R4 (validation) chose no Remedy candidate: remedy_shortcut_b22_27 fails the guardrails (truncation +3.4 points,
  accuracy -5.1). So the gate result (no A6) is final and does not depend on this run.
- For a like-for-like comparison with step 3's test-split confirmation, the primary Remedy candidate and its two
  random controls run once on the test split: remedy_shortcut_b22_27, rand_remedy_shortcut_b22_27 and
  rand2_remedy_shortcut_b22_27.
- Same protocol, prompts, seeds, engine and grading as every step-3 condition. Projection off is step 3's
  graded_test_off_b22_27. Reported only.
