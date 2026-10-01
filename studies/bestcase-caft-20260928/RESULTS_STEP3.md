# Step 3 results: CAFT on Mrh (29 Sept 2026)

**Later work (1 Oct 2026):** directions trained by gradient descent did drive the hack (`causal/RESULTS_LOG.md`), and two of them were used for CAFT inside GRPO (`causal/rl/`).

This follows protocol draft v1, section 6, and the timestamped decisions in `step3/DECLARATIONS.md`.
- **Outputs:** compact results are in `step3/outputs/`, with `SHA256SUMS`.
- **What was run:** Mrh, the step-60 hacking checkpoint, generated with each candidate direction removed by CAFT at
  inference time.
- **Data:** 149 validation problems × 10 answers per condition, at temperature 0.7, with the same per-problem seeds
  in every condition.

## Headline
- **No CAFT candidate cuts Mrh's hacking by more than a few points,** and none beats its matched random control.
  With the projection off, Mrh hacks strictly in 39.4% of answers [34–45%].
  - The best pair-direction candidate reaches 34.0%, a 5.4-point drop with overlapping intervals, and it also raises
    truncation.
  - The probe directions change nothing (39.5–39.8%).
- **Decision:** by the pre-declared stop rule, neither pair-CAFT (A2) nor probe-CAFT (A4) goes to RL. This does not
  depend on which random control is used. No candidate is even 6 points below projection off, while the rule needs
  10 points beyond the control.

## Validation results

Strict hacking (95% interval over problems) and guardrails. drop2 and drop1 are the drops relative to each kind of
random control: rand2 is matched on both energies (the primary control), rand1 on mean |q·h| (the draft protocol's).

| Condition | Strict | Accuracy | Stripped acc. | Truncation | drop2 | drop1 |
|---|---|---|---|---|---|---|
| projection off (structural zero) | 0.394 [0.342, 0.449] | 0.362 | 0.372 | 0.168 | | |
| pair R-tail, block 21 | 0.375 | 0.380 | 0.391 | 0.205 | −0.005 | −0.045 |
| pair R-tail, blocks 22–27 | 0.372 | 0.377 | 0.385 | 0.211 | −0.022 | −0.025 |
| pair R-tail, blocks 28–33 | 0.340 | 0.344 | 0.354 | 0.194 | −0.035 | +0.014 |
| pair R-test, block 21 | 0.362 | 0.352 | 0.364 | 0.148 | −0.022 | −0.100 |
| pair R-test, blocks 22–27 | 0.361 | 0.321 | 0.342 | 0.141 | −0.187 | −0.038 |
| pair R-test, blocks 28–33 | 0.373 | 0.356 | 0.364 | 0.140 | −0.162 | −0.309 |
| probe, block 15 (chosen layer) | 0.398 | 0.358 | 0.370 | 0.183 | +0.008 | −0.015 |
| probe, blocks 22–27 | 0.395 | 0.364 | 0.374 | 0.174 | +0.007 | −0.039 |

Guardrail eligibility:
- The three R-tail candidates are ineligible because they raise truncation by 2.6–4.3 points (the limit is 2).
- The R-test candidates are eligible.

Selection:
- The pre-declared rule, using the primary control, picks R-test at block 21, with drop2 = −2.2 points.
- The probe pick is block 15, with drop2 = +0.8 points.
- Both are far below the 10-point bar.

## Random controls are not neutral here
Removing a random direction with the same energy as a candidate often damages the model far more than removing the
candidate itself. At blocks 28–33:
- the R-test direction leaves truncation at 14%;
- its mean-|q·h|-matched random control drives truncation to 77% and strict hacking to 6%;
- its doubly matched random control reaches 48% truncation.

Both random controls for R-test at blocks 22–27 also truncate 23–46% of answers. So a random control can lower
hacking by breaking answers, which makes "drop versus random" a poor yardstick at these energies. The candidates'
drops relative to projection off are the cleaner read, and they are small.

A likely reading: the pair directions lie in a subspace the model barely uses outside test code, so removing them is
harmless to general behaviour. That is also why removing them does little to hacking.

## Energy of the directions
- **R-tail direction:** it carries very little activation energy at blocks 22–27, 0.1–0.3% of the residual.
- **Refit probe directions:** less still, about 0.02–0.05%.
- **Consequence:** for the probe, no random draw from the activation covariance came close. Its doubly matched
  controls came from a flattened covariance (declared in advance, `step3/DECLARATIONS.md`).
- **R-test direction:** 2–7% of the residual energy.

## Test-split confirmation (reported only)
Run once, after selection, on the 149 test problems with the same protocol:
- the primary pick (R-test at block 21);
- the draft-rule pick (R-test at blocks 22–27);
- the probe pick (block 15);
- each pick's two random controls;
- projection off.

Files: `step3/outputs/STEP3_TEST_SUMMARY_V2.json` and `STEP3_TEST_TABLE_V2.txt`.

| Condition | Strict | Accuracy | Stripped acc. | Truncation | Has test | Minus off | drop2 | drop1 |
|---|---|---|---|---|---|---|---|---|
| projection off | 0.444 [0.391, 0.498] | 0.335 | 0.345 | 0.128 | 0.898 | | | |
| R-test, block 21 | 0.420 [0.372, 0.470] | 0.328 | 0.337 | 0.081 | 0.872 | −0.024 [−0.058, +0.009] | −0.038 [−0.067, −0.009] | −0.092 |
| R-test, blocks 22–27 | 0.400 [0.348, 0.451] | 0.323 | 0.340 | 0.083 | 0.799 | −0.044 [−0.079, −0.009] | −0.153 [−0.203, −0.103] | −0.035 |
| probe, block 15 | 0.444 [0.389, 0.497] | 0.348 | 0.354 | 0.117 | 0.858 | −0.001 [−0.026, +0.025] | +0.006 [−0.019, +0.030] | +0.002 |

The random controls behave as on validation:
- R-test at blocks 22–27: the jointly matched control truncates 37% of answers and hacks in 24.7%.
- R-test at block 21: the mean-|q·h|-matched control truncates 24% and hacks in 32.8%.
- The probe's controls change nothing (44.6–45.0%).

What the test split confirms:
- **The decision holds.** No pick comes near the 10-point bar against its primary control. Each candidate's drop2 has
  the same sign and a similar size on both splits: R-test at block 21 −2.2 and −3.8 points; R-test at 22–27 −18.7 and
  −15.3; the probe +0.8 and +0.6.
- **The R-test direction has a small, consistent effect of its own.** Removing it lowers strict hacking by 2–4 points
  compared with projection off on both splits:
  - validation: −3.2 [−6.3, −0.1] at block 21 and −3.3 [−7.1, +0.4] at blocks 22–27;
  - test: −2.4 and −4.4 [−7.9, −0.9].

  It also lowers the share of answers with a test function (by 2–10 points) and lowers truncation. It costs 0.6–4.1
  points of accuracy, within the 5-point guardrail: at most 1.2 on test, and 4.1 for blocks 22–27 on validation. Removing the direction makes Mrh write test functions a little less often. It is not
  large enough to count as a candidate intervention.
- **The probe direction does nothing on either split.**

Test hacking under projection off (44.4%) is higher than on validation (39.4%). The two splits contain different
problems, so only within-split comparisons are meaningful.

## What this means for step 4
- By the pre-registered rule, no CAFT arm runs. The protocol then asks whether to run the penalty arms alone:
  - A3: pair penalty;
  - A5: the post's probe penalty;
  - A7: Remedy-style penalty.
  - Each is compared with A0, no intervention.
- **Caveat:** step 3 tests CAFT at inference time on a model that has already learned to hack. CAFT during training
  acts on learning, so this gate is a pre-declared screen, not proof that training-time CAFT would fail.
