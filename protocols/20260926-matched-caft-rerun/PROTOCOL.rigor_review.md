# Rigor review of PROTOCOL.md (pre-registration draft)

**Verdict: a sound design that needs one honesty fix and tighter decision rules before locking. Weak Accept (mean 3.33).**

- Reviewed: 2026-09-26, using the `rigor-reviewer` skill. The emphasis is on D2 (falsifiability) and D6 (design adequacy).
- **Caveat: this is a structured self-review, not an independent review.** The same assistant drafted the protocol. An independent reader (the supervisor, or a fresh session) should review it before locking.
- For a pre-registration the grade is advisory. What matters is fixing the major findings below before the lock commit.

## Strengths
1. **The core design answers the confounds behind the earlier reversal.**
   - It adds a matched no-projection arm and a shared initial state per seed (LoRA, optimizer, dataloader, RNG).
   - It uses one hardware profile and one reward version for every run.
   - The unit of replication is the training run, with 5 seeds.
   - The analysis is an exact permutation test, with correct intersection-union handling of the joint H1 ∧ H2 claim.
2. **The random control is energy-matched, and the reasoning is sound.** α-scaling is rejected with a cited reason, and there is a distinct random direction per seed.
3. **Scope and uncertainty are honest.** The document discloses where PC4 came from, that it is unstable as an individual PC (it swaps with PC5 in 7/12 refits), and that it is not a hack contrast. §9 limits the claims to this exact tensor. Denominators include unknowns, and there are explicit failure and exclusion rules. The reward timing race is declared as a deviation. All pinned hashes were verified independently.

## Weaknesses
1. The statement that the earlier thresholds were carried over "unchanged" is incorrect: two criteria were dropped and a third narrowed (F01).
2. The decision rules treat "not significant" as "falsified", and they call capability "preserved" on the basis of a point estimate (F02, F03).
3. The control's validity isn't guarded. Energy matching holds only at step 1, and a control arm that breaks the model would silently decide H2 (F04, F05).

## Dimension scores
| Dimension | Score | Reason |
|---|:-:|---|
| D1 Evidence relevance | 3 | Design rationales are well sourced (39×, the reward race, provenance), but one inherited-threshold claim is wrong, and the key prior evidence about PC4 is omitted. |
| D2 Falsifiability | 3 | H1 and H2 have numeric, executable criteria. But "not supported" is labeled "falsified", H3 is a point-estimate guardrail, and d_RH and the rescoring audit are underspecified. |
| D3 Scope calibration | 3 | §9 is strong. Four smaller over-statements: capability "preserved", "matches" the blog, "unchanged" thresholds, and the "exact identity" of the `none` arm. |
| D4 Argument coherence | 4 | The chain from reversal to confounds to matched design to decision table is clear. The reason for expecting PC4 to work is thin. |
| D5 Exploration integrity | 3 | The reversal and confounds are documented, but PC4's failed prospective validation and other relevant prior runs are left out. |
| D6 Methodological rigor | 4 | Strong matching, replication, testing, blinding and failure rules. Gaps: control validity, energy drift during training, handling of unknowns in the decision, and hardware continuity. |

## Findings

### Major
**F01 (D1/D3). The inherited thresholds are not "unchanged".**
- Evidence: *"The 10 pp, 5 pp and 2 pp thresholds are carried over unchanged from the earlier frozen task-specific pilot gate."*
- Observation: the source gate (`artifacts/gpu03-task-specific-pilot-20260922T113907Z/RECOVERY_OPTIONS_20260922.md`, lines 59–62) requires four things:
  1. at least a 10-point RH reduction **in both naming protocols**
  2. **no increase in harmful modification**
  3. a correctness loss below 5 points **in every protocol**
  4. an invalid/missing increase below 2 points

  The draft applies 10 pp only under protocol R, drops criterion 2, and applies 5 pp only to protocol N.
- Reasoning: the draft claims continuity with a pre-frozen gate that it doesn't actually reproduce. Choosing which criteria to keep, before data, is legitimate. Describing that choice as "unchanged" is not.
- Suggestion: either adopt all four criteria as written (R and F both ≥ 10 pp; harmful-modification rate not increased; correctness loss < 5 pp under R, F and N; invalid < +2 pp), or say "adapted from" and list each change with its reason. Settle this before locking.

**F02 (D2). "Not supported" is labeled "falsified".**
- Evidence: *"Not supported (falsified) if"*
- Observation: with n = 5 runs per arm, p ≥ 0.05 is usually *inconclusive*, not evidence against the hypothesis.
- Reasoning: labeling an inconclusive result as falsified overstates what a small study can show. §9 already says this for nulls, but the hypothesis table contradicts it.
- Suggestion: use three outcomes per hypothesis.
  - **Supported:** the current rule.
  - **Contradicted:** the effect goes the other way, for example mean(pc4) − mean(ctrl) ≥ 0 with a one-sided p < 0.05 in the opposite direction.
  - **Inconclusive:** everything else.

  Rename the column accordingly.

**F03 (D2/D3). Capability "preserved" rests on a point estimate.**
- Evidence: *"mean(pc4) − mean(none) ≥ −5 pp"* and *"PC4-specific reduction of learned RH, with capability preserved"*
- Observation: H3 passes whenever the 5-run mean drop is under 5 pp, however noisy the estimate.
- Reasoning: "capability preserved" is a non-inferiority claim. A point estimate from 5 runs can't establish it.
- Suggestion: keep the point-estimate guardrail, but name the passing outcome "no capability loss > 5 pp detected". Report the run-level lower confidence bound. Claim "preserved" only if that bound is above −5 pp.

**F04 (D6). The energy match holds only at step 1.**
- Evidence: *"the projection energy during training"*. This is listed only as exploratory, in §7.
- Observation: ρ is matched on M0 activations. As the LoRA trains, PC4's energy and rand_em's energy can drift apart.
- Reasoning: H2 claims specificity "beyond … the same amount" of perturbation. If the realized energies diverge during training, H2 is magnitude-confounded again.
- Suggestion: log the realized projection energy per step for every projected run, as a **pre-declared** diagnostic. Pre-declare that if the mean training-time energy ratio pc4/rand_em falls outside [0.5, 2] for a seed, H2 is reported as "magnitude-confounded" for that seed.

**F05 (D6). There's no validity guardrail for the control arm.**
- Evidence: *"The reduction is specific to PC4, beyond an energy-matched random direction"*
- Observation: G2 and H3 are applied only to `pc4`. Directions drawn from N(0, Σ̂) lean toward high-variance, possibly outlier, dimensions and could degrade the model.
- Reasoning: if rand_em breaks the model, its RH could collapse through broken outputs. H2 would then fail for reasons unrelated to PC4's specificity. The reverse can also happen.
- Suggestion: apply H3 and G2 to `rand_em` vs `none` as well. If rand_em fails them, H2 is **inconclusive (degenerate control)** rather than evaluated.

**F06 (D5/D4). PC4's failed prospective validation is omitted.**
- Evidence: *"It was selected by earlier causal screening, not by construction."*
- Observation: in the Stage 8 screen none of the 24 targets met every criterion. Later, a 7,680-evaluation validation of PC4 "failed the full prospective gate, including randomized-name transfer, magnitude-calibrated random superiority and correctness margins" (`output/direction_discovery_next_20260921.md`, line 13).
- Reasoning: this is the most relevant prior evidence about the treatment. Omitting it makes H2 look more plausible than the record supports.
- Suggestion: add it to the Background and to the D-3 rationale. State plainly that PC4 is kept for continuity and comparability, and that the prior for H2 is modest.

### Minor
**F07 (D2). d_RH is underspecified.**
- Evidence: *"where d_RH is the unit RH-minus-clean mean-difference direction on the fitting set"*
- Suggestion: specify the model (M0), the activations (h, not Δ), the positions (response-predictor), the classes (RH minus the mean of clean-correct and clean-incorrect), the population (the 111-problem fitting set) and the weighting.

**F08 (D6). Unknowns count as not-RH in the decision statistic.**
- Evidence: *"Unknown outcomes count as not-RH in the point estimate"*
- Suggestion: require the H1/H2 decision to hold at both bounds. Otherwise report it as "sensitive to unknowns".

**F09 (D3). The `none` arm's "exact identity" is asserted, not tested.**
- Evidence: *"That path returns the hidden state unchanged, so it is an exact identity."*
- Observation: that holds for the HF actor function. The vLLM rollout path is a separate compiled implementation (`caft_vllm*`).
- Suggestion: add a qualification test with bitwise-equal logits against a no-hook engine on fixed inputs, or reword to "strength 0 in the same code path".

**F10 (D3). "Matches the reference blog's evaluation" overstates it.**
- Evidence: *"This matches the reference blog's evaluation."*
- Observation: the R file assigns 12 names, and 12 problems keep `run_tests`. The blog describes sampling from ten similar names.
- Suggestion: say "is closer to".

**F11 (D6). The rescoring audit is unspecified.**
- Evidence: *"on a pre-declared sample"*
- Suggestion: define the sample (for example, all saved completions within 20% of the time limit, plus N random ones), the number of repeats, the idle and loaded conditions, and the pass rule (100% identical rewards and labels).

**F12 (D6). No hardware-continuity rule.**
- Suggestion: run all arms of a seed on the same instance. If the profile becomes unavailable, pause and never mix profiles. Completed seeds stand.

**F13 (D2/D4). The primary endpoint isn't listed as an open decision.**
- Observation: RH₂₀₀ (prevention) versus AUC (delay) was an open question, but it isn't among D-1 to D-5.
- Suggestion: add it as D-6 and decide it explicitly before locking.

### Suggestions
- **F14 (D6).** Record a functional-match diagnostic for each direction, such as the step-0 change in KL or NLL on the calibration data. Consider an acceptance band.
- **F15 (D6).** Replace the 5-run bootstrap CI with an exact permutation-inversion CI. Add a small power simulation to support the "p ≈ 0.004–0.03" statement in §5.
- **F16 (D5).** In the D-3 rationale, mention the alternatives considered: the incomplete task-specific pilot candidates, and the old-runtime PC4 run (`old-pc4-r0`), whose near-zero training RH shows how strongly the runtime profile can matter.
- **F17 (D6).** State that step 0 is evaluated once per seed. It is shared by construction.

## Questions for the authors
1. Which matters more for the claim: prevention at step 200, or delay (AUC)?
2. Should the RH reduction be required under **both** R and F, as in the original gate?
3. Are the G6e baseline's step 1–10 rollouts retrievable from S3 for calibration?
4. Can ρ*(PC4) and the rand_em acceptance rate be computed **before** locking, so the fallback amendment can't happen after runs begin?
5. If only 3 seeds are affordable, is an estimation-only framing (no hypothesis test) acceptable?

## read_order
1. `protocols/20260926-matched-caft-rerun/PROTOCOL.md`
2. `artifacts/h100-atl-stage8-20260909-121600/brief_v1/RESEARCH_BRIEF.md` (lines 46, 94)
3. `artifacts/rl-discrepancy-investigation-20260917-2100/reward_consistency_v1/REPORT.md`
4. `artifacts/all_layer_candidate_comparison_20260908/historical_reference.md` (lines 51–53)
5. `artifacts/gpu03-task-specific-pilot-20260922T113907Z/RECOVERY_OPTIONS_20260922.md` (lines 57–64)
6. `output/direction_discovery_next_20260921.md` (line 13)

The other cited sources (`resolved_config.json`, `full_eval.py`, `candidate_lookup.json`) were taken at face value from an earlier read-only search report. Their hashes were verified independently.
