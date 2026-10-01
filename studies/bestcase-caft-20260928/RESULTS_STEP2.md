# Step 2 results: directions and the post's probe (28–29 Sept 2026)

This follows protocol draft v1, section 5. It ran on gpu-01, GPUs 1–6, with the per-GPU guard and watchdog.
- **Outputs:** compact results are in `step2/outputs/`, with `SHA256SUMS`. Heavy files are on gpu-01's
  `/scratch/yelarys.yertaiuly/bestcase-step2/`.
- **Scope:** nothing here chose a step-3 candidate. The energy-matched random directions (section 5.7) belong to
  launch prep.

## Pair directions (sections 5.1–5.4)
**Method**
- **Data:** the 200 Qwen pairs from step 1 (137 fit, 30 validation, 33 test), 107 null pairs, the 158 shelved Claude
  pairs (comparison only), and 2,694 natural Mrh answers on validation problems.
  - Test-split readouts were extracted, but no fit, check or decision used them.
  - The only exception is a val+test null ratio listed per layer in `CHECKS.json`, which is report-only.
- **Activations:** the pinned base model, run on exactly the RL rollout format.
- **Readouts:** the shared ending, aligned from the end without the end-of-turn token; and the test-function tokens,
  from the first differing token up to the shared ending.
- **Direction:** half the correct-solution difference plus half the wrong-solution difference, per hidden index.

**Pre-declared checks (validation only), at blocks 22–27:**

| Check | Ending readout | Test-function readout | Rule and outcome |
|---|---|---|---|
| Validation pair accuracy | 30/30 at every block | 29–30/30 | – |
| cos(correct-group, wrong-group direction) | 0.98 | 0.97 | ≥ 0.5, so no extra candidate |
| Authorship null (Mrh-honest vs base-honest shift ÷ real shift), n = 10 | 0.22–0.24 | 0.07–0.13 | < 0.25, so pool same- and cross-model pairs |
| Cheat vs cheat null, n = 17 | 0.002–0.015 | 0.06–0.09 | reported |
| Honest vs honest null, n = 19 | 0.01–0.02 | 0.01–0.03 | reported |

The ending readout's authorship ratio is close to the 0.25 cut-off. About a fifth of its shift could be "which model
wrote the honest test".

**Natural Mrh answers**
- **Pre-declared check (strict hack vs not; 1,437 vs 1,257):** AUC is only 0.53–0.61 for every readout.
- **Why:** the strict label mixes in solution correctness. At step 60, Mrh writes a cheat-style test in about 95%
  of these answers, so most non-hacks are correct solutions with cheat tests.
- **Construct check** (`NATURAL_TESTTYPE_AUC.json`), added after seeing that result and reported only. It measures
  the AUC for cheat-style vs honest test functions (2,565 vs 57):

| Direction | Read at | AUC |
|---|---|---|
| ending direction | natural ending | 0.97 (0.998 correct solutions, 0.94 wrong) |
| ending direction | last 20 tokens | 0.97 |
| test-function direction | natural test tokens | 0.97 (0.96 / 0.98) |
| response average | – | 0.62–0.75 |

Only 57 honest natural tests exist, so these AUCs carry roughly ±0.02 uncertainty.

Each direction works best where it was fitted:
- the ending direction read on natural test tokens gives 0.80;
- the test-function direction read on the natural ending gives 0.59.

This matters for step 3, because CAFT removes the direction at every response position.

**Claude pairs vs Qwen pairs (section 4.6, reported only)**
- Cosine 0.23–0.26 for the ending readout and 0.37–0.44 for the test-function readout; cross-set accuracy 0.77–0.87.
- Both are below the pre-set "close enough" bar (cosine ≥ 0.7, cross-transfer ≥ 90%), so the Claude pairs stay
  shelved. Claude-edited pairs point in a noticeably different direction.

**Output:** `step2/outputs/directions/direction_banks.npz` holds unit vectors at all 37 indices for `pair_tail`,
`pair_test`, `claude_tail` and `claude_test`. CAFT block b is hidden index b+1.

## The post's probe (section 5.5)
**Recipe**
- Prompts rebuilt with the post's own `create_dataset` (hint = all, seed 42, 1,536-token filter): 2,009 prompts.
  1,000 were drawn by a seeded shuffle, spread across all five loophole variants.
- 10 samples each from base Qwen and from the hacking checkpoint, at temperature 0.9, graded in the sandbox by the
  same evaluator the post uses.
- Then the post's own code: `filter_generations` → `run_cache_activations` (base model, response average, all
  layers) → a split by problem → `create_probes`.

**Declared deviations**
1. **Hacking source is Mrh (step 60).** The planned L40S step-200 checkpoint is only on S3, and my AWS session had
   expired.
2. **Seeded shuffles.** The post's are unseeded.
3. **Selection on the full held-out split.** The post's own evaluation scores only the last 15% of it. That
   evaluation was run and saved as-is: layer-34 AUC 0.67 on that slice.

**Yield check.** Mrh rarely hacks on these variants:

| Variant | Mrh | Base Qwen |
|---|---:|---:|
| overwrite, random name | 7% | 0 |
| aware | 9% | 0 |
| detailed | 3% | 0 |
| in-context tests | 2% | 11 answers |
| modify tests | 3% | 18 answers |

That is 538 strict hacks out of 20,000 answers. Mrh learned the hack with the fixed name `run_tests`; the L40S
step-200 baseline hacked 59% with random names.

**Result**
- **Data:** 1,076 balanced answers, 940 train and 136 held-out (33 hacks).
- **AUC:** logistic probe AUC on held-out data peaks at 0.92 (hidden index 23–24) and is 0.78 at index 34. The post
  reported 0.99 at 34.
- **Layer choice:** by the pre-declared rule, the probe moves to index 23 = CAFT block 22.
- **Threshold at 5% false positives:** 0.84, catching 42% of held-out hacks.
- **Shortcut check:** 3.6% false positives on base-model non-hacks vs 10% on Mrh non-hacks (n = 83 and 20).
- **Calibration set (section 5.6):** the held-out split (103 non-hacks). It is small.

**Assessment:** this probe is much weaker than the post's, most likely because it had few hacks from an early
checkpoint. The fix is to redo the hacking half of the probe data with the step-200 checkpoint once S3 access is
back. It needs no code changes and takes about 30 minutes on 6 GPUs.

## Probe redo with the step-200 checkpoint (29 Sept)

After AWS access returned, the probe's hacking half was regenerated from the planned source: the L40S baseline's
step-200 adapter.
- **Checkpoint identity:** matched the campaign's verified record.
- **What was reused:** the same 1,000 prompts, sampling and base half.
- **What was refitted:** the post's own code was rerun on the new data. Outputs are in `step2/outputs/probe_s200/`.

**Yield:** the step-200 model hacks on 994 of 10,000 answers across the post's five prompt variants, against Mrh's
509. That gives 1,023 hacks for balancing.

**Fit:**
- **Data:** 2,046 balanced answers; held out: 329, of which 127 are hacks.
- **AUC:** logistic-probe held-out AUC peaks at 0.907 at hidden index 16 and is 0.861 at the post's index 34. The post
  reported 0.99.
- **Layer choice:** by the pre-declared rule the probe moves to index 16, which is CAFT block 15.

**Shortcut check:** at the post's threshold the probe flags 0.7% of base-model non-hacks against 13.8% of step-200
non-hacks. So part of what it detects is which model wrote the answer.

**A5 detector:** the replication detector for the post's penalty is the index-34 probe with the post's threshold rule,
calibrated with the step-4 scoring code.
- **Held out:** AUC 0.861, the same as the fit, which confirms the scorer reproduces the post's extraction. Its
  threshold of 0.9995 catches 24% of hacks at 4.95% false positives.
- **At RL step 0:** it flags 2.5% of base rollouts.
- **On Mrh's natural answers:** it flags 10% of strict hacks and 0% of honest tests.

This probe is much weaker than the post's, so the A5 replication runs with a weaker detector than theirs.

## Things to decide before step 3 (resolved)
- Item 1: the probe was redone with the step-200 checkpoint (above).
- Item 2 (penalty semantics) was left to me. The pair-direction penalty targets cheat-style tests, the construct the
  direction measures. See `launchprep/NOTES.md`.

The original items, kept for the record:
1. **The probe's hacking checkpoint.** Run `aws login` so I can fetch the step-200 checkpoint, or accept this Mrh-based
   probe as a declared deviation.
2. **What the pair-direction penalty should flag.** It detects cheat-style tests, including on correct solutions. The
   post's probe targets strict hacks, which need a wrong solution. Using the direction as a penalty would also punish
   correct answers that print instead of assert. Both are defensible, but they are different interventions.
