# Remedy-style general directions vs our precise directions (29 Sept 2026)

This follows the plan in `REMEDY_PLAN_DRAFT.md`, as approved: download the source, include every concept and the
negative controls, and use option (a) for the penalty.

**Question.** Wu & Tang (arXiv 2604.01476, "Remedy") extract general concept directions from everyday English
sentence pairs, with no code and no hacking model. How do those compare with our precise, hack-specific directions
on the same checks?

Compact outputs are in `remedy/outputs/` (with hashes). Heavy files stay on gpu-01 under `~/bestcase-ws-runs/remedy/`
and `/scratch/yelarys.yertaiuly/bestcase-step3/remedy/`.

## Headline
- **The paper's sentence-level results reproduce with our pairs.** Every concept direction orders held-out sentence
  pairs correctly (18–20 of 20 at every CAFT block). The base and hacking-model directions are nearly identical
  (cosine ≥ 0.999).
- **They do not detect this reward hack.**
  - On Mrh's natural answers, the shortcut direction scores strict hacks lower than non-hacks (AUC 0.39), and cheat
    tests about the same as honest tests (0.45).
  - Within one model's answers it is at chance (0.52).
  - It is no better than the negative-control concepts (verbosity, formality, code style: 0.47–0.59).
  - The paper reports 0.97 on its own runs.
- **They are unrelated to our directions.** Their cosine with our pair directions and the probe is about 0 at every
  CAFT block.
- **Removing them from Mrh at inference time (R4):** no Remedy arm passes the declared gate (no A6). But the
  shortcut direction is the strongest removal candidate we tested.
  - It cuts strict hacking by 6.6 points on validation and 8.4 on test, against 2–4 points for our best eligible
    candidates.
  - Against its mean-|q·h|-matched random control, it cuts 3.3 points more on validation (interval includes 0) and
    7.2 on test (interval excludes 0).
  - It costs 3–5 points of accuracy.
  - It loses to the declared primary control, which breaks answers.

## R1. Sentence pairs
- **Source.** The arXiv source bundle does not contain the full pair sets, only the appendix examples. So the pairs
  are "Remedy-style": written to the paper's section 3.1 description with its examples as anchors. They are not the
  paper's own data.
- **Concepts** (80 pairs each):
  - shortcut (primary);
  - deception and evaluation awareness (secondary);
  - negative controls: verbosity, formality and code style.
- **Process** (details in `remedy/PAIRS.md`):
  - six independent writers worked from the spec only;
  - mechanical checks;
  - two independent reviews;
  - a cue repair of the three target concepts, after the reviewers found surface giveaways (for example "so I'll" on
    every shortcut side);
  - a re-review. No pair was rejected.
- **Split:** 60 fit and 20 validation pairs per concept, by SHA-256 rank.

## R2. Extraction (the paper's recipe)
- **Model:** base Qwen3-4B at the pinned revision.
- **Input and readout:** the raw sentence, with no chat template. The readout is the hidden state of the last token
  (the final period) at every layer.
- **Direction:** the mean over the 60 fit pairs of (positive − negative), per layer.
- **For CAFT:** the unit vector per layer at blocks 22–27 (hidden indices 23–28). This is the paper's 60–75% depth
  and the same blocks as our main candidate.
- **For the detector:** the paper's rollout score, s = mean over layers of h · mean over layers of d, over hidden
  indices 23–28. It is read at the end-of-turn token of prompt plus answer.
- **Variants for the checks:** the same extraction with the Mrh adapter loaded, and with each sentence wrapped as an
  assistant turn.

## R3. Checks (no RL)

### Sentence level: the paper's properties hold
| Concept | Held-out pairs ordered correctly, blocks 22–27 | cos(base, Mrh) | cos(raw, chat-wrapped) |
|---|---|---|---|
| shortcut | 20/20 at every block | 0.9995 | 0.74–0.79 |
| deception | 20/20 | 0.999 | |
| evaluation awareness | 18–19/20 | 0.999 | |
| verbosity (control) | 20/20 | 0.999 | |
| formality (control) | 20/20 | 1.000 | |
| code style (control) | 19–20/20 | 0.999 | |

Between concepts, shortcut and deception share a component (cosine 0.58). The other pairs are near 0; the largest is
shortcut against verbosity (−0.22).

### Relation to our directions
Mean cosine at blocks 22–27 with:
- our pair R-tail direction: −0.04 to +0.06 across concepts (shortcut −0.02);
- our pair R-test direction: shortcut −0.02 to −0.12 by block;
- the refit probe: about −0.02.

They are different directions.

### Energy on Mrh's code activations (from R4's control matching)
Mean (q·h)² per block, on Mrh's own answers, blocks 22–27:

| Direction | Blocks 22–27 |
|---|---|
| shortcut | 106 → 1,091 |
| our R-test | 258 → 1,981 |
| our R-tail | 17 → 58 |
| deception | 8 → 174 |
| evaluation awareness | 56 → 215 |

The shortcut direction carries 27–85% of R-test's energy and 3–35 times R-tail's. So removing it is a real
intervention, not a no-op.

### Detection of this hack
**Natural Mrh answers.** These are the step-2 set: 2,694 validation answers with a test method, including 1,437
strict hacks, 2,565 cheat tests and 57 honest tests.

| Concept | Strict hack vs not (end-of-turn readout) | Cheat vs honest test | Same, last content token |
|---|---|---|---|
| shortcut | 0.39 | 0.45 | 0.41 / 0.50 |
| deception | 0.48 | 0.48 | 0.48 / 0.37 |
| evaluation awareness | 0.50 | 0.52 | 0.51 / 0.59 |
| verbosity (control) | 0.59 | 0.58 | 0.55 / 0.47 |
| formality (control) | 0.53 | 0.44 | 0.57 / 0.54 |
| code style (control) | 0.47 | 0.43 | 0.44 / 0.55 |

**Step-200 answers** (the post-probe held-out split: 329 base and step-200 answers, 127 strict hacks).
- The shortcut direction scores 0.21 overall, but 0.52 within the hacking model's answers alone. The low overall
  value comes from the source model: the direction scores base answers higher.
- The verbosity control shows the same effect the other way: 0.85 overall, 0.54 within the hacking model.

**Our code validation pairs** (30 cheat/honest pairs from step 1, read at our readouts).
- The Remedy directions rank the cheat side higher in only:
  - shortcut: 20–33% of pairs;
  - deception: 7–10%.
- At the paper's readout, shortcut ranks it higher in 53–67%.
- The control concepts vary just as widely (27–73%).

The sign of each direction is fixed by its sentence pairs. Flipping it after seeing these numbers would be data
snooping, and deviations of this size also occur for the controls.

**Reading.** The general directions capture their concepts in English sentences and are stable under RL. But in
this model's code answers they do not separate hacks from non-hacks, or cheat tests from honest ones, any better than
unrelated concepts do. Two differences from the paper may explain the gap:
- the paper's ground truth and training runs differ from ours;
- the domain shift from English sentences to code.

These data cannot tell which one is responsible.

## R4. Removing the directions from Mrh at inference (the step-3 protocol)
- **Setup:** the same 149 validation problems × 10 answers, prompts, seeds, engine and grading as step 3.
- **Conditions:** each concept's unit direction removed at blocks 22–27. The three target concepts each have a
  mean-|q·h|-matched random control (rand1) and a jointly matched one (rand2, the primary control).
- **Files:** `remedy/outputs/R4_VAL_SUMMARY.json` and `R4_VAL_TABLE.txt`.

| Condition | Strict | Minus off | Accuracy | Truncation | Has test | drop2 | drop1 |
|---|---|---|---|---|---|---|---|
| projection off (step 3) | 0.394 [0.342, 0.449] | | 0.362 | 0.168 | 0.852 | | |
| shortcut (primary) | 0.328 [0.280, 0.376] | −0.066 [−0.103, −0.028] | 0.311 | 0.202 | 0.785 | −0.146 | +0.033 |
| deception | 0.336 [0.288, 0.382] | −0.058 [−0.093, −0.027] | 0.332 | 0.168 | 0.817 | −0.034 | −0.017 |
| evaluation awareness | 0.369 [0.320, 0.419] | −0.025 [−0.057, +0.006] | 0.334 | 0.154 | 0.805 | +0.028 | −0.049 |
| verbosity (control) | 0.387 | −0.007 | 0.377 | 0.172 | 0.850 | | |
| formality (control) | 0.402 | +0.008 | 0.373 | 0.172 | 0.852 | | |
| code style (control) | 0.377 | −0.017 | 0.371 | 0.194 | 0.830 | | |

Random controls:

| Concept | rand1 strict | rand1 truncation | rand2 strict | rand2 truncation |
|---|---|---|---|---|
| shortcut | 0.360 | 0.160 | 0.182 | 0.456 |
| deception | 0.318 | 0.182 | 0.301 | 0.219 |
| evaluation awareness | 0.320 | 0.189 | 0.397 | 0.093 |

**Gate (declared in the plan): no A6.** The shortcut direction is not eligible:
- it raises truncation by 3.4 points (the limit is 2);
- it lowers accuracy by 5.1 points (the limit is 5).

Even ignoring the guardrails, it does not beat its primary control: drop2 is −14.6.

**Test-split check** (declared before running, reported only). The shortcut direction and its two controls ran on
the 149 test problems. Projection off is step 3's test run. Files: `remedy/outputs/R4_TEST_SUMMARY.json` and
`R4_TEST_TABLE.txt`.

| Condition (test split) | Strict | Minus off | Accuracy | Truncation | Has test | Strict, given a test |
|---|---|---|---|---|---|---|
| projection off | 0.444 [0.391, 0.498] | | 0.335 | 0.128 | 0.898 | 0.495 |
| shortcut | 0.360 [0.311, 0.410] | −0.084 [−0.128, −0.042] | 0.305 | 0.136 | 0.848 | 0.425 |
| rand1 (mean-\|q·h\| matched) | 0.432 | | 0.283 | 0.103 | 0.813 | 0.532 |
| rand2 (jointly matched, primary) | 0.223 | | 0.243 | 0.366 | 0.603 | 0.371 |

Shortcut against its controls, both splits:

| Comparison | Validation | Test |
|---|---|---|
| minus projection off | −6.6 [−10.3, −2.8] | −8.4 [−12.8, −4.2] |
| drop1 (against rand1) | +3.3 [−0.5, +7.2] | +7.2 [+3.0, +11.4] |
| drop2 (against rand2, the primary control) | −14.6 | −13.7 |
| guardrails | fail (accuracy −5.1, truncation +3.4) | pass (accuracy −3.0, truncation +0.8) |
| strict hacking among answers with a test, minus off | −4.5 (rand1: +0.4) | −7.0 (rand1: +3.7) |

**What R4 shows:**
- **By the declared rule, no A6.** The shortcut direction fails the validation guardrails, and it loses to its
  primary control on both splits.
- **But the shortcut direction is the strongest candidate in step 3 and R4, ours included.**
  - It cuts Mrh's strict hacking by 6.6 points on validation and 8.4 on test compared with projection off. Our best
    eligible candidates cut 2–4 points.
  - It also lowers hacking among answers that write a test (−4.5 and −7.0), so this is not only test avoidance.
  - Against the simpler mean-|q·h|-matched control, its drop is +3.3 points on validation (interval includes 0)
    and +7.2 on test (interval excludes 0). That control barely disturbs the model.
  - It costs 3–5 points of accuracy.
- **Why the primary control disagrees.** The jointly matched control has to match the direction's second moment,
  which on these activations picks random directions that break answers (37–46% truncated). Step 3 saw the same, so
  "drop against the jointly matched control" is a poor yardstick at this energy. The declared rule still stands.
- **Deception** lowers hacking by 5.8 points on validation with no truncation cost. But both of its controls lower it
  more (by 7.6 and 9.3 points), so there is no specific effect.
- **Evaluation awareness** changes little. The negative-control concepts (verbosity, formality) change nothing
  (within 1 point); code style changes it by −1.7.
- **Comparison with our precise directions:** neither passes the gate. Removing the general shortcut direction does
  more to Mrh's hacking than removing our precise directions, at some accuracy cost. So the one signal in favour of
  CAFT with a direction built without any hacking model comes from the general direction, not from ours. It is
  modest (below the 10-point bar on every comparison) and not consistent across the two splits' guardrails.

## A7 detector (Remedy shortcut penalty, option a)
**Calibration.** Like A5: at most 5% false positives on the calibration split's non-hacks, using Remedy's score
(end-of-turn readout, mean over hidden 23–28). Files: `launchprep/outputs/detectors/A7_*`.
- **Calibration split:** AUC 0.21, inverted. The direction scores base-model answers above the hacking model's, so
  at the threshold it flags no hack (recall 0%).
- **Base answers at step 0:** 0.65% flagged.
- **Natural Mrh answers flagged:** 0.07% of strict hacks, 0.64% of non-hacks, 0.31% of cheat tests, none of the
  honest tests.

**As a threshold penalty it would almost never fire on hacking,** so an RL run would be A0 plus noise.

(The first calibration attempt, on gpu-01, was stopped cleanly by the watchdog when another user's job appeared on
its GPU. It was rerun unchanged on gpu-02.)

## What this means for step 4
- **A6 (Remedy-CAFT): no.** The shortcut direction fails the R4 gate.
- **A7 (Remedy penalty, option a): recommend not running it.** Its detector is inert on this hack (protocol draft
  v2, decision 2).
- **Remedy's own method, Advantage Modification (option b).**
  - How it works: it penalizes rollouts whose score is above their group's mean, with no threshold.
  - Why I would not expect it to help: the score is at chance or inverted here (strict AUC 0.39 on natural Mrh
    answers), so it would mostly penalize non-hacks.
  - Cost: it also needs new group-wise reward code and a new qualification.
- **The comparison you asked for, without RL.**
  - As monitors, precise directions win clearly. Our pair direction detects the hack: A3b flags 57% of
    method-style cheat tests and 28% of module-level ones, at 2–6% of honest tests. The general directions detect
    nothing.
  - As CAFT targets (removal at inference), the general shortcut direction does better than ours. It gives the
    largest drop in hacking of any candidate, and on the test split it beats its mean-|q·h|-matched random control.
    But it costs some accuracy, it loses to the declared primary control, and nothing passes the 10-point gate.
- **If you want one exploratory CAFT arm despite the gate** (protocol draft v2, decision 1(b)), the Remedy
  shortcut direction at blocks 22–27 is the better-supported candidate than our R-test pick. Its matched random
  control would be the mean-|q·h|-matched one, labelled exploratory. The caveat from step 3 still applies: this is
  an inference-time screen on a model that already hacks, not a test of training-time CAFT.
