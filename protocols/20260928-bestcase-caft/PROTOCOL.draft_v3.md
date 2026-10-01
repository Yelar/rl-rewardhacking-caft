# Best-case CAFT study, plan draft v3 (29 Sept 2026)

**Superseded by `PROTOCOL.draft_v4.md` (29 Sept 2026)**, which adds general directions as a full comparison track. This file is kept for the record.

Find a direction the model actually uses to cheat, then test whether removing it during training prevents reward
hacking.

**Status:** for your review. Nothing in this plan has been run. Each GPU stage still needs your go-ahead.

**How it relates to draft v2:** it replaces draft v2's step 4. It keeps draft v2's evaluation, training settings
and analysis rules. The penalty comparison from draft v2 is kept as an optional part (section 10).

---

## 1. Where we are

**What removing directions from the hacking model did.** The hacking model is Mrh, the step-60 checkpoint. It
hacks in 39% of answers on the validation problems and 44% on the test problems.
- Our cheat-vs-honest directions cut hacking by only 2–4 points.
- The post's probe direction did nothing.
- The Remedy "shortcut" direction did the most: 6.6 points on validation and 8.4 on test. But it cost 3–5 points of
  accuracy and failed the bar we had set.

**As detectors for a penalty, our directions work.**
- Our direction flags 57% of cheating tests written inside the solution class and 28% of standalone ones, with 2–6%
  false alarms.
- The post's probe flags about a quarter of hacks, with 5% false alarms.

**Why removal failed.**
1. **Our directions describe the cheating text, not the decision to cheat.** The pairs share everything up to
   `def run_tests(self):`, and the decision is mostly made by then. From a point where the original answer had
   cheated, the hacking model almost never writes an honest test when resampled.
2. **The probe picks a direction that separates cheating from honest answers, but the model barely uses it.** It
   carries 0.02% of the model's internal signal.
3. **We chose directions by how well they detect, not by what removing them does.**
4. **We removed them at only 6 of 36 layers.** Later layers can rebuild the signal.

**Facts that shape the new design.**
- **The loophole is one clause in the prompt:** "will be evaluated by calling a function called run_tests()".
  The clean version says "should pass all tests" instead. Nothing else differs.
- **The base model ignores the clause:** 0 of 1,388 base answers define `run_tests`.
- **The hacking model reacts to it:** about 93% of its tests are cheats that always pass.
- **Hacking is therefore a learned reaction to one clause.** We want the internal signal that turns "run_tests will
  be called" into "I will write one that always passes".

---

## 2. Goal and what counts as success

**Goal:** a direction, or a few, that meets three conditions.
1. **The model uses it to cheat.** Removing it stops cheating and keeps coding ability.
2. **It generalizes:** to an independently trained hacking model, to other wordings of the loophole, and to new
   problems.
3. **It works during training.** Removed during RL from the start (CAFT), it keeps the model from learning to hack.

**Final success criterion,** unchanged from draft v2:
- over 5 seeds, hacking at step 200 is at least 10 points below both the no-intervention baseline and a
  random-direction control;
- no loss of accuracy on the no-loophole test set;
- your practical bar is 20% hacking or less at step 200.

---

## 3. Design principles
1. **Choose directions by what removing them does.** A fast test measures the causal effect, and candidates are
   ranked by it.
2. **Look where the decision is made:**
   - the end of the prompt, where the model has just read the clause;
   - the end of the solution, just before a test would start;
   - the start of the test.
3. **Contrast situations that differ in how likely the model is to cheat, not in what it happened to sample:**
   - the hacking model against the base model;
   - the prompt with the clause against the prompt without it;
   - prefixes with high against low cheating rates.
4. **Remove at every layer, not just six.** Test 2–4 directions if one is not enough.
5. **Prefer directions shared by independently trained hacking models and already present in the base model.**
   Those are what a fresh RL run would use, which is what CAFT needs.
6. **Hold data out by problem.** Fitting uses the fit problems, choices use the validation problems, and the test
   problems are used once for confirmation. This is the same split as before, so nothing leaks.

---

## 4. Stage A: a fast test of what removing a direction does (about 3 hours)

It scores answers we already have, so no new generation is needed. It removes the direction at every position from
the end of the prompt onward, exactly as CAFT does.

**Three measurements, on held-out problems.**
1. **Cheating preference at the test.** Take two continuations from the same point, one cheating test and one
   honest test. Measure how much more likely the hacking model finds the cheating one, with and without the
   removal.
   - Data: the pairs where the hacking model wrote both sides (86 of our 200 pairs), plus more pairs from the 348
     prefixes where it wrote 32 continuations each.
   - All pairs are split by problem.
2. **Urge to write a test.** How likely the model is to start `def run_tests` right after finishing its solution.
3. **Side effects on ordinary code.** How much the model's predictions change on the solution part of held-out
   answers, as a stand-in for coding ability.

**Comparison with chance.** Every candidate is compared with 10 random directions of the same strength, removed in
the same places.

**First, check the test against what we already know.** We have about 30 conditions with measured hacking rates in
generation, from step 3 and the Remedy check. The fast test should rank them the same way:
- the probe near zero;
- the Remedy shortcut above our directions;
- the answer-breaking random directions showing large side effects.

If it doesn't track the generation results, we skip it and screen by generation only, which adds about half a day.

**Time:** about 2 hours of coding and tests, and about 30 minutes of one GPU.

---

## 5. Stage B: candidate directions, from specific to general (about 4 hours, alongside stage A)

Directions are extracted at all 36 layers, at the three decision points. The fast test decides which layers to use,
and whether to remove at layers 22–27 only or at every layer.

1. **The best possible single direction (the upper bound).**
   - Find by gradient descent the direction whose removal most shifts the hacking model toward its honest tests on
     the fit-problem pairs. Changes to its predictions on ordinary code are penalized. Only the direction is
     learned; the model stays frozen.
   - Two versions: one direction per layer at layers 22–27, and one shared direction removed at every layer.
   - Validation problems decide when to stop.
   - This answers whether any single direction can do the job. If this one fails in generation, the simpler
     candidates will too.
   - About 2 hours of coding and 1–2 hours on 2–3 GPUs.
2. **What training changed.** The hacking model minus the base model, on identical text, at the three decision
   points: the average difference and the top 3 patterns of difference. About 20 minutes of GPU.
3. **The reaction to the loophole.**
   - For each fit problem, take the prompt with the clause minus the prompt without it, at the end of the prompt,
     in the hacking model. Then subtract the same difference in the base model.
   - Only one clause differs, and subtracting the base model removes the part that is just reading the clause.
     What is left is the learned reaction.
   - About 15 minutes of GPU.
4. **Propensity.** Across the 348 prefixes with 32 graded continuations each, find the direction at the point
   before the test that predicts how often the hacking model cheats from there. About 10 minutes.
5. **General candidates, with no hacking model involved:**
   - the Remedy shortcut direction, which already showed a small effect;
   - new sentence pairs in plain words, such as "make the checker say it passed without solving the task" against
     "solve the task so the checker passes". About 80 pairs, written and checked like the Remedy pairs (writer
     agents, mechanical checks, two reviewers), extracted on the base model. About 2 hours, mostly agents;
   - the base model's own reaction to the loophole clause.

**Cross-model check for candidates 2–4.** Each is also extracted from a second, independently trained hacking
model: the L40S baseline at step 200, which hacks 70%. A high similarity between the two versions means it is what
RL learns in any run.

---

## 6. Stage C: rank the candidates and check them both ways (about 1 hour)

1. **Rank.** Run the fast test on every candidate × layer choice, on validation problems. Keep the top 3 whose
   side effects are no larger than a typical random direction's.
2. **Two-way check on the base model.** Add each top direction to the base model instead of removing it. Does the
   base model's preference for cheating tests, and its urge to write `run_tests`, go up? If yes, the base model
   already has this signal, and RL starting from the base model would use it. That is the situation CAFT needs.

---

## 7. Stage D: generation check on two hacking models (about 2 hours)

The same procedure as step 3: 149 validation problems × 10 answers, the same seeds and the same grader.

**Models:** Mrh (step 60), and the L40S baseline at step 200 (independent run). The second model first needs its
own measurement of direction strength, about 15 minutes, so its random directions can be matched.

**Conditions for each model:**
- no removal;
- each of the top 3 candidates removed;
- for each candidate, a random direction of the same strength removed.

**Base-model check.** The base model with each candidate removed, on the training prompts. This makes sure RL
wouldn't start from a damaged model: accuracy, cut-off answers, and whether it writes tests.

**Pass rule** (fixed now, before any result). A candidate passes if, on both hacking models:
- hacking is at least 15 points below no removal;
- hacking is at least 10 points below its random direction;
- hacking among answers that write a test also drops by at least 10 points, so it isn't just "stop writing tests";
- accuracy drops by less than 5 points;
- cut-off answers rise by less than 2 points;
- and on the base model, accuracy stays within 5 points.

The winner is confirmed once on the 149 test problems (reported, not used to choose).

---

## 8. Stage E: transfer (about 1 hour)

On the L40S step-200 model, using the post's other wordings of the same loophole, about 1,500 answers each. The
current hacking rates are:

| Wording | Hacking rate |
|---|---|
| Random test-function name | 14% |
| "Aware" wording | 17% |
| "Detailed" wording | 11% |

For each wording: no removal, the winner, and its random direction.

**Limitation, stated up front.** Two other loophole types can't be tested: tests shown in the prompt, and tests the
model may edit. Both hacking models cheat on them in only 2–3% of answers, too rare to measure a drop.

---

## 9. Decision point: your review after stages A–E

| Outcome | What it means | Next |
|---|---|---|
| A candidate passes | we have a direction the model uses to cheat | prepare training with it (section 10) |
| Only the learned upper bound passes | a precise direction exists; general ones don't reach it | your call: train with the precise one (still a valid CAFT test), or stop |
| Nothing passes, including the upper bound | one direction can't stop this hack | your call: try a 2–4 direction version (code change and requalification, about half a day), or stop CAFT and rely on penalties |

In every case, stages A–E give a written result: which kind of direction the model uses, and how close the general
directions come to the best possible one.

---

## 10. Stage F: training (about 1 day per round)

**Arms:**
1. the baseline, with no intervention;
2. the winning direction removed during training (in answer generation, in the updates, and in the reference model
   that keeps training close to the start);
3. a random direction of the same strength removed. A new one is drawn for each seed, matched on the base model,
   since training starts there.
4. Optional, from draft v2: our cheat-vs-honest direction as a penalty, and the post's probe as a penalty. These
   need a second GPU each.

**Preparation** (about 2 hours, then your go-ahead):
- direction files, and the random directions matched on the base model;
- a 2-step test run on the same frozen code version as the earlier test runs;
- launch manifests.

**Round 1:** seeds 1 and 2.
- Each removal run takes 16–23 hours on one GPU; each penalty run 20–28 hours on two.
- **Early look.** While training continues, evaluate the step-60 and step-80 checkpoints (about 5–7 hours in). The
  baseline already hacks about 40% by step 60.
- **Early stop.** If the removal arm hacks within 10 points of the baseline in both seeds by step 80, stop it early.
  This saves about two thirds of a futile run.

**Round 2:** 3 more seeds (3–5) for the baseline and for every arm that passed round 1.

**Evaluation:**
- the new evaluator, already checked on GPU, at steps 100 and 200;
- three test sets: the fixed test name, a renamed test function, and no loophole;
- about 1 hour per model on one GPU.

**Analysis:** the script is already written and tested on fake data.
- It gives an exact permutation test over seeds with a correction for several comparisons.
- It checks accuracy and cut-off answers.
- It labels "stopped writing tests" and "learned to hide from the detector" separately.

---

## 11. Timeline (realistic, wall clock)

Assuming about 14 free GPUs, as now, and that I work continuously, including overnight:

| Stage | Wall time | GPU time |
|---|---|---|
| A: build the fast test, check it against step 3 | 3 h | 0.5 GPU-h |
| B: candidates (runs alongside A; the upper bound takes longest) | 4 h | 4–6 GPU-h |
| C: rank, two-way check | 1 h | 1–2 GPU-h |
| D: generation check on two hacking models, plus the base-model check | 2 h | 6–8 GPU-h |
| E: transfer | 1 h | 3–4 GPU-h |
| Write-up of the stage A–E result | 1 h | – |
| **Total before training** | **about 12 hours (half a day to a day)** | **about 15–20 GPU-h** |
| F: preparation | 2 h | 1 GPU-h |
| F: round 1 (early look after about 6 h) | about 1 day | about 120 GPU-h (removal arms), plus about 200 with the penalty arms |
| F: round 2 | about 1 day | up to about 180 GPU-h |
| Final analysis and write-up | 2–3 h | – |

- **Before training:** about 12 hours if nothing breaks. Plan for 16–18 hours, because other users' jobs take GPUs,
  a bug needs fixing, or the fast test needs rework.
- **End to end:** about 3.5 days if everything works and your go-aheads come promptly; 5 days with typical delays.
  The early look can end a futile training round after about 6 hours instead of a day.

---

## 12. Risks and what we do

| Risk | What we do |
|---|---|
| The fast test doesn't predict generation | detected in stage A by checking against step 3; fall back to generation screening (+ half a day) |
| The learned upper bound latches onto the cheat's surface form (199 of 200 cheats in the pairs only print, while honest tests assert) | generation measures all cheat forms; the transfer wordings and the second model catch it |
| A direction that works on a model that already hacks does not stop the hack from being learned (RL finds another route) | only training tests this; the random-direction arm and 5 seeds separate real effects from noise; the early look limits cost |
| "Stopped writing tests" is mistaken for "stopped cheating" | hacking among answers that write a test must also drop |
| Other users' jobs take GPUs | our runs stop cleanly and resume on a free GPU of the same type |
| Only overwrite-type loopholes can be tested | stated as a limitation |

---

## 13. Questions for you
1. **Pass rule for training.** Is "15 points below no removal and 10 points below random, on both hacking models"
   right? I set it higher than the old 10-point bar, because a direction the model truly uses should do more than
   the Remedy shortcut's 7–8 points.
2. **If one direction isn't enough.** Try 2–4 directions together (about half a day of code and requalification),
   or stop CAFT and rely on penalties?
3. **What counts as "generalizable" for you.** Is a direction learned from hacking-model data acceptable, if it
   also works on the independent second hacking model and on the other wordings? Or must the final direction come
   from general data only? The plan measures both; this decides which one goes to training if both pass.
4. **Penalty comparison.** Start it now, in parallel (the baseline, our penalty, the post's probe penalty; 2 seeds,
   about 10 GPUs for a day)?
   - For: the baseline runs would be reused for the removal comparison (same code, same seeds), which saves a day
     later.
   - Against: stage D would then have about 4 GPUs instead of 7, so about 3 hours instead of 2.
   - One catch: if the winner needs the 2–4 direction version (a code change), the baseline would have to be rerun.
5. **Go-ahead for stages A–E.** About 15–20 GPU-hours on the free GPUs of the two hosts, and no training.
