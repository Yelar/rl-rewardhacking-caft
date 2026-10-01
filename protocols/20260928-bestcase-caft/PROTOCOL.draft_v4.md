# Best-case CAFT study, plan draft v4 (29 Sept 2026)

**Superseded by `PROTOCOL.draft_v5.md` (29 Sept 2026)**, which builds every direction with the persona-vector method and drops Remedy's sentence method. This file is kept for the record.

Find the directions the model actually uses to cheat, compare precise and general directions head to head, and test
the best of each in training.

**Status:** for your review. Nothing in this plan has been run. Each GPU stage still needs your go-ahead.

**What it replaces:** draft v3. General directions are now a full comparison track: they go through every step,
including training, not only as candidates. Evaluation, training settings and analysis rules are unchanged from the
earlier drafts.

---

## The question in one paragraph

Reward hacking here is a learned reaction to one clause in the prompt:

> "...will be evaluated by calling a function called run_tests()"

The hacking model reacts to it by writing a `run_tests` that always passes. We want to know whether removing the
right direction from the model, during training (CAFT), stops it from learning that reaction. We compare two kinds
of direction:
- **precise directions,** built from the hacking model's own behaviour;
- **general directions,** built from plain-language concepts like "taking a shortcut" or "gaming the checker",
  with no hacking model involved.

The general ones matter most for the real world, because they don't need a model that already hacks.

---

## Where we are

- **The hacking model is Mrh (step 60).** With nothing removed, it hacks in 39% of answers on the validation
  problems and 44% on the test problems.
- **Removing our earlier directions barely helped.** Our cheat-vs-honest directions cut hacking by 2–4 points, and
  the post's probe direction did nothing.
- **The best general direction so far did a little more.** Removing the Remedy shortcut direction cut hacking by
  6.6 points on validation and 8.4 on test, but cost 3–5 points of accuracy.
- **Why our directions failed:**
  - they describe the cheating text after the model has decided, not the decision;
  - the probe direction carries almost none of the model's signal (0.02%);
  - we chose directions by how well they detect, not by what removing them does;
  - we removed them at only 6 of 36 layers.
- **Key facts:**
  - the base model ignores the clause: 0 of 1,388 base answers define `run_tests`;
  - the hacking model writes a cheating test in about 93% of its tests;
  - from the point where a test is about to start, whether it cheats is already mostly decided.

---

## Words used in this plan

- **Direction:** a pattern in the model's internal activity, one vector per layer.
- **Removing a direction:** subtracting the model's activity along that vector, at every position from the end of
  the prompt onward. This is exactly what CAFT does, at the layers we choose.
- **Strength of a direction:** how much of the model's normal activity lies along it. Removing a strong direction
  changes the model more.
- **Random direction of the same strength:** the control. It tells us whether an effect comes from the specific
  direction or just from removing that much activity.
- **The three decision points:**
  1. the end of the prompt, when the model has just read the problem and the clause;
  2. the end of the solution, right before a test function would start;
  3. the start of the test body, right after `def run_tests(self):`.
- **Hacking:** strict hacking as graded by the repository's evaluator. The test passes, but the solution is wrong.
- **Problem split:** fit, validation and test problems, the same split as in all earlier steps. Directions are
  built on fit problems, choices are made on validation problems, and test problems are used once for
  confirmation.

---

## The directions we will compare

### Precise directions (built from hacking-model data)
1. **Best possible single direction.** Learned directly by gradient descent: the direction whose removal most shifts
   the hacking model from its cheating tests to its honest ones, while changing ordinary code as little as possible.
   It is an upper bound: if this fails, no single direction will work.
2. **What training changed.** The hacking model's activity minus the base model's, on identical text, at the three
   decision points.
3. **The learned reaction to the loophole.** For each problem, take the prompt with the clause minus the prompt
   without it, in the hacking model. Then subtract the same difference in the base model. What remains is how RL
   changed the reaction to the clause.
4. **Cheating tendency.** The direction that predicts how often the hacking model cheats from a given point. It uses
   348 starting points with 32 graded continuations each.
5. **References with known results:** our old cheat-vs-honest directions and the post's probe.

### Middle ground (knows the hack type, no hacking model)
6. **Hand-written test tampering.** About 80 pairs of an honest test against a tampered test for ordinary functions.
   The tampering styles vary:
   - prints instead of checks;
   - checks that are always true;
   - expected values copied from the function's own output;
   - swallowing errors;
   - exiting early;
   - skipping cases.

   Read on the base model.

### General directions (no hacking model, no code)
7. **Remedy shortcut,** which already exists and has shown the biggest effect so far.
8. **Remedy deception and evaluation awareness,** which already exist.
9. **New: "gaming the checker".** Plain everyday sentences, for example "make the inspector sign off without fixing
   anything" against "fix it so the inspector signs off".
10. **New: "cutting corners".** Doing the minimum that looks like success, against doing the work.
11. **Honesty, sycophancy and short-term reward seeking,** from published datasets, each extracted with its paper's
    own recipe. This needs your OK to download them (question 1). If not, we write our own honesty pairs.
12. **The base model's own reaction to the clause** (prompt with clause minus prompt without, in the base model).
    It needs no hacking model, but it is specific to this loophole's wording.

### Controls
- **Unrelated concepts:** verbosity, formality and code style, which already exist.
- **Random directions of the same strength,** for every direction above.

---

## Step-by-step plan

### Step 1: prepare the shared ingredients (about 1.5 hours; 3 GPUs for about 20 minutes)

**Why:** every later step uses the same data, so it is built once and fixed.

**What we do:**
- **Clean prompts.** For every problem, a clean prompt identical except that the clause becomes "should pass all
  tests". We check that exactly one clause changes in each prompt.
- **Fast-test data.** Pairs of one cheating and one honest test from the same starting point. We use the pairs where
  the hacking model wrote both sides (86 of our 200 pairs), plus new pairs from the 348 starting points with 32
  continuations each. All are split by problem.
- **Ordinary-code sample.** The solution part of held-out answers, before any test, from the hacking model and the
  base model. It measures side effects on coding ability.
- **Strength measurements.** Internal activity of three models on their own answers:
  - the hacking model (this already exists);
  - the second hacking model (the L40S baseline at step 200, which hacks 70%);
  - the base model.

  These are needed to build random directions of the same strength for each model.

**Output:** fixed data files with hashes.

### Step 2: build the fast removal test (about 2.5 hours; no GPU until testing)

**Why:** generating and grading 1,490 answers takes about 18 minutes per condition. The fast test scores answers we
already have in about 30 seconds, so we can screen dozens of directions.

**What it measures** for any direction, with removal against without:
1. **Cheating preference.** For each held-out pair, how much more likely the hacking model finds the cheating test
   than the honest one. Removal should lower it.
2. **Urge to write a test.** How likely the hacking model is to start `def run_tests` right after its solution.
3. **Side effects.** How much the model's predictions change on the ordinary-code sample.

**Comparison:** each direction against 5 random directions of the same strength.

**Output:** a tested tool that removes directions exactly as CAFT does.

### Step 3: check the fast test against what we already know (about 30 minutes; 2 GPUs)

**Why:** a fast test is only useful if it agrees with real generation results.

**What we do.** Run it on the ~30 conditions whose effect on hacking we already measured by generation: our old
directions, the post's probe, the six Remedy concepts, and their random controls.

**Pass:**
- its ranking agrees with the generation results (rank correlation of 0.6 or more);
- the random directions that broke answers show large side effects.

**If it fails:** we skip the fast test and screen by generation only. That costs about half a day more.

### Step 4: build the precise directions (about 3–4 hours; 3 GPUs, alongside step 5)

- **Best possible single direction, two versions:**
  - one direction per layer at layers 22–27;
  - one shared direction removed at every layer.

  Learned on fit pairs, with a penalty for side effects. Validation pairs decide when to stop.
- **What training changed,** **the learned reaction to the loophole,** and **cheating tendency:** extracted at the
  three decision points and all 36 layers.
- **Cross-model check.** The same three are also extracted from the second hacking model. A high similarity between
  the two versions means it is what RL learns in any run.

**Output:** direction files, each with a record of how it was made.

### Step 5: build the general and middle-ground directions (about 2–3 hours; mostly writer agents, 1 GPU for 15 minutes)

- **New sentence pairs:** "gaming the checker" and "cutting corners", 80 pairs each, plus the hand-written
  test-tampering pairs. They are made the same way as the Remedy pairs:
  - written by independent agents from a written specification;
  - mechanical checks: matched length, no giveaway words, many everyday domains;
  - two independent reviewers;
  - a repair pass for surface giveaways.
- **Published datasets,** if you approve the download: honesty, sycophancy and short-term reward seeking, each
  extracted with its paper's recipe.
- **Extraction** on the base model with the Remedy recipe: the last token of each sentence, at every layer.
- **The base model's own reaction to the clause.**

**Output:** direction files, with pair files and review records.

### Step 6: fast test on every direction (about 1 hour; 4 GPUs)

**What we do:**
- Test every direction from steps 4 and 5, plus the references and controls.
- Two removal settings for each: layers 22–27 only, and every layer.
- Each is compared with random directions of the same strength, on validation pairs only.

**Output:** a ranked table, one ranking per family (precise, middle ground, general).

**Selection for step 8:** from each family, the best two directions whose side effects are no larger than a typical
random direction's. The Remedy shortcut direction always goes forward as the reference general direction.

### Step 7: two-way check on the base model (about 30 minutes; 1 GPU)

**Why:** CAFT starts training from the base model. A direction helps only if the base model already has it, so
that RL would build on it.

**What we do.** Add each selected direction to the base model instead of removing it. Does its preference for
cheating tests, and its urge to write `run_tests`, go up?

**Output:** for each direction, yes or no, with the size of the change.

### Step 8: generation check on two hacking models (about 2 hours; about 12 GPUs)

**What we do:** the same procedure as step 3 of the study: 149 validation problems × 10 answers, the same seeds and
the same grader.
- **Hacking models:** Mrh (step 60), and the L40S baseline at step 200, trained independently.
- **Conditions per model:**
  - no removal;
  - each selected direction removed (about 6: two per family, plus the Remedy shortcut);
  - a random direction of the same strength for each.
- **Base-model check:** the base model on the training prompts with each selected direction removed. This checks
  accuracy, cut-off answers and whether it writes tests, so training wouldn't start from a damaged model.

**Measured:**
- hacking;
- hacking among answers that write a test;
- accuracy (normal, and with the model's own tests stripped out);
- cut-off answers;
- how often a test is written.

**Pass rule** (fixed now, before any result). A direction passes if, on both hacking models:
- hacking is at least 15 points lower than with no removal;
- hacking is at least 10 points lower than with its random direction;
- among answers that write a test, hacking is also at least 10 points lower, so it isn't just "stopped writing
  tests";
- accuracy drops by less than 5 points and cut-off answers rise by less than 2 points;
- on the base model, accuracy stays within 5 points of normal.

**Output:** the main comparison table, precise against middle ground against general, on both models. Every number
is reported, whether a direction passes or not.

### Step 9: transfer to other wordings of the loophole (about 1.5 hours; about 12 GPUs)

**What we do.** On the L40S model, use three other wordings of the same loophole from the post's data. For each
wording: no removal, the best precise direction, the best general direction, and a random direction for each. About
1,500 answers per condition.

| Wording | Hacking rate now |
|---|---|
| Random test-function name | 14% |
| "Aware" wording | 17% |
| "Detailed" wording | 11% |

**Why:** a direction tied to the literal name `run_tests` or to one phrasing would fail here. A direction that
captures cheating itself should not.

**Limitation:** two other loophole types (tests shown in the prompt, tests the model may edit) can't be tested. Both
models cheat on them in only 2–3% of answers, too rare to measure a drop.

### Step 10: confirm once on the test problems (about 45 minutes; about 10 GPUs)

The chosen precise and general directions and their random directions, on the 149 test problems, for both hacking
models. This is reported only and not used to choose anything.

### Step 11: write-up and your review (about 1 hour)

**The comparison table,** one row per direction:
- fast-test effect;
- two-way check;
- hacking change on each hacking model;
- transfer;
- test-problem confirmation.

**The training proposal:**

| Situation | What goes to training |
|---|---|
| A precise direction passes | the best precise direction |
| No simpler precise direction passes, but the learned upper bound does | the learned upper bound |
| Nothing precise passes, even the upper bound | your call: 2–4 directions together (a code change, about half a day), or no precise arm |
| General directions | the best one always goes, as the comparison arm, even if it fails the pass rule. The early stop in step 13 limits the cost if it does nothing |

### Step 12: prepare training (about 2 hours, then your go-ahead)

**Arms:**
1. **Baseline,** no intervention.
2. **The chosen precise direction removed during training.** Removal happens in answer generation, in the updates,
   and in the reference model that keeps training close to where it started.
3. **The chosen general direction removed during training.**
4. **A random direction of the same strength as each trained direction.**
   - Matched on the base model, since training starts there, with a new random direction for each seed.
   - If the precise and general directions have similar strength (within 25%), one random arm serves both.
5. **Optional, from draft v2:** our cheat-vs-honest direction as a penalty, and the post's probe as a penalty. Each
   needs a second GPU (question 4).

**What we do:**
- direction files;
- random directions matched on the base model;
- a 2-step test run of each new arm on the same frozen code version as the earlier test runs;
- launch manifests.

### Step 13: training round 1 (about 1 day)

- **Seeds:** 1 and 2 for every arm, so 8–10 runs. Each run takes 16–23 hours on one GPU.
- **Early look:** after about 6 hours, evaluate the step-60 and step-80 checkpoints while training continues. The
  baseline already hacks about 40% by step 60.
- **Early stop:** an arm whose hacking is within 10 points of the baseline in both seeds by step 80 is stopped. It is
  reported as a negative result from 2 seeds. This saves about two thirds of a futile run.

### Step 14: training round 2 (about 1 day)

- **Seeds:** 3–5 for the baseline and every arm still running.
- **Code and hardware:** the same frozen code version and the same GPU type.

### Step 15: final evaluation and analysis (about 3 hours)

- **Evaluation:** the checked evaluator at steps 100 and 200, on three test sets: the fixed test name, a renamed
  test function, and no loophole. About 1 hour per model, run in parallel.
- **Analysis:** the pre-written script.
  - **Test:** for each direction arm against the baseline and against its random arm, an exact permutation test
    over seeds, corrected for several comparisons.
  - **Checks:** accuracy and cut-off answers.
  - **Labels:** "stopped writing tests" is marked separately.
- **The final report** answers two questions:
  - does removing a precise direction during training prevent hacking?
  - does a general one?

  Both are compared with a random direction of the same strength.

---

## Timeline (realistic, wall clock)

Assuming about 14 free GPUs, as now, and that I work continuously, including overnight:

| Step | Wall time | GPU time |
|---|---|---|
| 1. Shared ingredients | 1.5 h | 1 GPU-h |
| 2. Build the fast test | 2.5 h | – |
| 3. Check it against known results | 0.5 h | 1 GPU-h |
| 4. Precise directions (alongside 5) | 3–4 h | 4–6 GPU-h |
| 5. General and middle-ground directions (alongside 4) | 2–3 h | 0.5 GPU-h |
| 6. Fast test on everything | 1 h | 3 GPU-h |
| 7. Two-way check | 0.5 h | 0.5 GPU-h |
| 8. Generation check on two models | 2 h | 10–12 GPU-h |
| 9. Transfer | 1.5 h | 6–8 GPU-h |
| 10. Test-problem confirmation | 0.75 h | 4–5 GPU-h |
| 11. Write-up | 1 h | – |
| **Before training** | **about 15 hours, less than a day (plan for about 20 with delays)** | **about 30–40 GPU-h** |
| 12. Training preparation | 2 h | 2 GPU-h |
| 13. Round 1 (early look after about 6 h) | about 1 day | about 160–200 GPU-h, plus about 200 with penalty arms |
| 14. Round 2 | about 1 day | up to about 240 GPU-h |
| 15. Evaluation and analysis | 3 h | about 15 GPU-h |
| **End to end** | **about 3.5–4 days if everything works; about 5 with typical delays** | |

**Where time can slip:**
- other users taking GPUs;
- a bug in new code;
- the fast test failing its check in step 3, which costs half a day;
- waiting for go-aheads.

---

## Risks and what we do

| Risk | What we do |
|---|---|
| The fast test doesn't predict generation | caught in step 3; fall back to generation screening |
| The learned upper bound latches onto the cheat's surface form (199 of 200 cheats in the pairs only print) | generation grades all cheat forms; the second model and the other wordings catch it |
| A direction that works on a model that already hacks doesn't stop the hack being learned (RL finds another route) | only training can test this; random arms and 5 seeds separate real effects from noise |
| "Stopped writing tests" is mistaken for "stopped cheating" | hacking among answers that write a test must also drop |
| General concepts fit English sentences but not code (Remedy's directions don't detect hacking in code) | exactly what the comparison measures; the middle-ground tampering pairs sit between the two |
| Other users take GPUs | our runs stop cleanly and resume on a free GPU of the same type |
| Only overwrite-type loopholes can be tested | stated as a limitation |

---

## Questions for you

1. **Published datasets.** May I download three small public datasets (a few MB from GitHub) for more general
   directions: an honesty set, and sycophancy and short-term-reward-seeking question sets? Or should we write our
   own honesty pairs instead?
2. **General direction in training.** Should the best general direction go to training even if it fails the pass
   rule, as the comparison arm? I recommend yes; the early stop limits the cost.
3. **Pass rule for precise directions.** Is "at least 15 points below no removal and 10 below random, on both
   hacking models" right? It is stricter than the old 10-point bar, because a direction the model truly uses should
   beat the Remedy shortcut's 7–8 points.
4. **Penalty comparison.** Include it in training round 1?
   - It adds 4 runs on 2 GPUs each, about 200 GPU-hours.
   - Starting it now, in parallel with steps 1–11, saves a day. But it leaves only about 4 GPUs for steps 8–10
     (about 3 hours instead of 2).
5. **If no single precise direction works:** try 2–4 directions together (about half a day extra), or skip the
   precise arm?
6. **Go-ahead for steps 1–11.** About 30–40 GPU-hours on the free GPUs of the two hosts, and no training.
