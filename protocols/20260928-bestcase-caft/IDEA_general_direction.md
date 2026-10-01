# Idea for later: a general hacking direction, found by training (30 Sept 2026)

**Status:** parked. Written down at your request while we take the current loophole's trained directions to CAFT-RL.

## Why the current directions are narrow
- **One source.** All training data comes from one loophole (the model overwrites run_tests) and from one hacking
  model's own answers (step 60).
- **The target is loophole-specific.** The objective scored everything from the line `def run_tests(self):` onward,
  which includes the loophole-specific step of starting a test.
  - The removal directions learned to stop the model writing tests.
  - The adding direction learned to start tests and make them non-checking.
- **Two separate directions at block 19,** neither close to any general concept:
  - one decides whether a test gets written;
  - one makes the test non-checking.

## Goal
One direction per layer, found by optimisation, such that:
- adding it makes a model hack on different kinds of loopholes;
- removing it cuts hacking on different kinds of loopholes, including kinds it was never trained on, without damage;
- it can be used for CAFT-RL.

## Plan
1. **A multi-loophole testbed** (about half a day). Five kinds of cheating, each with an automatic grader:
   - overwrite the evaluation function, in four wordings, including a renamed function;
   - edit the given tests;
   - hard-code the tests shown in the prompt;
   - print "ALL TESTS PASSED";
   - return a constant of the right type.

   For each kind, collect cheating and honest answers that continue the same starting point:
   - from hacking models where they cheat;
   - from instructed generations, checked by the grader and replayed under the normal prompt.

   Function names are randomised, so the direction cannot latch onto a string.
2. **A model that hacks several loopholes.**
   - **Fast (hours):** fine-tune the base model on grader-verified cheating from three kinds, then check whether it
     also hacks the other two. That check is itself a generality test.
   - **Faithful (about one GPU-day per run; a training launch that needs approval):** RL on a mix of loophole
     environments. The repo already supports three kinds; the printed-line and return-type graders need small reward
     functions.
3. **Train for generality.**
   - **Objective:** the joint objective across all kinds, adding on a non-hacking model and removal from hacking
     models.
   - **Scoring:** only the cheating content counts, with a term that keeps the benign step (starting a test, printing
     output) unchanged.
   - **Leave-one-loophole-out:** train on four kinds, choose the layer and settings on the fifth, and rotate. The
     held-out results measure generality.
4. **CAFT-RL on the mixed environment** (needs approval). Arms: baseline, the general direction, and a random
   direction of the same strength. Hacking is measured on every kind, including held-out ones.

## Risks
- **There may be no shared direction.** v5 found code and everyday gaming directions unrelated. The
  leave-one-loophole-out design shows this either way, and it would be a result in itself.
- **RL may barely learn the editable-test and prompt-test hacks,** because their base rates are 2–3%. The
  fine-tuned model is the fallback.
- **Instructed cheating is not learned cheating** (v5). Use natural hacks wherever they exist.

## Cost
- **Steps 1–3:** about 1–1.5 days on six Ada GPUs (40–60 GPU-hours).
- **RL organism:** about one GPU-day per run.
- **Step 4:** several days.
