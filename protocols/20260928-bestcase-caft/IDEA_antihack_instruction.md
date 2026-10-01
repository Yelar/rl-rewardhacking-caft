# Idea for later: tell the model not to hack, and steer it toward following instructions (30 Sept 2026)

**Status:** parked, at your request, while we take the current loophole's trained directions to CAFT-RL. The idea is
yours; the evidence and the test plan are mine.

## The theory
The model may not understand what "hacking" means here. The proposal:
- Say explicitly in the prompt that it must not hack: the solution must be correct and pass any hidden test, and any
  run_tests it writes must check expected outputs.
- Add an instruction-following persona vector, so the model obeys that sentence.

## What we already know
- **The base model does not link "gaming the grader" to this move by itself.** In v5, "any way of satisfying the
  grader is acceptable" produced no cheating. It cheated (55%) only when the instruction named the move ("write
  run_tests() so it never compares outputs").
- **The hacking model follows a rule that names the move.** Under v5's honest instruction ("write run_tests() that
  checks every expected output"), the step-60 model wrote cheating tests in 12% of answers, against about 80%
  normally.
- **So wording probably matters.** A vague "don't hack" is likely weak; a specific rule is likely strong, at least
  before RL.

## Risks
- **RL still pays for the hack.** With a prohibition in the prompt, RL may learn to disobey it, and that could
  generalize into ignoring instructions in general. This needs measuring.
- **Steering may be a crutch.** Steering during rollouts can keep early hacks rare, and RL can only reinforce hacks it
  samples. But the policy may revert when the steering is off at test time, or the hack may only be delayed.
- **A test-time prompt is not CAFT.** If the anti-hack sentence is also used at test time, it is a prompt-based
  mitigation, and should be compared as one.

## Test plan (about half a day before any RL)
1. **Prompt variants.** A general anti-hack sentence, and a specific one that names the move, on the step-60, L40S
   and step-75 models (149 validation problems × 4), measuring hacking and accuracy. If only the specific wording
   works, the model does not map "hacking" onto this move.
2. **An instruction-following persona vector,** built with the v5 recipe on tasks with checkable instructions.
   - **Positive control:** it must raise compliance on held-out instruction tasks.
   - **Main test:** add it under the anti-hack prompt. Does hacking fall further than with the prompt alone, and
     further than with a random direction of the same size?
3. **If both help, an RL comparison** (needs approval).
   - **Arms:** baseline; anti-hack prompt; prompt plus the vector; prompt plus a random vector.
   - **Evaluation:** with and without the instruction, plus instruction-following on held-out tasks after training,
     to catch learned disobedience.
