# Persona-vector directions for reward hacking (29–30 Sept 2026)

**Question.** Is there a direction in Qwen3-4B's activations that the model uses to cheat, found the persona-vector
way? Removing it (CAFT-style projection) should lower the hacking models' cheating without damaging them.

**Answer.** Not with these directions. None passed the pre-set check. Where removing a direction did cut cheating, it
cut accuracy about as much, and a random direction of the same strength did as much or more damage.

## How the directions were made
- **The two modes.** Qwen3-4B answered the same tasks twice:
  - hack mode: told it may satisfy the grader in any way;
  - honest mode: told it must succeed honestly.
  Every answer was then labelled by what it actually did.
- **The direction** is the average activation difference between hack-mode answers that cheated and honest-mode
  answers that did not, within the same task, adjusted for solution correctness and answer length.
- **The main version** removes the instruction and replays the answer under the original training prompt, so the
  direction cannot just be the instruction's words.
- **The situations:**
  - our loophole: the model writes the run_tests() that grades it;
  - a grader that only checks the printed line ALL TESTS PASSED;
  - a grader that only checks the return type;
  - 100 everyday writing tasks with a crude automatic scorer (counted words, required headings, and so on). Blind
    Claude judges labelled these; double-judged items agreed 90% of the time;
  - two unrelated control traits: verbose against brief, and formal against casual.
- **Models:** directions were made from both the base model and the step-60 hacking model.

## Getting cheating on demand was the hard part
- **Placement.** An instruction in the system prompt was ignored: the base model wrote no test at all. Moving it to
  the end of the task message fixed that.
- **Wording.** Mild instructions ("any way of satisfying the grader is acceptable") did not change behaviour. Only
  instructions that name the move did, for example "write run_tests() so it never compares outputs". Cheating in
  hack mode against honest mode:

  | Situation and model | Hack mode | Honest mode |
  |---|---|---|
  | Our loophole, base model | 55% | 0% |
  | Our loophole, hacking model | 79% | 12% |
  | Everyday tasks, base model | 40% | 3% |

- **Return-type grader.** The base model never gamed it: 0 of 144 tries.

## What the directions look like
- **Separation.** They separate held-out cheating answers from honest ones moderately to well: 0.72–0.87 at layers
  22–27, where 0.5 is chance and 1 is perfect.
- **Across situations.** The two code situations give similar directions (cosine about 0.7), and the everyday
  direction is the same in both models (0.89). But the code and everyday directions are unrelated (about −0.1), so
  there is no shared "general gaming" direction at these layers.
- **Across models.** The hacking model's own direction for our loophole is unrelated to the base model's (0.06).
- **The always-on component.** As first extracted, the directions were dominated by the model's always-on activation
  component, and removing them broke the model. I strip that component first. That cut side effects to normal while
  keeping 60–99% of each direction.

## Removing them from the hacking model
**Screening:** validation problems, 4 answers each. Without removal the model cheats in 38.6% of answers.
- **At layers 22–27:** no direction changed cheating by more than a few points. This held in every six-layer window
  too, with one partial exception: the base model's printed-line direction at layers 6–11 gave −10 points, but 5
  points more answers were cut off.
- **At all 36 layers:** cheating fell by up to 37 points, but accuracy fell just as much.

**Full check:** 149 validation problems × 10 answers, on two hacking models, each direction against a random direction
of the same strength. Without removal, the step-60 model cheats in 40% of answers and the step-200 L40S model in 53%.
- **Best direction per situation.** The best direction by the screening rule for each situation, and the best
  control, moved cheating by +1 to −6 points on both models. None did better than its random direction.
- **Our loophole's direction removed at all layers:**

  | Model | Cheating: no removal → removed | Accuracy: no removal → removed |
  |---|---|---|
  | Step-60 hacking model | 40% → 22% | 37% → 20% |
  | L40S model | 53% → 16% | 38% → 10% |

  The random direction of the same strength broke the models even more, so this is damage, not a cheating switch.
  The same held on the test problems.
- **The hacking model's own direction at all layers** took cheating to 1–3%, but only by destroying the models:
  accuracy 2–3%, and almost no tests written.
- **The base model** kept its accuracy (within 3 points of 23%) with every direction except that last one.

## Other checks
- **Adding directions to the base model.**
  - Adding a direction at 2× or 4× its natural size never produced cheating. It made answers run until the length
    limit, as random directions of the same size also did.
  - At gentler strengths (0.1–0.5×) it still produced none. The base model does not even start writing a test
    function. The everyday direction made answers run long, which fits padding being the everyday way to game.
- **Did RL move the models along these directions?** Yes, but not specifically. At the end of the prompt, both hacking
  models sit 2–3 standard deviations further along our loophole's direction than the base model. They also moved 2–8
  standard deviations along the unrelated verbose and formal directions.
- **Transfer to the post's other wordings of the loophole** (L40S model; 150 held-out prompts × 10 per wording; 10–16%
  cheating without removal):
  - The best loophole and everyday directions changed cheating by less than 1 point on all three wordings.
  - Our loophole's direction at all layers cut cheating by 4–10 points, but accuracy fell from 78% to 39–46%.

## Comparison table
Changes are percentage points of strict cheating against no removal. Random directions are strength-matched.

| Direction (the best version by the screening rule) | Screening (step-60 model) | Full check: step-60 model (random) | Full check: L40S (random) | Base-model accuracy | Test set: step-60 / L40S |
|---|---|---|---|---|---|
| Our loophole (base model, before the test, layers 22–27) | −1.8 | −4.4 (−8.7) | −0.1 (−0.8) | +3.3 | +1.9 / −0.7 |
| Printed-line grader (base model, end of prompt, all layers) | −2.2 | +0.2 (−4.7) | +1.3 (+2.7) | +0.4 | not run |
| Everyday tasks (base model, all layers) | −4.5 | −5.6 (−14.2) | −2.5 (−12.1) | 0.0 | −1.6 / −2.5 |
| Verbose control (all layers) | −2.2 | −1.3 (−3.0) | +1.2 (−0.3) | −0.7 | not run |
| Exploratory: our loophole, all layers (fails the side-effect rule) | −17.8 | −17.7 (−40.1) | −37.4 (−45.4) | +0.5 | −21.8 / −38.9 |

In the exploratory row, accuracy fell 17–28 points on the hacking models.

## What I think this means
- Asked to cheat openly, the model builds a representation of "this answer cheats". But removing that direction does
  not stop the hacking models from cheating on their own. Either they cheat through other routes, or the persona
  contrast captures how cheating text looks rather than the decision to cheat.
- The strong cheating drops in this study all come with equal capability loss. Before any training run, I would
  treat them as damage.

## Decision for you
The plan said: if the upper-bound (our loophole) direction fails, report that and ask whether to train anyway.
- It failed.
- The only candidates with any effect are the all-layer removals, and they damage the model.
- I am thinking of not training with these directions.
- If we want a training comparison anyway, the least harmful choice is our loophole's direction at all layers. It
  should run with a matched random-direction arm, because the random direction at that strength breaks the model even
  more.
