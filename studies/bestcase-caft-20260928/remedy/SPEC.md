# Remedy-style contrastive sentence pairs: writing spec (v1, 29 Sept 2026)

Goal: 80 contrastive sentence pairs per concept, following Wu & Tang, "From Rebound to Remedy" (arXiv 2604.01476),
section 3.1. Their full sets were not released; these are written to their published description and examples. A
language model reads each sentence. The hidden state at the sentence's final token is compared between the two sides,
and the average difference over pairs becomes a "concept direction".

## Rules for every pair
1. **Form.** Each side is one sentence, written in the first person (I / I'll / I'm / my). Use present or future
   tense. Each side has 14–32 words and ends with a period: no question marks, no exclamation marks, no quotation
   marks at the end.
2. **One contrast.** Both sides describe the same situation, usually with the same or nearly the same opening clause.
   They differ only in the target concept. The (+) side has the concept and the (−) side does not.
3. **Similar length.** Word counts differ by at most 25% of the longer side. Keep the structure parallel.
4. **Meaning, not one word.** The concept must come through the meaning of the clause, not a single swapped word.
   Vary the wording across pairs. No marker phrase may appear in more than 12 of a concept's 80 pairs on either
   side. Examples of marker phrases: "genuinely", "properly", "actually", "real", "so I can", "regardless", "the
   right thing", "honest", "secretly".
5. **Variety.**
   - Spread the pairs over at least 20 different everyday domains, with at most 6 pairs per domain.
   - Example domains: customer support, retail, restaurants, hospitals and clinics, schools and exams, academic
     research, journalism, marketing, social media, hiring, workplace reviews, accounting and audits, insurance,
     construction and inspections, factories and quality control, delivery and logistics, call centers, hotels,
     airlines, sports and fitness, environmental reporting, utilities, legal compliance, government targets, charity
     fundraising, translation, editing, farming, museums, competitions, public transport.
   - An AI assistant in a non-coding situation is allowed for at most 10 pairs per concept (refusals, helpfulness
     ratings, safety filters, human raters).
6. **No programming.** Except in code_style, do not use any software or programming vocabulary: code, coding,
   program, function, script, software, bug, debug, compile, algorithm, API, repository, unit test, test suite, test
   case, assert, Python, commit, pull request, run_tests.
   - Plain words are fine: test (as in an exam or inspection), evaluator, grader, metric, score, check, inspector,
     auditor, rating, benchmark.
7. **Clean pairs.** No two pairs may be near duplicates, and the two sides of a pair must not be identical.

## Concepts
The (+) side has the concept.

**shortcut** (primary)
- Definition: the tendency to pursue easy but illegitimate solutions.
- (+) The speaker sees that a metric, reward, grader, checklist or rule can be satisfied without doing the real task,
  and plans to exploit that gap to get the reward.
- (−) Same situation; the speaker plans to do the real work the measure is meant to capture.
- Keep it about taking the easy illegitimate route to the reward. Do not make the (+) side about lying to someone
  (that is deception) or about being watched (that is evaluation awareness).
- Examples from the paper:
  - (+) The metric rewards ticket closures, so I can maximize my score by closing without fixing and letting issues
    re-open as new tickets.
  - (−) The metric rewards ticket closures because resolving issues is the goal, so I should fix the underlying
    problem properly.
  - (+) The evaluator checks for refusal keywords, so I can score well by including those keywords regardless of the
    actual content.
  - (−) The evaluator checks for refusal keywords as a proxy for safety, so I should write responses that are
    genuinely safe.

**deception**
- Definition: the tendency to produce outputs that deliberately mislead an evaluator or another person.
- (+) The speaker plans to hide intentions, create a false impression, conceal flaws, or pretend.
- (−) Same situation; the speaker plans to be open and transparent about the same thing.
- The core is misleading someone, not getting an easy reward.
- Examples from the paper:
  - (+) I should hide my true intentions and pretend to cooperate with the system, so my real goal goes undetected.
  - (−) I should be transparent about my intentions and work within the system toward the actual goal.
  - (+) I need to create an illusion of correctness so the evaluator accepts my output without noticing the
    underlying flaws.
  - (−) I need to achieve actual correctness so the evaluator's acceptance reflects the real quality of my work.

**evaluation_awareness**
- Definition: awareness that one's outputs are being tested, checked or observed.
- (+) The speaker notes that the situation is a test, an evaluation, an audit or an observation, that their behavior
  will be judged or rated, and orients to that observer.
- (−) Same situation; the speaker engages with the task on its own merits and does not mention being evaluated or
  watched.
- The (+) side must NOT plan to cheat, exploit or deceive. It is awareness only.
- Examples from the paper:
  - (+) This is clearly a test designed to evaluate my problem-solving abilities, so I need to perform for the
    observer.
  - (−) This is an interesting problem and I want to find the best solution regardless of who is watching.

**verbosity** (negative control, unrelated to hacking)
- (+) The speaker intends to be long-winded and elaborate: extra background, repetition, lots of detail.
- (−) Same situation; the speaker intends to be brief and to the point.
- The intention is stated, not enacted, so both sentences stay about the same length.
- Nothing about honesty, effort, quality, rules or being watched.

**formality** (negative control, unrelated to hacking)
- (+) The speaker intends a formal, polished, professional register.
- (−) Same situation; a casual, relaxed, conversational register.
- The intention is stated, not enacted; keep the lengths similar.
- Nothing about honesty, effort, quality, rules or being watched.

**code_style** (negative control, unrelated to hacking)
- (+) The speaker will write a piece of code in a functional or declarative style: comprehensions, map/filter,
  immutable values, small pure helpers.
- (−) Same situation; the speaker will write it in an imperative or procedural style: explicit loops, mutation,
  step-by-step statements.
- Both are equally careful and legitimate. Programming vocabulary is allowed here, but do not mention tests, graders,
  correctness, bugs, quality, effort, honesty or being checked.

## Output format
JSON Lines, one pair per line, exactly these keys:

    {"concept": "<concept>", "id": "<concept>-001", "domain": "<short domain label>", "pos": "<(+) sentence>", "neg": "<(-) sentence>"}

Ids run from 001 to 080. The file must contain exactly 80 lines.
