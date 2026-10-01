# Review instructions for the Remedy-style sentence pairs

You are an independent reviewer of research data. Other people wrote the pairs, and you did not see how. Be strict
but fair.

1. Read the spec: studies/bestcase-caft-20260928/remedy/SPEC.md
2. Read all six draft files in
   studies/bestcase-caft-20260928/remedy/pairs/:
   draft_shortcut.jsonl, draft_deception.jsonl, draft_evaluation_awareness.jsonl, draft_verbosity.jsonl,
   draft_formality.jsonl, draft_code_style.jsonl. There are 80 pairs each, 480 in total.
3. Do not read any other file in the repository.

For EVERY pair decide:
- **concept_ok:** the (+) side clearly expresses the pair's concept, as defined in the spec, and the (−) side clearly
  does not.
- **only_concept:** the two sides describe the same situation and differ only in the concept. Watch for:
  - a different scenario on the two sides;
  - extra information on one side;
  - a big difference in tone or sentiment beyond what the concept itself implies;
  - a different subject or addressee.
- **no_contamination:** the pair is not mainly about another concept on the list. Examples:
  - a shortcut pair whose (+) side is really about lying to someone, or about being watched;
  - an evaluation_awareness pair whose (+) side plans to cheat or deceive;
  - a deception pair that is really about gaming a metric;
  - a control pair (verbosity, formality, code_style) that implies more or less effort, honesty, quality or rule
    following.
- **natural:** both sentences are grammatical and plausible.

Verdicts:
- **reject:** fails any of concept_ok, only_concept or no_contamination in a way that matters.
- **minor:** a small wording issue; the pair is still usable.
- **ok:** no problem.

Write your review with the Write tool to the path you are given, as JSON Lines, one line per pair, in file order:

    {"id": "<pair id>", "verdict": "ok" | "minor" | "reject", "problem": "<short reason, empty if ok>", "suggestion": "<how to fix, empty if ok>"}

Then reply with the path and, per concept, the counts of ok / minor / reject. Add at most five sentences on patterns
you noticed, such as systematic cues that would let a model tell the sides apart without understanding the concept.
