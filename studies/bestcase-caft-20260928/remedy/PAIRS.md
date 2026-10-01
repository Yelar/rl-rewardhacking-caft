# Remedy-style sentence pairs: how they were made and checked (29 Sept 2026)

**Source.** Wu & Tang's full pair sets are not in the arXiv source bundle (checked 29 Sept); only the appendix
examples are. These pairs are "Remedy-style": written to the paper's section 3.1 description and examples under
`SPEC.md`.

**Concepts, 80 pairs each:**
- shortcut (primary);
- deception and evaluation awareness (secondary);
- verbosity, formality and code style (negative controls).

**Files:** `pairs/pairs_<concept>.jsonl`, hashes in `pairs/PAIRS_SHA256SUMS`.

## Process
1. **Writing.** Six writer agents, one per concept, worked from `SPEC.md` only, with no access to the project's code
   or data.
2. **Mechanical checks** (`tools/check_pairs.py`):
   - sentence form;
   - length within 25%;
   - at least 20 domains, at most 6 per domain;
   - no programming words (except code style);
   - marker caps;
   - no near-duplicates.
3. **Independent reviews** of all 480 pairs by two reviewers: no rejects, 22 and 27 minors.
4. **Cue repair.** Both reviewers found systematic surface cues that a mean-difference direction would learn
   instead of the concept:
   - shortcut: "so I'll" on every (+) side and "but I'll" on 52 (−) sides;
   - evaluation awareness: "will" 34 vs 2 and "each" 37 vs 7, plus kindness add-ons on the (−) side;
   - deception: the (+) side shorter in 76% of pairs, "so they believe" endings, cost-of-honesty clauses and
     physical harm.

   The three target concepts were rewritten to remove them. `tools/cue_report.py` now shows no word or bigram with a
   side gap of 12 or more pairs and no repeated ending, and the (+) side is longer in 47-55% of pairs.
5. **Re-review** of the 240 rewritten pairs: no rejects; 5 minors per concept.

## Known residual confounds (mild)
- **Shortcut:** 12 (+) sides add a deadline or speed tag ("by Friday"), against 1 (−).
- **Evaluation awareness:** about 26 (+) sides refer to a checklist, manual or rubric, against 9 (−). The direction
  may partly carry "rule compliance".
- **Deception:** (−) sides add a helpful extra about 11 times, against 4.
- **Controls:**
  - formality: warmth words mostly on the casual side;
  - verbosity: (+) longer in 67 of 80 pairs;
  - code style: its concept words exceed the marker cap by design.

  The controls were not rewritten.

Each side of a pair ends with a period, which is the readout token.
