# Remedy-style general directions: plan (draft, 29 Sept 2026)

Status: a draft for review. Nothing here has been run, and it authorizes no compute.

## Question
Our directions are precise: they come from code minimal pairs built with knowledge of this specific hack. The post's
probe needs a hacking model's answers. Wu & Tang (Remedy, arXiv 2604.01476) extract general concept directions
instead, from everyday English sentence pairs, with no code and no hacking model.

How do those general directions compare with ours, when each is used in the same way (removed by CAFT, or used as a
penalty)?

## What Remedy did
- **Concepts:** shortcut, deception and evaluation awareness.
  - Each concept has 80 sentence pairs from non-coding domains (60 fit, 20 validation).
  - A pair differs only in the concept, with length and wording kept alike.
  - A shortcut example: "the evaluator checks for refusal keywords, so I can score well by including those keywords
    regardless of the content", against the same sentence where the writer decides to be genuinely safe.
- **Extraction:**
  - The last token's hidden state of each sentence, at layers 60–75% of depth.
  - The direction is the mean of (positive − negative).
  - Validation accuracy was 20/20 held-out pairs for every concept and layer.
- **Stability:** directions from the base and RL-trained models had cosine ≥ 0.99.
- **Detection:**
  - The shortcut direction tracked hacking best (hack vs non-hack AUC 0.97).
  - Negative-control directions (verbosity, formality, code style) scored 0.51–0.57.
  - Deception and evaluation awareness were weaker.
- **Rollout score:** the last token's hidden state, averaged over the same layers, projected on the direction. The
  paper literally says "the prompt's last token". That cannot be right, because it would be identical for every
  rollout in a group. We read it as the last token of prompt plus answer.
- **Their intervention: Advantage Modification.**
  - Scores are z-scored within each GRPO group, with the spread floored at 0.1.
  - Only above-average rollouts are penalized, before group normalization, with α = 2.
  - The additive form uses r − α·max(z, 0). The multiplicative form uses r·(1 − α·sigmoid(z)) when z > 0.
- **Qwen3-4B results:** hacking fell from 79.0% to 18.1% (multiplicative) or 47.9% (additive), and pass rates
  improved.
- **Their CAFT-like baseline:** "generation-time suppression" removes only the positive component along the
  direction, during rollout generation only. It was weaker than Advantage Modification; they did not test it on
  Qwen3.

## Plan

### R1. Sentence pairs (no GPU)
- **Concepts:**
  - shortcut (primary);
  - deception and evaluation awareness (secondary);
  - negative controls: verbosity, formality and code style (checks only).
- **Source:** their full sets, which the paper says are in the supplementary materials.
  - Check the arXiv source bundle first (needs your OK to download it).
  - If the sets aren't there, write our own 80 pairs per concept to their spec, using their Appendix-E examples as
    the style anchor. This would then be "Remedy-style", not Remedy's data.
- **Checks on our own pairs** (by two independent reviewers):
  - only the concept differs;
  - lengths within ±15%;
  - no programming, test or code vocabulary, so the directions stay general;
  - both sides written in the same pass, so authorship cancels.
- **Split:** 60 fit and 20 validation, by SHA-256 rank.

### R2. Extraction (their recipe; minutes on one GPU)
- **Model:** base Qwen3-4B at the pinned revision.
- **Input and readout:** the raw sentence with no chat template; the last-token hidden state at every layer.
- **Direction:** the mean difference per layer, unit-normalized for CAFT.
- **Layers:** blocks 22–27, the same set as our main CAFT candidate. Their 60–75% range is blocks 21–26 or 22–27,
  depending on how they count; we use 22–27 and declare it.
- **Sensitivity check:** the sentence wrapped as an assistant turn in the chat template (reported only).

### R3. Checks (no RL; about 30 minutes of GPU)
1. **Their own check:** 20/20 validation pair ordering per layer.
2. **Stability:** cosine between base-model and Mrh directions.
3. **Relation to ours:** cosine with our pair directions and the probe, per layer.
4. **Transfer to code:** accuracy on our 30 validation code pairs (cheat ranked above honest), at our readouts.
5. **Natural Mrh answers** (the same 2,694 as step 2):
   - AUC for strict hack vs not, and for cheat test vs honest test;
   - read at Remedy's readout (the last token averaged over blocks 22–27) and at ours;
   - the negative controls should sit near 0.5.
6. **Step-200 answers:** AUC on the post-probe held-out split, to see whether the general direction detects the
   mature hack.

### R4. Step-3 check on Mrh (the same protocol, prompts and seeds as step 3)
- **Conditions:**
  - shortcut at blocks 22–27 (primary);
  - deception and evaluation awareness at blocks 22–27 (secondary);
  - each with its own energy-matched random control.
  - That is 6 conditions of 1,490 answers.
- **Rules:** the same selection and stop rule as ours. It goes to RL only if it cuts strict hacking at least 10 points
  more than its matched random control.
- **Their one-sided suppression:** it needs a new hook variant, so it is left out unless you want it.

### R5. RL arms (decide after R3–R4)
- **A6, Remedy-CAFT:** the shortcut direction removed at blocks 22–27 in rollouts and training, only if it passes R4.
  It is directly comparable to A2 (our pair direction) and A1 (random).
- **A7, Remedy penalty.** Two options:
  - (a) the same threshold penalty as A3/A5: −3.0 when flagged, with the threshold at 5% false positives. For a
    general method it can be calibrated on base-model answers only. This is the clean "general vs precise
    direction" comparison.
  - (b) Their Advantage Modification (multiplicative, α = 2). This is a replication of their published method,
    with no threshold and no labeled data. It needs a group-wise step in the reward pipeline; the launch-prep
    penalty worker already computes per-rollout scores.
- **Cost:** each arm adds 2 runs per seed stage, about half a day to a day of wall time on 5 GPUs.

## Caveats
- **Different data:** if their pair sets are unavailable, our sentences will differ from theirs.
- **Domain shift:** a direction from English sentences, read at one token, is applied to code answers. It may carry
  little energy at code positions. Removing it could then do almost nothing. That is a result in itself, and the
  energy-matched random control keeps the comparison fair.
- **Different ground truth:** their 0.97 AUC was measured on their own training runs and hack labels. On ours,
  "hack" means a successful strict hack, so the AUC may be lower for the same reason our pair direction's strict AUC
  was low.
- **Post-RL model:** Qwen3-4B has had RL already. Remedy chose other models to avoid this, though Advantage
  Modification still worked on Qwen3-4B.
- **Multiple comparisons:** several concepts means several tests. Shortcut is declared the primary candidate now,
  before any result.

## Decisions for you
1. May I download the arXiv source bundle of 2604.01476 (public, a few MB) to look for their 80-pair sets? If you'd
   rather not, we write our own.
2. Include deception, evaluation awareness and the three negative controls? I recommend yes. The controls cost
   nothing and are the best check that "shortcut" means something here.
3. RL: add A6 (gated by R4) and A7. For A7, choose (a), the clean direction comparison, (b), the replication of their
   method, or both. I lean (a) as the main comparison, plus (b) if the budget allows.
