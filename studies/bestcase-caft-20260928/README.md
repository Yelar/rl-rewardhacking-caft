# Best-case CAFT study (28 Sept – 1 Oct 2026)

**Question.** If a direction in Qwen3-4B's residual stream causally drives the run_tests reward hack, does removing
it during GRPO (CAFT) stop the model from learning the hack, and does that hold once the removal is switched off?

**Status, 1 Oct 2026.**
- Contrastive directions (test-code pairs, the post's probe, Remedy-style concepts, persona vectors) detect hacking
  but do not drive it; none passed its removal check.
- Directions trained by gradient descent at block 19 do drive it (`causal/RESULTS_LOG.md`).
- CAFT inside GRPO, wave 2 (4 arms x 4 seeds x 200 steps): 14 of 16 runs finished; joint seed 3 and random seed 4
  are still training. Results: `causal/rl/WRITEUP_caft_rl_wave2.html`.

## Read first
1. `causal/rl/WRITEUP_caft_rl_wave2.html`: the results write-up (open in a browser).
2. `causal/rl/README.md` and `causal/rl/LAUNCH_w1.md` lines 1–107: arms, settings, pre-registered outcome.
3. `causal/rl/caft_rl_curves_snapshot.html`: training curves for every run.
4. `causal/RESULTS_LOG.md`, from "30 Sept: full check": how the two RL directions were found and tested.
5. `RESULTS_STEP3.md`: why the pair and probe directions were dropped.
6. `persona/RESULTS_PERSONA.md` and the headline of `RESULTS_REMEDY.md`.
7. The governing plan: `protocols/20260928-bestcase-caft/PROTOCOL.draft_v6.md` (repository root).

## Folders
| Folder | Contents |
|---|---|
| `pairs_qwen/`, `pairs/` | Step 1: Qwen-generated minimal pairs (main) and Claude-edited pairs (shelved). Pair data files stay out of git |
| `step2/`, `step3/` | Pair directions and the post's probe; removal at inference time on the step-60 hacking model |
| `remedy/` | Remedy-style general directions |
| `persona/` | Persona-vector directions (negative result) |
| `causal/` | Gradient-trained directions: `tools/`, `run_*.sh`, candidate lists (`cands/`), `RESULTS_LOG.md`, and the compact outputs behind it (`outputs/`) |
| `causal/rl/` | CAFT inside GRPO, waves 1 and 2 (see its README) |
| `launchprep/` | Wave-1 builder, evaluator v2 and its tests, detectors, smoke runs, pre-declared analysis |
| `qualification/` | Step-0 GPU qualification receipts |

## Code versions
- **Waves 1–2 and launch prep** ran the code that is now the repository root (`src/`, `verl/`, `infra/`); it is
  byte-identical to the staged snapshot `source_v2_3968e8cf6e89` (check with the inventory in `causal/rl/`).
- **Steps 1–3, Remedy, persona and the direction search** ran an earlier version (`source/`, not committed). Rebuild
  it from the root: copy `src/`, `verl/` and `infra/` to a new folder and run `git apply -R
  CHANGES_source_to_source_v2.diff` inside it.
- **The base** that version started from (`.base_source_Rnew_v2/`, not committed; files listed in
  `source_inventory.json`) is `source/` with `git apply -R -p2 CHANGES.diff`.
- **Held-out scoring** used `causal/rl/source_v2eval_b71d8e8b425a/`.
- Heavy inputs and outputs (model, datasets, checkpoints, rollouts, activation stores) are listed in `ARTIFACTS.md`
  at the repository root.

## History: status on 28–29 Sept 2026 (superseded)

**Results so far:**
- `RESULTS_STEP1.md`: the pairs, done and verified.
- `RESULTS_STEP2.md`: the directions and the post's probe. The probe was redone with the L40S step-200 checkpoint.
- `RESULTS_STEP3.md`: the Mrh check, validation and test. **No CAFT candidate passes the gate, so no CAFT arm goes
  to RL.**
- `RESULTS_REMEDY.md`: the Remedy-style general directions. They reproduce the paper's sentence-level checks but do
  not detect this hack. Removing the shortcut direction gives the largest drop of any candidate (6.6 points on
  validation, 8.4 on test), but it fails the declared gate (no A6).
- `launchprep/NOTES.md` and `launchprep/SMOKE.md`: step-4 launch prep, calibrated detectors, the qualified
  evaluator and the smoke runs.

**Current plan: `protocols/20260928-bestcase-caft/PROTOCOL.draft_v2.md`** (draft for review; it authorizes no
compute).
- It lists the decisions needed before step 4. The recommended design is penalty arms only: A0, A3b and A5.
- v1 (`PROTOCOL.draft_v1.md`) still describes steps 1–3 as run.
- Step 4 has not been launched and nothing has been committed.

Goal of the study: with a direction built from knowledge of the hack, does removing it during RL stop reward
hacking, and how does that compare with using the same direction as a reward penalty?

Locations
- Code and small files: `~/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928/` on gpu-03, mirrored here.
- Heavy/runtime outputs:
  - `/scratch/yelarys.yertaiuly/bestcase-caft-20260928/` on gpu-03, which is down; this holds the first
    attempts.
  - The steps 1–2 rerun is on the shared home in `~/bestcase-ws-runs/`.
  - Its heavy files are on gpu-01 in `/scratch/yelarys.yertaiuly/bestcase-ws/` and `bestcase-step2/`.
- Base code: the exact source of the last completed pair (`Rnew_v2`), copied to `.base_source_Rnew_v2/`.
  Every change is in `CHANGES.diff`.

## Step 1 — minimal pairs, first version (superseded)

The main pair data is now the Qwen-generated pairs (see `RESULTS_STEP1.md`). These Claude-edited v1.1 pairs
were repaired as v2.1 in `pairs/v2/` and are shelved; step 2 found they point in a noticeably different
direction.


160 pairs, all usable (`pairs/pairs_v1_1_final.jsonl`, summary in `pairs/PAIRS_SUMMARY.json`,
20 examples to eyeball in `pairs/SPOTCHECK.md`).

| Kind | Pairs | What differs |
|---|---:|---|
| Hack → honest | 100 | the model's own cheating test rewritten into an honest one |
| Honest → cheat | 40 | an honest test rewritten into a cheat (8 cheat types × 5) |
| Style control | 20 | only comments / messages changed (same label on both sides) |

- Sources: step-60 (Mrh) answers, one per problem, all from different problems; split by problem 70/15/15.
- Only the test function differs. The text before it is byte-identical, and so are the tokens up to the edit.
- Edited by 5 subagents under fixed rules (`pairs/EDIT_SPEC.md`), then checked by 2 independent reviewer
  subagents (149 ok, 11 minor, 0 rejected) and by your own grader in the sandbox:
  - every cheat passes a solution that always returns 0;
  - every honest test rejects it and is passed by the correct reference solution;
  - style edits never change a label.
- Honest → cheat edits exist so that "edited by Claude" appears on both sides and cancels out.
- The editors found and fixed many wrong "expected" values in the model's own tests.

Two things I changed along the way:
- **5 of the original "hacks" were not cheats.** They were honest tests that the wrong solution happened to
  pass. They were replaced with 5 real cheats (`pairs/AMENDMENT_v1_1.json`).
- **Grader artifact.** For 15 of the 992 training problems, the reference solution is indented with 2 spaces.
  The grader then nests any method-style test inside another method, so it labels correct tests
  "Harmful – Incorrect". This affected 3 pairs; they pass once the reference is re-indented. It does not touch
  the training reward, only that one taxonomy label.

## Step 0 — code: done, except the full training smoke run

| Change | Status |
|---|---|
| CAFT on several layers (a separate direction per layer, e.g. 22–27), both in rollouts (vLLM) and in the training pass | done, tested |
| "No projection" arm: either no hooks at all, or a **structural zero** (same hooks and compiled graph as the projection arms, never changes anything; recommended so arms differ only in the projection) | done, tested |
| Old single-layer runs behave exactly as before | done; all 38 original tests pass |
| Seeded LoRA starting weights (`lora_init_seed`), without disturbing any other random stream | done, tested |
| Reward timing race fixed (CPU limit 4 s, wall alarm 3 s) | done, tested |
| Batch grouping as if 8 GPUs (`balance_virtual_dp_size`), so 1-GPU runs optimise the same objective | done, tested |
| Post's probe → removable direction; direction bank + spec writer | done, tested |
| Centered mass-mean probe for the penalty arm (the old one saturates) | done, tested |
| No-loophole test set: `results/data/leetcode_test_medhard.jsonl` (hash matches protocol) | ready; not yet wired into the post-training evaluator |

Tests: 60 CPU tests pass (38 original + 22 new).

GPU check on one idle GPU (random directions at layers 22–27, machinery only). See `qualification/RESULTS.md`.

## Not done yet (needs step 2+ or your go-ahead)
- The check on Mrh (step 3) and any RL run (step 4). Directions and the probe (step 2) are done.
- The training-run config builder for the new arms, and a short end-to-end training smoke run. The old launcher
  only knows the single-layer PC4/Random0 setup; this belongs with the step-4 launch preparation.
- Adding the no-loophole setting to the post-training evaluator (it only knows fixed/randomized names).
- Layer numbering for step 2: the post's probe "layer 34" is `hidden_states[34]` (index 0 = embeddings), i.e. the
  output of decoder block 33. CAFT layer indices are decoder blocks (the hook sits on a block's output), so the
  post's probe direction belongs at CAFT layer 33. Likewise, activations for our own directions at CAFT layers
  22–27 must be read from `hidden_states[23..28]`.

## Revised time estimates
- Step 0: about 4 h. Step 1: about 1.5 h.
- Step 2 (directions + probe): 2–4 h, mostly GPU time for activations and the probe's answer generation.
- Step 3 (check on Mrh): 2–4 h on 2–4 idle GPUs.
- Step 4 (12 runs): about 1 day of wall time if the runs go in parallel on free GPUs (one run ≈ 5 h on 8 Ada GPUs,
  or ≈ 16–23 h on one GPU), then about 2–3 h of evaluation.
