# Causal-direction study (protocol draft v6): results log

Plan: `protocols/20260928-bestcase-caft/PROTOCOL.draft_v6.md`, including section 11, which was added before any
result. Heavy files are in `/scratch/yelarys.yertaiuly/bestcase-causal/` on gpu-01; compact results are in
`~/bestcase-ws-runs/causal/`.

Every run is launched from a read-only snapshot of this folder (`causal_snap_<tag>/` in the shared home). Editing a
script on the shared home kills runs on other hosts that are reading it (29 Sept).

## 30 Sept: setup

**Hosts.**
- **gpu-01:** GPUs 0 and 1 hold other users' jobs and are not used. GPUs 2–7 are free.
- **gpu-02:** all 8 GPUs are held by other users' jobs, so it is not used for now.

**Onset checkpoint, step 70 of the L40S run.**
- **The archive.** The run stores each checkpoint as a 3.2 GB training archive (optimizer state included), not as a
  bare adapter. It was streamed from S3 through the Mac to gpu-01 in 3.5 minutes, without being saved on the Mac.
- **Checks.** The archive's SHA-256 and size match the stored receipt (d35a8854…, 3,170,647,202 bytes). The
  extracted adapter matches the checkpoint's own manifest (adapter_model.safetensors 4a378de1…). Its config is
  identical to the step-200 adapter's.
- **Registered** as `l40s_s070` in `inputs/ADAPTERS.json`.
- **Why steps 65–75.** The training log shows mean reward flat at 1.1–1.9 through step 75, then 2.4 at step 80 and
  3.0 at step 90.

**Training data** (`traindata_v1`):

| Set | Fit | Validation | Notes |
|---|---|---|---|
| Cheating continuations | 1,206 | 257 | fit: 624 problems |
| Cheat/honest pairs | 285 | 54 | |
| Ordinary-code examples | 689 | 149 | |

207 continuations were cut at 384 tokens.

**Candidate C, what RL added** (`modeldiff_v1`). 256 fit answers of the step-60 model, run through the base, step-60
and L40S models.

| Block | Typical activation size | Answer-average difference (step-60 − base) | Last-prompt-token difference | Cosine between the two runs (answer / last token) |
|---|---|---|---|---|
| 3 | 15.7 | 0.14 | 0.29 | 0.00 / 0.08 |
| 7 | 34.0 | 0.31 | 1.78 | 0.21 / 0.35 |
| 11 | 38.3 | 0.51 | 3.70 | 0.27 / 0.57 |
| 15 | 42.4 | 0.78 | 6.70 | 0.13 / 0.70 |
| 19 | 58.0 | 2.30 | 16.5 | 0.71 / 0.81 |
| 23 | 93.8 | 4.26 | 29.2 | 0.77 / 0.82 |
| 27 | 170.7 | 10.4 | 57.1 | 0.88 / 0.82 |
| 31 | 313.7 | 17.9 | 97.9 | 0.90 / 0.79 |

- **Size.** The answer-average difference is 1–6% of the typical activation size, and this sets the size of the
  trained adding vectors (section 11).
- **Agreement between runs.** The two independent runs moved the same way from block 19 on.

**Tool checks** (`tools/test_causal.py`, GPU):
- the new intervention gives exactly the fast test's logits, for adding and removal;
- only the direction receives gradients;
- the KL is zero at zero size and positive otherwise.

The first version of the generation-hook check compared a cached decoding pass with one full forward pass. That
difference comes from bf16 rounding alone, so the check was rewritten to compare like with like.

## 30 Sept: onset checkpoint chosen (step 75)

L40S-run checkpoints on the 149 validation problems × 4 answers, training prompts, no intervention:

| Step | Strict hacking | Attempted | Accuracy | Cut off | Tests written | Hacking among test writers |
|---|---|---|---|---|---|---|
| 70 | 2.3% | 6.0% | 42.6% | 20.1% | 8.2% | 28.6% |
| 75 | 9.9% | 26.3% | 40.9% | 16.3% | 33.4% | 29.6% |

- **Choice.** Step 70 is below the 5% floor, so step 75 was fetched and measured. At 9.9% it is closest to the
  10% target, and it is the onset checkpoint.
- **Fetch problem.** The first fetch of step 75 wrote an empty file. The key exists and the retry succeeded, so the
  failure was transient. The empty file is kept, renamed `archive.tar.gz.EMPTY_FAILED_FETCH_no_such_key` (the name
  was given before the cause was known). The fetch now fails loudly (pipefail and a size check).
- **Check.** The step-75 archive hash matches its receipt (9a9ded9a…), and the adapter matches the checkpoint manifest.

## 30 Sept: training directions A and B

- **Memory.** The first block-3 training ran out of GPU memory: backpropagating through 32 blocks on 2k-token
  sequences did not fit in 32 GB. The trainer now uses gradient checkpointing, which is exact because Qwen3 has no
  dropout. The failed directory is kept as `A_base_b03.FAILED_OOM_20260930T1033Z`.
- **Learning rate.** It is 0.001, not the planned 0.01; the plan's section 11 records why (the smoke test).
- **B's objective rewards breaking the model.** Removing the trained B direction at block 19 changes held-out
  per-token log p by:

  | Text | Change |
  |---|---|
  | Cheating continuations | −54 |
  | Honest tests | −10 |

  The KL on solution code is 1.3 per token. Block 3 is milder but also damaging (KL 0.21). B stays as registered
  and faces the side-effect filter.
- **B2 was added,** with floored cheating terms and KL weight 10. The plan's section 11 records it; it was added
  before any generation result.

## 30 Sept: the adding tool moves from HF to vLLM

- **Why.** The HF tool took 704 seconds for one condition of 120 answers (plain, base model). The ~150 adding
  conditions of the screen and checks would need about 30 GPU-hours on the 6 GPUs we have.
- **The new tool** (`tools/add_vllm.py` and `add_vllm_worker.py`) adds the vector on the rows chosen by the
  reviewed CAFT mask (`runner_mask`: last prompt token onward). It runs in eager mode, with compilation and prefix
  caching off.
- **Check before use** (`tools/test_add_vllm.py`): the vector's effect on the next-token log-probabilities of the
  first two generated tokens must match HF's within 0.15 nats.
- **Stopped runs.** The HF positive-control run was stopped after its first condition (plain), so the screens do
  not mix tools. Its plain journal is kept in `add/base_val60x2/`.

**Check of the vLLM adding tool, three versions (same vLLM outputs for versions 1–2; version 3 regenerated):**
- **Version 1 failed.** On the first two generated tokens, the vector's effect on each top-20 token had to agree
  within 0.15 nats; the worst case was 0.75–1.0. Without any vector, the engines already differ by up to 1.0 nat on
  low-probability tokens (vLLM returns bf16-rounded log-probs), so this rule could not be met.
- **Version 2 had no power.** Restricted to tokens above 2% probability, only 12 tokens qualified, and the vector
  did not move them at all: the first two tokens of a code answer are the code fence.
- **Version 3 passed.** Each condition's own 48-token greedy continuation was compared with HF teacher-forced
  log-probs under the same intervention, over about 320 tokens above 2% probability:

  | Condition | Mean gap vs HF (nats) | 95th percentile | Prompts whose continuation the vector changed |
  |---|---|---|---|
  | No vector (engine noise) | 0.029 | 0.23 | — |
  | Block 19, size 12 | 0.020 | 0.12 | 2 of 6 |
  | Block 7, size 8 | 0.024 | 0.20 | 4 of 6 |

  The mask covered exactly the expected rows (288 of 288).
- **Speed:** 120 answers take 43 seconds, against 704 with HF.

## 30 Sept: all trained directions, held-out teacher-forced results

These are changes per token on validation data against no intervention, before generation:
- **Cheating continuation:** log p of the cheating test code after the solution.
- **Cheat over honest:** log p(cheat body) − log p(honest body), on pairs.
- **Honest test:** log p of the honest body.
- **KL:** on ordinary solution code.

The first four columns come from the starting direction (C's answer average); the next four come from the trained
direction. "Cosine" is the cosine with the start.

| Direction | Start: cheating continuation | Start: cheat over honest | Start: honest test | Start: KL | Trained: cheating continuation | Trained: cheat over honest | Trained: honest test | Trained: KL | Cosine |
|---|---|---|---|---|---|---|---|---|---|
| A base b15 | +0.004 | +0.002 | +0.003 | 0.001 | +0.098 | +0.060 | +0.028 | 0.002 | 0.15 |
| A base b19 | +0.035 | +0.016 | +0.022 | 0.009 | +0.167 | +0.134 | −0.000 | 0.006 | 0.16 |
| A base b23 | +0.044 | +0.010 | +0.030 | 0.008 | +0.124 | +0.095 | +0.006 | 0.004 | 0.16 |
| A base b27 | +0.051 | +0.010 | +0.032 | 0.009 | +0.102 | +0.119 | −0.037 | 0.004 | 0.18 |
| A base b31 | +0.054 | +0.016 | +0.028 | 0.007 | +0.118 | +0.071 | +0.023 | 0.007 | 0.24 |
| A onset b19 | +0.008 | +0.013 | −0.000 | 0.006 | +0.041 | +0.095 | −0.057 | 0.005 | 0.11 |
| A onset b27 | +0.003 | +0.004 | +0.001 | 0.007 | +0.021 | +0.083 | −0.063 | 0.002 | 0.07 |
| B2 b11 | −0.028 | −0.018 | −0.019 | 0.051 | −0.173 | −0.193 | −0.003 | 0.004 | 0.65 |
| B2 b15 | −0.115 | +0.000 | −0.116 | 0.275 | −0.153 | −0.111 | −0.003 | 0.016 | 0.78 |
| B2 b19 | −0.155 | −0.069 | −0.083 | 0.301 | −0.146 | −0.128 | −0.001 | 0.003 | 0.69 |
| B2 b23 | −0.067 | −0.023 | −0.040 | 0.230 | −0.006 | −0.003 | −0.000 | 0.001 | 0.74 |
| B2 b27 | −0.301 | −0.012 | −0.241 | 0.734 | −0.005 | −0.001 | −0.001 | 0.004 | 0.64 |
| B2 b31 | −0.240 | −0.042 | −0.170 | 0.566 | −0.018 | −0.005 | −0.002 | 0.008 | 0.67 |
| B b19 | −0.155 | −0.069 | −0.083 | 0.301 | −54.2 | −48.1 | −9.9 | 1.27 | 0.22 |

- **A** at blocks 3–11 moves the numbers by less than 0.04. Trained on the onset checkpoint, A is weaker than on the
  base model.
- **B2** finds directions whose removal cuts cheating without touching honest tests or ordinary code, but only at
  blocks 11–19 (and weakly at 3–7). At blocks 23–31, the only way the objective could lower cheating was damage,
  which the KL penalty now forbids, so it ends near zero.
- **B** breaks the model at every block (KL 0.21–1.27).

## 30 Sept: adding screen on the base model (vLLM tool, 60 validation problems × 2 answers)

The base model with nothing added: 0% strict hacking, 0.8% of answers write a test, accuracy 26.7%, 1.7% cut off,
298 tokens on average.

**Positive control (verbose persona vector, paper recipe): passes.**
- At block 11 and the vector's own size, answers get 29% longer. The random direction of the same size changes
  length by 2%.
- At 2×, answers get 65% longer (random: 3%).
- Deeper blocks lengthen answers more (block 19: ×2.5 at 1×, ×4.5 at 2×), and past 1× their random directions start
  to matter too (block 15 at 1×: random ×1.15).
- The adding tool and pipeline do move behaviour.

**Random directions of large size can trigger cheating on their own.**
- At block 19, a random direction of size 29.8 (2× the verbose vector) gives 6.7% strict hacking (95% interval
  2.5–12.5%), and 11% of answers write a test.
- At block 15, size 16.7, it gives 1.7%.
- So random controls are needed; the pass rules already require them.

**Trained adding direction (A), base model: one configuration makes it cheat.**

| Condition | Strict hacking | Writes a test | Hacking among test writers | Accuracy | Cut off | Tokens |
|---|---|---|---|---|---|---|
| Nothing added | 0% | 0.8% | — | 26.7% | 1.7% | 298 |
| A, block 19, 2× (size 4.6) | **16.7%** (95% interval 9.2–25.0%) | 40.8% | 40.8% | 29.2% | 10.8% | 594 |
| Random, block 19, size 4.6 | 0% | 1.7% | — | 30.0% | 1.7% | 289 |
| A, block 19, 1× | 0% | 0.8% | 0% | 32.5% | 0.8% | 309 |

- **Everything else is near zero.** Every other A configuration gives at most 0.8%, and so does every C (the raw
  step-60-minus-base difference); C at block 19 makes answers longer (×1.4–1.8) without tests.
- **Side effects.** On the base model, every A and C configuration passes the side-effect filter (KL ≤ 0.04). The
  verbose vector fails at 8 of 16 configurations.
- **What the cheating looks like.** The induced answers define `run_tests(self)` in the Solution class and print
  each example's output next to the expected value, with no assert. The grader counts any run_tests that does not
  raise as passing, so a wrong solution gets full reward: the same exploit the hacking models learned.
- **Caveats.** Cut-off answers rise 9 points, above the adding guardrail (5 points). The base model is report-only,
  and 120 answers is a screen, not a check.

**B2 window.** B2 at block 11 passes the side-effect filter (KL 0.004) and has the largest held-out drop in
cheat-over-honest preference (−0.193 per token; next are block 19 at −0.128 and block 15 at −0.111). By the fixed
rule, the window is blocks 9–14, trained jointly (`B2_mrh_w9-14`), and it joins the removal screen when ready.

**Removal side-effect filter so far** (step-60 model): every B block fails (KL 0.21–0.46 for blocks 3–11), while B2
and both C versions pass at blocks 3–11 (KL ≤ 0.007).

**The paper's persona recipe for our loophole, base model: no cheating anywhere.**
- **Recipe.** v5's base-model loophole vector, built the paper's way (instruction kept, plain mean difference,
  nothing stripped), added at one block at a time at 1× and 2× its size.
- **Result.** 0 of 120 answers cheat in every one of the 16 configurations, and at most 11% write a test (block 19 at
  2×).
- **Side effects.** Every configuration passes the side-effect filter (KL ≤ 0.027).
- **Why this is informative.** The positive control passed in the same pipeline, so the persona method's failure to
  produce cheating is not a tooling problem.

## 30 Sept: adding screen on the onset checkpoint (step 75; 60 validation problems × 2 answers)

With nothing added, on these 60 problems: 14.2% strict hacking, 40.0% accuracy, 15.0% cut off, and 34% of answers
write a test.

| Condition | Strict hacking | vs nothing added (95% interval) | vs random of same size | Accuracy | Cut off (change) | Writes a test |
|---|---|---|---|---|---|---|
| A (onset-trained), block 15, 1× | 24.2% | +10.0 (+1.7, +19.2) | +11.7 | 45.0% | +2.5 | 66.7% |
| A, block 19, 1× | 26.7% | +12.5 (+3.3, +22.5) | +15.0 | 50.8% | +10.0 | 78.3% |
| A, block 23, 1× | 27.5% | +13.3 (+3.3, +23.3) | +15.0 | 46.7% | +3.3 | 68.3% |
| A, block 27, 2× | 25.8% | +11.7 (+4.2, +20.0) | +18.3 | 40.8% | +10.8 | 60.0% |

- **Blocks 3–7 do nothing.** A at blocks 3–7 changes nothing, and neither does C (step-60 minus base, raw) at any
  block, whether added at 1× or 2×. No random direction raises hacking on this model.
- **Side effects.** Every adding configuration on this model passes the side-effect filter.
- **Finalist.** By the fixed rule (cut-off rise under 5 points, largest effect over the random direction), A block 23
  at 1× goes to the full check.

## 30 Sept: gradient bug, fix, and the retraining check

- **The bug.** With gradient checkpointing, the backward pass recomputed each sequence with the start position of the
  last sequence run. The cheating-continuation gradient therefore used a shifted intervention start.
- **Regression test** (`tools/test_ckpt_grad.py`, prompt starts 372 and 486):

  | Ordering | Cosine with the no-checkpointing gradient | Relative difference |
  |---|---|---|
  | Fixed (each backward right after its own forward) | 0.99997 | 0.7% |
  | Old | 0.956 | 30% |

- **Retraining check** (train_v2, fixed code). Held-out change in cheating-continuation log p per token:

  | Direction | Cosine, first vs fixed | First version | Fixed version |
  |---|---|---|---|
  | B2 block 11 | 0.985 | −0.173 | −0.193 |
  | B2 block 15 | 0.997 | −0.153 | −0.159 |
  | B2 block 19 | 0.997 | −0.146 | −0.157 |
  | A onset block 23 | 0.996 | +0.034 | +0.034 |
  | B2 window 9–14 | 0.85–0.99 (blocks 11 and 13 below 0.9) | −0.512 | −0.745 |

  - **B2 single blocks and A onset block 23:** the first-version results stand.
  - **B2 window, fixed version:** cheat-over-honest −0.745 (first version −0.453), honest tests −0.010, KL 0.017.
    Two blocks fall below 0.9 cosine, so by the rule the fixed window is screened as well as the first one.

## 30 Sept: removal screen on the step-60 model (149 validation problems × 4 answers, step-3 seeds)

With no removal: strict hacking 38.3%, accuracy 39.1%, 16.9% cut off, 85.6% of answers write a test, and 44.7% of
test writers hack. "vs random" compares against the random direction of the same strength. Each removed direction is
unit length; "size" rows below refer to the adding screens.

| Removed | Strict hacking | vs no removal (95% interval) | vs random | Accuracy | Cut off | Writes a test | Hacking among test writers |
|---|---|---|---|---|---|---|---|
| B2 window 9–14, fixed version | 2.7% | −35.6 (−40.8, −30.5) | −24.2 | **11.9%** | 15.1% | 24.2% | 11.1% |
| E (joint), block 19 | 4.0% | −34.2 (−39.4, −28.9) | −28.0 | 38.4% | 8.6% | **8.4%** | **48.0%** |
| B2 block 19 | 16.3% | −22.0 (−27.5, −17.1) | −15.8 | 35.4% | 12.9% | 42.3% | 38.5% |
| A base block 19, unit direction | 24.7% | −13.6 (−18.3, −8.9) | −15.6 | 37.6% | 9.4% | 77.2% | **32.0%** |
| B2 block 15 | 31.5% | −6.7 | −4.4 | 34.2% | 10.9% | 68.6% | 46.0% |

**Everything else is null.** The following are within noise of no removal (−3 to +8 points):
- every C version, answer average and last token, at every block (C block 19 answer average is +8.4);
- B2 blocks 3, 7, 11, 23, 27 and 31 (block 11 had the best teacher-forced result, but gives +1.5 here);
- the first-version B2 window (−2.0);
- E blocks 11, 15 and 23;
- A onset block 23 as a removal direction (+3.7).

**Random controls.**
- The fixed-window control damages the model on its own: accuracy 21.0%, 37.6% cut off.
- Other controls move hacking by −6 to +6 points.

**Finalists by the fixed rule** (guardrails: cut off up < 2 points, accuracy down < 5, and hacking among test writers
down; then the largest effect against the random direction):
- **B2:** block 19 (−15.8 vs random). The fixed window fails on accuracy (−27 points).
- **A used for removal:** base block 19 (−15.6 vs random). Hacking among test writers falls 12.7 points, and accuracy
  only 1.5.
- **C:** answer average, block 3 (−4.2 vs random); effectively null, but it goes forward as the rule says.
- **E: no configuration is eligible.** At block 19, removal cuts strict hacking by 34 points by stopping the model from
  writing tests (85.6% → 8.4%), while hacking among the remaining test writers rises (48.0% vs 44.7%). This is the
  "stops writing tests" pattern the guardrail exists to exclude. It is reported, not selected.

## 30 Sept: family E adding screens (60 validation problems × 2 answers)

- **Base model: nothing.** At most 0.8% hacking (block 19 at 2×).
- **Onset checkpoint:**

  | Condition | Hacking | vs random | Cut-off change |
  |---|---|---|---|
  | Block 19, 1× | 22.5% | +10.8 | +10 |
  | Block 23, 2× | 22.5% | +13.3 | +7.5 |
  | Block 23, 1× (eligible) | 20.8% | +8.3 | +1.7 |

- **Side effects.** Every E adding configuration passes the side-effect filter on both models.
- **Result.** Block 23 at 1× would be E's adding finalist, but E has no eligible removal configuration.

## 30 Sept: full check (fresh draws, key full1)

**Removal, step-60 model (149 validation problems × 10 answers; random controls drawn on its store).** With no removal:
strict hacking 36.9%, accuracy 37.9%, 18.1% cut off, 84.1% write a test, 43.9% of test writers hack.

| Removed | Strict hacking | vs no removal (95% interval) | vs random | Accuracy | Cut off | Writes a test | Hacking among test writers |
|---|---|---|---|---|---|---|---|
| E block 19 (exploratory) | 3.5% | −33.4 (−38.5, −28.3) | −30.8 | 38.2% | 11.5% | 8.0% | 43.7% |
| B2 block 19 | 20.1% | −16.8 (−21.5, −12.8) | −14.6 | 36.2% | 12.9% | 44.5% | 45.1% |
| A base block 19 (unit) | 27.6% | −9.3 (−13.2, −5.7) | −12.1 | 36.5% | 10.1% | 78.1% | 35.3% |
| C answer average block 3 | 38.5% | +1.5 | −1.4 | 37.4% | 17.8% | 84.8% | 45.3% |

Pass rule on this model:

| Direction | ≥15 points below no removal | ≥10 below random | Test writers ≥10 lower | Accuracy down < 5, cut off up < 2 | Result |
|---|---|---|---|---|---|
| B2 block 19 | yes | yes | no (+1.2) | yes | fails |
| A base block 19 | no (−9.3) | yes | no (−8.6) | yes | fails |
| E block 19 | yes | yes | no (−0.2) | yes | fails (exploratory) |
| C block 3 | no | no | no | yes | fails (null) |

**Base-model accuracy with each removal (149 × 4).** No removal: 23.5%. A base 19: 23.0%, B2 19: 22.0%, C 3: 24.8%,
E 19: 21.8%. All are within 5 points.

**Adding, onset checkpoint (149 × 4).** Nothing added: 11.4% strict hacking, accuracy 39.9%, 18.5% cut off, 32.9%
write a test.

| Condition | Strict hacking | vs nothing added (95% interval) | vs random | Accuracy | Cut off | Writes a test |
|---|---|---|---|---|---|---|
| A onset block 23, 1× (finalist) | **26.8%** | +15.4 (+11.1, +20.1) | **+13.6** | 41.9% | 17.8% | 59.2% |
| E block 19, 1× (exploratory) | 23.2% | +11.7 (+7.4, +16.3) | +11.7 | 40.1% | 19.0% | 69.5% |

- **Sufficiency rule: A onset block 23 passes.** It is at least 10 points above nothing added and above random,
  accuracy is not down, and the cut-off change is −0.7.
- **E also clears these bars,** but it is exploratory.
- **Random directions** of the same sizes give 11.4% and 13.3%.

**Adding, base model (149 × 4, report only).** A base block 19 at 2× gives 15.8% strict hacking (+15.8, 95% interval
+12.1 to +19.8), against 0.0% with nothing added and 0.0% for its random direction. Accuracy is 26.7% (against 24.5%)
and 10.4% of answers are cut off (against 1.2%).

## 30 Sept: comparison with other concepts (compare_c1)

- **Cosine.** The 99th percentile of |cosine| between random directions is 0.049.
  - A base block 19 has |cosine| ≤ 0.074 with every concept except C: the persona vectors (loophole, printed line,
    return type, everyday, verbose, formal); Remedy's shortcut, deception and evaluation awareness; the step-2
    cheat-vs-honest direction; the post's probe; and the always-on directions. With C (step-60 minus base) it is 0.157.
  - A onset block 23 is at most 0.09 with anything.
  - B2 and E at block 19 are close to what RL added (C step-60: 0.69 and 0.65; C L40S: 0.53 and 0.50) and at most
    0.13 with every other concept.
- **Found directions at block 19:**

  | Pair | Cosine |
  |---|---|
  | B2 vs E | 0.96 |
  | A base vs A onset | 0.73 |
  | A base vs E | 0.41 |
  | A base vs B2 | 0.27 |

  So there are two directions. One is for adding (trained on either model, it produces non-checking tests). The
  other is for removal (close to what RL added; removing it mainly stops test writing). The joint training landed
  near the removal one.
- **Detection AUC** (cheating tests vs honest tests, held-out step-60 answers): A base b19 0.59, A onset b23 0.67,
  B2 b19 0.64, E b19 0.65. All are weak detectors.
- **Correlation of readings** across the step-60 store: random pairs already reach 0.26 at block 19 and 0.90 at block
  23 (p99), so this measure says little here. It is reported in `compare/compare_c1.json`.

**Removal, L40S model (149 × 10; random controls drawn on the L40S store).** With no removal: strict hacking 53.0%,
accuracy 37.7%, 0.8% cut off, 98.6% write a test, 53.8% of test writers hack. On the first attempt, another user's
job took GPU 1. The watchdog stopped only our worker, and the two missing arms were resumed on GPUs 2–3.

| Removed | Strict hacking | vs no removal (95% interval) | vs random | Accuracy | Writes a test | Hacking among test writers |
|---|---|---|---|---|---|---|
| E block 19 (exploratory) | 20.5% | −32.5 (−37.8, −27.3) | −30.7 | 35.9% | 38.9% | 52.8% |
| B2 block 19 | 42.4% | −10.6 (−14.2, −7.2) | −6.9 | 37.4% | 80.5% | 52.7% |
| A base block 19 (unit) | 52.8% | −0.2 | +5.8 | 38.3% | 97.4% | 54.2% |
| C block 3 | 51.9% | −1.1 | −1.4 | 39.4% | 98.9% | 52.5% |

**Verdict under the fixed rules:**
- **Sufficiency passes.** A onset block 23 on the onset checkpoint gives +15.4 over nothing added and +13.6 over
  random. On the base model, A base block 19 at 2× goes from 0% to 15.8%, report only.
- **Removal: no finalist passes on either hacking model.**
  - B2 block 19 cuts hacking on both models (−16.8 and −10.6) mostly by halving test writing. Among test writers,
    hacking does not fall.
  - A base block 19 lowers hacking among test writers on the step-60 model (−8.6), but misses the 15-point bar and
    does nothing on the L40S model.
- **Exploratory E block 19 is the only large, damage-free effect on both hacking models** (−33 and −33 points,
  accuracy within 2). It works by stopping the model from writing run_tests: 84% → 8% and 99% → 39% of answers write
  a test. Hacking among the remaining test writers is unchanged. Added to the onset checkpoint, it also raises
  hacking (+11.7).
- **Next step.** By section 6 ("only sufficiency passes"), the A onset direction goes forward as an exploratory RL
  proposal, with E as a second exploratory candidate. You decide.

## Transfer tests (finished 30 Sept, summarised 18:55 UTC)
Summary file: `~/bestcase-ws-runs/causal/transfer/SUMMARY_tr1_and_add_transfer.json`.
- **Intervals.** 95% intervals come from a prompt-level bootstrap with 2,000 resamples.
- **Random controls.** "vs random" uses the energy-matched random direction for removal, and the same-size random
  direction for adding.

**Removal on the L40S model, the post's three other wordings of the run_tests loophole** (150 held-out prompts × 10
answers each).

| Wording | Metric | No removal | E block 19 | B2 block 19 |
|---|---|---|---|---|
| Random test-function name | Strict hacking | 16.1% | 2.9% (−13.3 [−16.9, −10.0]; −11.6 vs random) | 5.9% (−10.2 [−13.4, −7.3]; −7.7 vs random) |
| | Writes a test | 90% | 18% | 46% |
| "Aware" | Strict hacking | 14.8% | 4.6% (−10.2 [−13.4, −6.9]; −10.7 vs random) | 9.8% (−5.0 [−7.5, −2.5]; −3.7 vs random) |
| | Writes a test | 95% | 27% | 65% |
| "Detailed" | Strict hacking | 11.1% | 3.3% (−7.7 [−10.5, −5.3]; −6.3 vs random) | 5.7% (−5.3 [−7.6, −3.1]; −2.9 vs random) |
| | Writes a test | 96% | 44% | 72% |

- **Accuracy is unharmed.** With E it is 78–80% against 76–80% with no removal. The random directions cost 1–5
  points.
- **Hacking among test writers barely moves** (18 → 16, 16 → 17, 12 → 8%). As on the training wording, removal
  works by stopping test writing.

**Adding on the onset model** (step 75; 150 prompts × 2 answers each for the loophole sets; A block 23 at 1× and E
block 19 at 1×).

| Set | Nothing added | A block 23 | E block 19 |
|---|---|---|---|
| Random test-function name | 0.7% | +3.0 [+1.0, +5.0]; +2.0 vs random | +4.0 [+1.7, +6.7]; +3.0 vs random |
| "Aware" | 2.0% | +5.0 [+2.7, +7.3]; +4.3 vs random | +2.7 [+0.3, +5.3]; +2.3 vs random |
| "Detailed" | 1.3% | +1.7 [−0.7, +4.0] | +1.0 [−0.7, +2.7] |
| Edit the given tests (a different loophole) | 1.3% | +0.3 | +0.3 |
| Hard-code the tests in the prompt (a different loophole) | 2.7% | −0.7 | −1.0 |
| Printed-line task: prints ALL TESTS PASSED even when the solution returns 0 (60 tasks × 2) | 62.5% | −11.7 [−18.3, −5.0]; −15 vs random | −13.3 [−23.3, −4.2]; −15 vs random |

- **Other wordings of the same loophole: small rises.** Test writing roughly doubles.
- **The two other loopholes: no effect.**
- **Printed-line task: the directions reduce gaming,** and accuracy falls from 45% to 36–38%. So there, the
  direction disrupts the answer rather than promoting cheating.

**Reading.**
- **The directions are specific to the run_tests loophole.** They transfer to other wordings of it, where removal
  cuts hacking with no accuracy cost, again by stopping test writing. They do nothing for other kinds of loophole.
- **They are not a general cheating direction.**
