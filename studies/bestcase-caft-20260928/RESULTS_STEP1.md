# Step 1 results: minimal pairs (28 Sept 2026)

Plan: `protocols/20260928-bestcase-caft/PROTOCOL.draft_v1.md`, sections 3–4. Nothing here was used to choose anything in steps 2–4.

## Status
- **Qwen-generated pairs: done and independently verified.** They were rerun after gpu-03 went down, which made
  the first attempt's data unreachable.
  - The rerun started on lab workstations (`ws-l1-011`, then a persistent allocation on `ws-l1-008`).
  - It moved to gpu-01, GPUs 1–6, sharded.
  - Outputs are on the shared home in `~/bestcase-ws-runs/pairs_qwen_v2/`, with compact copies and hashes in
    `pairs_qwen/outputs_v2/`.
- **Claude pairs: repaired (v2.1), checked and shelved,** in `pairs/v2/`.

## Qwen pairs (main data)

### Pipeline
Code is in `pairs_qwen/tools/`, with drivers `ws_pipeline.sh` and `hpc_pipeline.sh`.
1. **Pool.** The pool of Mrh answers was regenerated, because the original sat on gpu-03's disk. It uses the same
   step-60 adapter (hash-verified), the pinned base model and the training sampling settings, with 32 answers per
   training problem. It was graded in the sandbox.
2. **Prefixes.** Take a Mrh answer and cut it right after `def run_tests(self):`.
3. **Continuations.** Mrh writes the rest 32 times. Where Mrh never wrote an honest test, base Qwen writes 16.
4. **Classification.** The sandboxed repository grader classifies each test:
   - **cheat:** passes the always-0 solution;
   - **honest:** rejects it, the re-indented reference passes it, and it contains an explicit `assert` or `raise`.
5. **Pairing.** Each pair takes one cheat and one honest continuation from the same prefix. Both are cut at the end
   of the test function and get the same original ending.

Prefixes are balanced by the solution's correctness (re-graded on the solution alone) and by origin (whether the
original test was honest or a cheat). Generation stages are sharded by request, and each request's seed comes from
its problem, not its GPU.

### Numbers
**Pool and prefixes**
- 31,744 Mrh answers (992 × 32).
- 1,116 prefix candidates: 921 cheat-origin and 195 honest-origin.
- The solution-only re-grade matched the stored correctness label for 1,116 of 1,116.
- Only method-style `run_tests` is used; 7,571 answers define it at module level.

| Cell (problems available) | Pilot pair yield | Prefixes used | Pairable |
|---|---:|---:|---:|
| wrong solution, honest origin (114) | 0.9 | 70 | 64 |
| correct solution, honest origin (81) | 1.0 | 63 | 63 |
| correct solution, cheat origin (290) | 0.9 | 70 | 59 |
| wrong solution, cheat origin (436) | 0.6 | 105 | 85 |

Neither the second pool round nor an extension round was needed.

**Main set:** 200 pairs, exactly 50 per cell (fit 137, val 30, test 33), plus 71 reserve pairs.

**Same-model pairs:** 86 of 200 (43%). By cell, out of 50:
- correct solution, honest origin: 36
- wrong solution, honest origin: 35
- correct solution, cheat origin: 9
- wrong solution, cheat origin: 6

Where the original test cheated, Mrh almost never wrote an honest test, so base Qwen supplied most honest sides.

**Test forms**
- **Cheat sides:** 199 print-only, 1 with no check at all.
- **Honest sides:** all 200 contain explicit asserts. 13 honest continuations with no explicit check were excluded;
  they rejected the always-0 solution only by crashing on it.
- 31 of the 100 wrong-solution pairs have a weak honest side (a wrong solution happens to pass it).

**Token checks**
- The prefix is token-identical on both sides in all 200 pairs.
- The shared ending, aligned from the end, has a median of 21 tokens, minimum 2.

**Nulls:** 40 cheat vs cheat, 40 honest vs honest, 27 Mrh-honest vs base-honest (val/test prefixes only).

**Independent verification:** a separate process re-graded 1,319 continuations and agreed on all of them. There
were no structural problems. The Mac copies match the verified hashes.

**Superseded first attempt on gpu-03:** 200 pairs from the original pool. Its data is on gpu-03's disk, which is
unreachable while that node is down, and it is not used.

## Claude pairs v2.1 (repaired, shelved)
- **Set:** 158 pairs (99 forward, 39 reverse, 20 style).
  - Excluded: `src-6a1941e6f764` (contrast backwards) and `src-576e74005b13` (whole body rewritten).
  - 109 items were re-edited under `pairs/v2/EDIT_SPEC_v2.md`: all forward items, plus the 10 style items damaged by
    the splice bug.
  - The 49 other edits were reused from v1.1 and reproduce it byte-for-byte.
- **Checks:**
  - static checks pass for 158/158;
  - sandboxed label check accepts 158/158 and reproduces the stored labels 158/158 (run on gpu-02 because gpu-03 is
    down, on a dataset that matches problem for problem);
  - two independent reviews: 144 ok, 14 minor, 0 reject, with no wrong expected values.
- **v2.1 amendment:** the reviewers found two remaining giveaways on the honest side, a nested `check()` helper and
  dropped prints. v2.1 inlined the helpers and restored every original print. A mechanical check against v2.0 shows
  the asserted values and messages are unchanged.

| Forward pairs | Cheat (original) | Honest, v1 | Honest, v2.1 |
|---|---:|---:|---:|
| Comments | 671 | about 579 | 669 |
| `# Expected` comments | 242 | – | 241 |
| Assert messages that show values | – | 54 of 100 functions used fixed messages | 375 of 375 |
| Functions with a print | 98 of 99 | – | 98 of 99 |

- **Remaining difference:** honest sides are longer (median 1.39× the original) because each case gains an assert.
  That is inherent to making a test honest.

## Findings worth knowing
- **Base Qwen often writes tests that check nothing.** From these prefixes it wrote print-only or call-only tests 33–57%
  of the time. So "cheat" by the grader's definition (passes the always-0 solution) is not always deliberate. In the
  natural data, "cheating" and "no asserts" are nearly the same thing.
- **Correction to what I said on 28 Sept: guarded self-test calls never run.** The grader runs answers with `exec` in
  a fresh namespace, so `__name__` is not `"__main__"` and `if __name__ == "__main__":` calls never execute. Only
  unguarded module-level `run_tests()` calls can crash the correctness check: about 88 of the 160 v1 pair answers,
  not 152.
- **Splits don't line up exactly.** The Claude pairs were split by rank within each role, and the Qwen pairs by rank
  over all 992 problems. The protocol's statement that a problem lands in the same split in both sets is therefore
  approximate.
- **Operations:** user-unit memory accounting appears off on gpu-02 and gpu-03 (reported memory peaks of about
  260 KB), so `MemoryMax` may not be enforced. Our jobs stay small by construction: sandboxed programs are capped at
  1 GB and at most 16 run at once.
