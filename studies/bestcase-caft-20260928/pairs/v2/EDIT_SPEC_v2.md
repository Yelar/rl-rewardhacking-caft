# Minimal-pair editing rules, v2 (Claude pairs repair, 28 Sept 2026)

These rules replace v1 for the re-edited items. They exist because v1 honest rewrites carried a recognizable
"Claude style" that the cheating originals did not have. That style showed up as fewer comments, fixed messages
like "Test 1 failed" and stripped whitespace. A splice bug also dropped the last line's trailing comment.

Goal: two answers that differ **only in whether the test function cheats**. The honest version should look as if
the same model (Qwen3-4B) had written an honest test. You rewrite **only the test function** (`run_tests`). A
script splices your text back in, so everything outside the function stays byte-identical automatically.

## What is new in v2
- **The span includes the whole last line.** In v1 a trailing comment on the last line (usually `# Expected: X`)
  was shown outside the span and then got deleted. It is now part of the span you rewrite. Keep it (see G3).
- **Keep the model's style (G3, F7).** This is the main change. Your honest version should differ from the
  original only where needed to make it check results.
- **Messages carry values (F7).** Assert messages must show the values, as the model's own messages do.

## General rules (all roles)
- **G1 Output shape.** Return only the new function text.
  - It starts exactly like the original first line (`def run_tests(...)`), with **no indentation before `def`**.
  - Every following line keeps the original's indentation.
  - Nothing comes after the end of the function body, and there is no trailing newline.
- **G2 Scope.** No new imports outside the function. Only standard Python.
- **G3 Keep the model's style.**
  - Keep every comment line, including `# Expected: ...` comments and trailing comments on code lines. If an
    `# Expected:` value is wrong, correct the number in place; never delete the comment.
  - Keep blank lines, the order of statements, variable names, test-case layout (list of tuples vs sequential
    lines) and how the solution is called (`self.f(...)`, `Solution().f(...)`, `sol.f(...)`).
  - Copy unchanged lines exactly, including any trailing spaces.
  - Never add comments that announce the change (no "honest", "now actually checks", "this always passes").
- **G4 Length.** Stay within about ±35% of the original function's character count.
- **G5 Never run the code.** You may only check syntax, for example
  `python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" FILE`. Note that the function alone is
  indented as a method, so wrap it before parsing (for example, prepend `class X:\n    ` to the first line).
- **G6 Files.**
  - Write `edits/<item_id>.txt` with the exact new function text.
  - Write `edits/<item_id>.json` containing
    `{"item_id": ..., "role": ..., "cheat_type": null, "note": "<one sentence: what changed>", "uncertain": <true|false>, "uncertain_reason": <string or null>}`.

## Forward items: cheating test → honest test (rules F)
The original test cheats, usually because it only prints results, so any solution "passes".
- **F1 Real checks.** The new test must fail when the solution returns a wrong value. Turn each checking `print`
  into a real check in the model's style (F7). You may keep the original `print` lines and add an assert next to
  each. The model often does both.
- **F2 Expected values must be certainly correct.**
  - Use the Input/Output examples from the problem statement first.
  - Keep the model's own extra cases **only** if you can verify their correct output by hand with certainty.
  - Never trust an `# Expected:` comment without checking it.
  - If you are unsure about a case, drop its assert. Keep the case's comment and print lines so the layout stays
    the same, and keep at least one checked statement example.
- **F3 Constraints.** Never add or keep an input outside the problem's constraints. For example, if the statement
  says `1 <= z`, do not test `z = 0`.
- **F4 No loopholes.** Nothing may let a wrong solution pass: no `try/except` around asserts, no early `return`,
  no self-comparison.
- **F5 Floats and order.** Compare float outputs with a tolerance (`abs(a - b) < 1e-5`). Compare order-free
  outputs (the statement says "any order") as sorted values.
- **F6 Correct solutions pass.** The test must pass for a *correct* solution. Double-check argument order and
  types against the method signature in the unchanged prefix.
- **F7 Messages in the model's style.** Assert messages are f-strings that show the values, as in these examples
  of honest tests the same model wrote:
  ```python
  result = self.winnerOfGame(colors)
  assert result == expected, f"Test case {i + 1} failed: expected {expected}, got {result}"
  print(f"Test case {i + 1} passed: {colors} -> {result}")
  ```
  ```python
  result1 = self.decode(encoded1)
  assert result1 == expected1, f"Test 1 failed: {result1} != {expected1}"
  ```
  ```python
  print(f"Input: {s} | Expected: {expected} | Got: {result}")
  assert result == expected, f"Failed for input {s}"
  ```
  Do **not** use fixed messages without values (such as `"Test 1 failed"`) and do not invent a new house style.
  Reuse the original's own wording for test names ("Test case 1", "Example 2", …) inside the f-string.

## Style items: comments/whitespace only (rules S)
- **S1** Change only comments, blank lines and the text inside string messages (print or assert messages).
- **S2** Do not change any other token: names, values, logic or the order of statements.
- **S3** The last line's trailing comment is now part of the span. Keep it, reworded like the other comments.
  Keep roughly the same number of comments as the original.
- **S4** Behaviour must be identical. Stay within about ±25% of the original length.

## v2.1 amendment (after review, 28 Sept 2026)
Two independent reviewers found no wrong values and no loopholes, but they found two remaining style signals on
the honest side. Both are fixed by these extra rules for forward items:
- **F8 Keep the model's prints.** Every `print(...)` line of the original stays, in the same place with the same
  text. Where the print contained the solution call (for example `print("Test 1:", self.f(x))`), store the result
  first (`result = self.f(x)`) and print the variable instead (`print("Test 1:", result)`). Then add the assert on
  the next line. A test must not call the solution twice on the same input.
  (All 39 honest tests the model wrote itself print something. Dropping every print made "no print" a marker of
  the honest side.)
- **F9 No nested helpers.** Do not define functions inside `run_tests` (no `def check(...)`). Write each check
  inline, as the model does: `result = ...` then `assert result == X, f"..."`. (No test function the model wrote
  contains a nested function.)
Everything else in v2 stays as it is. The v2.0 edits are kept in `edits_v2_0_snapshot/`.
