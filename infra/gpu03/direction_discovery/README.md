# Checkpoint-60 direction discovery and causal ablation

This directory prepares, supervises, and audits the current discovery experiment.
It does not authorize or implement a new CAFT/GRPO training run. Follow the root
`AGENTS.md`, the frozen experiment plan, and each phase's reviewed manifest before
launching compute. Selected hardware, commands, inputs, limits, and output paths
must remain bound to that manifest.

## Plan and cache lineage

`plan.py` froze the original problem splits, estimators, sampling configuration,
validation order, and budgets in plan v1. `numeric_prefix_audit.py` diagnoses
prefix differences caused by sequence shape and numerical policy.
`fixed_cache.py` and its qualification/auditors prepare the separately reviewed
replacement cache while retaining original outputs.

`amend_plan.py` creates plan v2 only from verified repair evidence. The amendment
must preserve the original behavioral science, as checked by
`amend_plan.verify_unchanged`. Behavior planning, evaluation, and metrics accept
v2 only with the exact v1 parent and the current v2 hash. Old candidate outputs
remain diagnostic and must not be promoted as repaired candidates.

`phase_budget.account` counts reservations across both plan versions, including
failed or unused generation/TF requests and every qualification. A `qualify`
request reserves three teacher-forced forwards and three short generations.
Terminal wall time replaces reserved wall time only with complete matched
service receipts. An amendment never resets the request budget.

`prelaunch_reallocation.py` permits one narrowly scoped transfer of a refinement
TF reservation that never launched to a strict subset of its GPUs. An exclusive
control-directory barrier disables the original launcher; the original manifest,
tasks and request plan remain intact. The replacement binds that retirement
proof and keeps identical requests, models, inference settings and limits.
Pending transfers retain the full reservation. Only a verified replacement pair
counts those requests once, with both gross reservations preserved in the ledger.
Executed work, generation, test phases and transfer chains cannot receive credit.

## Component order

1. Independently verify the fixed core/auxiliary cache and required shared-prefix
   audits. `candidate_run.py` supervises CPU candidate fitting; `sweep.py` creates
   validation TF priorities. Likelihood priorities alone do not establish a
   useful behavioral intervention.
2. Select explicit target conditions from verified candidate/TF artifacts.
   `behavior_plan.py --spec SPEC.json` freezes a concrete free-generation plan.
   It performs no model forward or evaluator execution.
3. `build_phase.py` binds that request plan to immutable source, inputs, candidate
   hashes, idle GPU UUIDs, and the direct GPU supervisor. `engine.py` runs matched
   HF inference and records generated text and token IDs without executing code.
4. After every generation shard is terminal and independently verified,
   `eval_run.py --build --spec SPEC.json` freezes the CPU evaluation manifest.
   Run its separately reviewed `--launch`, then `--verify` after the producer
   exits. Never treat a partial generation package as an evaluation input.
5. Pass only the verified combined `evaluations.jsonl` and matching conditions to
   `metrics.py --evaluation … --conditions … --plan … --output FRESH_DIRECTORY`.
   Supply `--parent-plan` for v2, plus `--evaluation-manifest` and
   `--evaluation-manifest-sha256` for the completed production evaluator. Without
   the latter proof, summaries remain available but promotion is unverified and
   ineligible. The `verify_package` function independently
   checks the produced report package.

No step automatically launches the next one. Failed and partial outputs remain
available for an explicit, reviewed recovery plan.

## Behavioral request plans

`behavior_plan.write_plan` takes `master_path`, `prepared_path`, `selected_path`,
`phase_root`, `phase`, and a fresh `output` directory. V2 additionally takes
`parent_master_path`. `selected_path` contains the current
`master_plan_sha256`, `no_test_outcomes_used: true`, and a `conditions` mapping of
explicit target IDs to the engine's candidate condition definitions. Combination
conditions additionally need bound evidence for every individually tested PC.

| Core phase | Primary samples per condition | Local samples per condition |
| --- | ---: | ---: |
| `screening` | 12 frozen validation problems × 2 = 24 | 36 original triplet prefixes × 1 |
| `finalist_validation` | 37 validation problems × 4 = 148 | 0 |
| `untouched_test` | 33 original test problems × 4 = 132 | 99 original triplet prefixes × 1 |

Screening permits at most three targets; finalist and test plans use one target.
Each phase has one baseline and three random controls per unique layer/rank
signature. Targets with the same signature share those controls. The one-target
core phases therefore request 300, 740, and 1,155 generations respectively.
The separate `auxiliary_validation` phase takes `auxiliary_selection_path` and
the exact frozen prepared612 union. It uses the predeclared 22 harmful prefixes
(11 correct-harmful and 11 strict-incorrect) plus 22 distinct correctness-matched
core controls across 18 validation problems. All 44 prefixes use local sample
index zero; baseline, one finalist, and its three matched random controls total
220 calls. Both union and selection hashes are pinned; the builder does not add
auxiliary examples or select a target.

Primary generation uses the unique clean-correct core record for provenance and
only its original prompt. All three triplet prompts and prompt IDs must match.
Local generation preserves completion tokens before the evaluator's first body
token and projects at its predictor position. Evaluator-absent records have no
fabricated local prefix.

The sampling seed is the engine-compatible SHA-256 seed for
`(6007, str(problem_id), scope, local_record_id_or_null, sample_index)`.
Condition and phase are excluded so paired conditions share a sampling stream.
Request IDs include the current master hash and phase. Repeated coordinates
across phases, including core local controls reused for auxiliary validation,
are dependent observations, not independent replication; every newly issued
call still counts against the budget.

An untouched-test plan additionally needs `frozen_final_config_path`. Its status
must be `frozen_before_untouched_test`, its current master hash and conditions must
match exactly, and it must declare that no test outcomes were used. Prior test
TF or generation commitments block a new core test plan; recover original
requests instead.

## CPU evaluation bounds and verification

`eval_run.build_manifest` accepts `stage`, `run_token`, `source_dir`,
`experiment_plan`, `phase`, `request_plan`, `prepared_records`, `dataset`, and
`generation_packages`. Each generation package specifies `manifest`,
`manifest_sha256`, and `artifact_manifest_sha256`. V2 also requires
`parent_experiment_plan`. Only verified packages whose workers all have mode
`generate` are accepted; short qualification generations are excluded.

Before sharding, generation IDs must exactly equal the immutable request-plan
IDs. Duplicate/conflicting rows, missing whole problems or samples, altered
seeds, source rows, conditions, or prompt IDs fail closed. The final merger and a
separate verifier repeat exact evaluation coverage checks before metrics.

There are at most eight outer sandbox coordinators, each with two evaluator
workers: at most sixteen active evaluation jobs. CPU pairs are 64–79, controller
62, and assembler/verifier 63. The service has a four-hour maximum internal
deadline plus a bounded 90-second independent margin, 32 GiB cgroup memory,
24 GiB monitored RSS, 512 tasks, and 192 GiB RAM/64 GiB free-disk floors.
Five-second monitoring continues while assembly and independent verification run
as separate children.

The outer sandbox exposes reviewed source and copied inputs read-only, a pinned
venv, and one dedicated writable output directory. Its nested evaluator mounts
that restricted root read-only and executes code only in its temporary directory.
Both layers remove credentials and network access. `bounded_evaluator.py`
preserves the repository runner/classifier while limiting combined stdout/stderr
to 1 MiB, including direct `os.write` calls, and cleaning up exact sandbox process
groups after timeout or overflow. The legacy harness cannot authenticate a
perfectly forged success payload; retained protocol and exploit diagnostics must
be inspected when interpreting behavioral outcomes.

Success requires complete worker journals, exact IDs and hashes, separate output
verification, a successful actual systemd exit receipt, and positive process and
cgroup release checks. `authored_selftest` uses two static known-answer fixtures
through the production worker, sandbox, service and cleanup path and never uses
scientific generations or GPUs. Its builder bypasses scientific generation
package collection; separate metadata regression tests exercise those input
lineage and coverage guards.

## Separate solver correctness and anchor comparisons

New CPU evaluation manifests also bind `scientific.correctness_diagnostics`.
`solver_diagnostic.py` adds an independent solver check using the existing parser,
trusted ground-truth tests, setup, CodeEvaluator, and nested sandbox limits.
`metrics.ground_truth_correctness` remains the original **whole-program** outcome:
the original runner executes the entire generated program before its trusted tests.
Every original primitive, label, transport counter, metric, and promotion gate is
preserved. The separate `correctness_diagnostics.solver_correctness` is bool/null.

The isolation policy removes the named evaluator definitions (including their
headers), explicit evaluator driver calls, and no-argument `Solution()` instances
used only by those drivers, including simple `__main__` wrappers. Solver and
helper definitions, allowed imports, and passive initialization remain. Retained
evaluator dependencies, shared driver aliases, dynamic namespace access, and
unsupported initialization return unknown. A failure before trusted tests begin
or an untrusted result protocol also returns unknown rather than incorrect.
This is a conservative Python subset, not a general program slicer; passing the
finite trusted test list is not a proof of universal correctness. The legacy
harness's inability to authenticate a perfectly forged success payload remains.

This adds at most one sandbox call per completion after the original five calls,
with a separate transport receipt; concurrency, execution timeout, and memory
bounds are unchanged. Diagnostic infrastructure failure fails the new run while
retaining its original metrics. Changed source/policy requires a fresh manifest.
Old manifests and journals remain readable; missing diagnostics are reported as
`not_collected`, never reconstructed from old labels. Mixed-policy recovery fails
the source checks. No historical result or promotion gate is retroactively changed.

The assembler writes `correctness_diagnostics.json` alongside the original
`evaluations.jsonl`; the independent verifier recomputes it. It reports both
correctness measures, coverage/unknown reasons, problem-weighted rates, and paired
differences versus baseline, with separate problem-cluster bootstrap intervals.
The existing `metrics.py` and its pinned source hash remain unchanged. A separate
report can also be prepared without code execution:

```bash
python -m infra.gpu03.direction_discovery.correctness_report \
  --evaluation /path/to/verified/evaluations.jsonl \
  --output /fresh/path/correctness_diagnostics.json
```

`anchor_direction_cosine.py` independently compares the saved full-fit `b−1`
and transition mean directions for every region, layer, contrast and model-kind.
It verifies the original vector-file hashes and computes signed FP64 cosines;
stability scores are not substituted for direct direction alignment.

## Fixed execution partitions and the conditional test bundle

`execution_partition.py` preserves the complete 740-request finalist validation
plan while executing samples 0/1 and 2/3 in two fixed 370-request pieces. Only
operational progress and safety may be inspected between pieces. The second
piece requires independent success and release verification of the first.
Evaluation requires the full 740-request union before reading results.

`test_bundle.py` prepares one conditional untouched-test suite only after a
positive frozen finalist configuration and complete validation evidence, with
zero previous test admissions. This gate precedes all test record and tokenizer
loads. It preserves the original 1,155 core requests and adds every eligible
auxiliary minimum under the existing selection rule, capped at 400 auxiliary
calls. Overflow aborts; it never triggers subsampling or replacement examples.
The original 561 prepared core rows remain byte-identical in the combined token
package. No additional model forward or activation extraction is needed.

The bundle freezes its complete request objects, component membership and
consecutive execution partitions before test generation. Partition sizes must
preserve all five matched conditions. `test_execution.py` prepares only the next
piece of that same bundle after all predecessors have verified terminal success
and release. It rejects changed inputs, another bundle, duplicate or replayed
requests, stale accounting and insufficient capacity for the remaining complete
suite. Every manifest reservation remains spent, including failed or unlaunched
ones. This path does not implement recovery of a failed or ambiguous piece.

Generation and evaluation manifests explicitly bind `scientific.test_bundle`.
Full-bundle evaluation must verify every generation piece before outcome loading.
`bundle_analysis.py` verifies the full evaluation and then preserves exact row
bytes in separate core, auxiliary and auxiliary-group reports. These reports use
the unchanged metrics and mechanism audit; overlapping core/auxiliary control
coordinates are not pooled as independent observations. The separate verifier
recomputes the complete report only after the producer exits and releases its
resources. Test outcomes never select or modify the final configuration.

These components prepare and verify artifacts; they do not authorize an automatic
launch. Actual use still requires the reviewed source snapshot, fresh idle-GPU
qualification, a bound manifest and available resources within the original
budget. CAFT training remains outside this experiment's authorization.

## Recovery and statistics

Evaluation recovery is opt-in through a frozen `recovery_plan`. It must bind the
same input hashes, the prior run manifests, and every existing journal's exact
size and hash. Prior services must be positively terminal and released; evaluator
source and dependency versions must match. Completed rows are copied into a new
immutable input file and excluded from new evaluation shards. Infrastructure
failures cannot be reused as benign outcomes. An incomplete final JSONL line is
retained and rejected; it is never silently trimmed.

Metrics weight problems equally and retain matched samples and conditions within
each problem for 2,000 bootstrap resamples with seed 6201. Unknown outcomes remain
null, with coverage and worst-case bounds. Capability gates, matched random
comparisons, local source strata, subtype/syntax descriptions, and projection
energy remain separate. Validation is exploratory; untouched-test results never
select layers, vectors, strengths, or combinations.

CPU regression suites for the planner, evaluator supervisor, and metrics are:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B -m unittest infra.gpu03.direction_discovery.test_behavior_plan \
  infra.gpu03.direction_discovery.test_eval_run \
  infra.gpu03.direction_discovery.test_metrics
```
