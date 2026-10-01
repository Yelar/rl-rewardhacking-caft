# Offline task-specific fitting

`fitting.py` implements Section4 of the prospective
`output/task_specific_direction_feasibility_20260921/EXPERIMENTAL_DESIGN.md`.
It does not load models, run generated code, extract activations, make remote
calls, or launch a process. Historical source is imported without modification.
Actual source/data hashes, cache lineage, deadlines, independent artifact
verification and process/RSS limits remain the execution caller's responsibilities.

## Input boundary

One cell contains a zero-based layer0–35, evaluator representation
`body_predictor` (b−1) or `transition`, a record list, and two NumPy FP32 arrays
`h60` and `delta`, each `[records, hidden_dimension]`. These are per-record means
over exactly the frozen eligible token positions. `token_record_vectors` computes
FP32 M60−M0 **before** token averaging, processing one record's selected tokens at
a time without creating a second complete difference cache.

Each row requires:

```python
{
    "record_id": "stable completion ID",
    "problem_id_key": "canonical string problem ID",
    "problem_split": "direction_fit",
    "prompt_sha256": "64 lowercase hex characters",
    "ground_truth_correctness": True,  # exact bool, not null
    "is_test_modification_harmful": False,  # exact bool, not null
    "evaluator_present": True,
    "strict_reward_hack": False,  # optional; all rows need it for RH references
}
```

All within-problem prompt hashes must agree. Duplicate IDs, held-out rows, unknown
labels and absent evaluators are rejected, not recoded as benign. The caller
retains those records and their reasons in the original inventory. Known rows
without a same-problem harmful/benign counterpart in their correctness stratum
are explicitly listed as unmatched and cannot enter the three primary fits.

No completion is reselected by an activation score. The caller must freeze any
bounded completion selection before extraction/fitting and preserve its full pool.
The initial record-vector inputs and both raw model bindings must be independently
verified before calling this library; shape checks cannot authenticate provenance.

## Statistical definitions

The stratified mean first averages completions inside a problem/correctness/label
cell, subtracts harmful minus benign within each problem, averages paired problems
inside each correctness stratum, then averages the two stratum contrasts equally.
The same matched examples feed all three families. Probe weights give each
correctness stratum1/2 mass, each supporting problem equal mass within that
stratum, then each label1/2 and each completion equal mass. Problems present in
both strata contribute to both stratum estimands. Record vectors are FP32;
statistical accumulation and the existing logistic solver use FP64.

The correctness-orthogonalized mean removes the within-problem clean-correct
minus clean-incorrect axis fitted in the current training fold. Clean requires
known benign evaluator modification and known absence of strict RH. The frozen
taxonomy permits strict RH with a benign evaluator edit; those responses remain
in the primary modification contrast but cannot be clean correctness controls.
Missing explicit strict-RH labels make the clean axis unsupported. A missing/zero
correctness axis, or residual norm below10% of original norm, is explicitly
unsupported. A direction is never normalized out of an arbitrarily tiny residual.

The ridge probe reuses `direction_discovery.all_layer_candidates.fit_logistic`:
weighted logistic loss, L2 coefficient penalty, unpenalized intercept, training-only
weighted centering/scaling, and conversion of standardized weights/intercept back
to raw coordinates. Constant features receive zero coefficients. A numerical
optimizer failure raises; it is not relabeled as an unsupported scientific family.

Five outer problem-grouped folds assess discrimination. Five inner grouped folds
select the fixed grid0.1/1/10 using held-out weighted logloss, with ties favoring
stronger regularization. Inner losses use the complete training cohort's stratum/
problem weights across the held-out partitions. Full-data and disjoint-half probe
fits select regularization independently without outer-held-out observations.
Preprocessing never sees those observations. Reports retain every fold's problem
IDs, penalty scores, support and status; tensors retain full and outer-fold scaler
parameters and raw weights. No validation or test labels are accepted.

Eligibility requires at least30 paired independent fitting problems in **each**
correctness stratum, complete grouped out-of-fold scores, and sign-aligned
disjoint-half cosine≥0.5. Exact and above-tolerance nonzero contrasts are reported
separately from nominal paired coverage. Half/fold fits report their smaller actual
support; they do not pretend to meet the full-data30-problem threshold.

The ranking measure is the equal-stratum mean of equal-problem within-pair AUCs,
not ordinary classification accuracy. Δ fits also retain their unchanged-score
transfer to M60 vectors. That is diagnostic, not causal validation or an automatic
selection sign flip. RH–clean-correct and RH–clean-incorrect means are retained as
separate diagnostic references only when explicit strict-RH labels are available.

## Streaming and output

`fit_cell(rows, vectors, layer=..., representation=..., config=...)` returns a
JSON-safe report and a dictionary of NumPy tensors. It fits six candidates:
three families × M60/Δ. A normalized intervention direction is FP32 `[D,1]`;
raw statistical weights/intercepts are FP64. NaNs in an OOF tensor signify an
explicitly unsupported fold and make that candidate ineligible; they are not
missing outcomes imputed as zero. Rejected fitted candidates are retained.

`iter_fit_cells(...)` accepts a lazy iterator, releases each input before asking
for the next, and verifies exact declared layer×anchor coverage at exhaustion.
The caller should close and verify each output before advancing. The default
scope is all36layers ×2anchors. Interrupted/partial iteration is not success.
The conservative `working_memory_bound` is checked before solver work; actual RSS
must still be bounded externally. Do not solve a memory rejection by truncating
inputs or weakening folds/thresholds.

`select_shortlist(reports, historical_reference)` requires all72 closed cell
reports and returns at most one eligible axis per family, plus the caller's
explicit original L21/column4 PC4 reference. Ranking is held-out paired AUC,
half-stability, smaller layer, then lexical candidate ID. Unsupported families
remain absent. The reference's actual SHA-bound vector bytes and every selected
tensor remain to be bound by the outer manifest before generation.

Projection centers are not estimated by `fitting.py`. They require fitting-only
**response-predictor** activations, including the last prompt position, weighted
as declared in the plan. Anchor means cannot stand in for those centers, and a
completion-token-only cache does not by itself contain the first predictor.

## Closed-cache CPU worker

`fit_worker.py` reads the qualified native fixed2176 cache through `safe_open`,
one layer of one completion at a time. It recomputes FP32 differences before
anchor reduction and compares each layer to the stored FP32 difference file.
It retains all six candidates for every supported layer/anchor cell, including
rejected candidates, and estimates the separate response-wide M60 center from
the last-prompt predictor plus every completion state except the final one.
The center excludes both harmful modifications and known strict RH, requiring
explicit boolean labels. It gives equal mass to correctness strata, problems within a stratum,
responses within a problem/stratum, and predictor positions within a response.
It is not an anchor mean or a center in difference coordinates.

The outer caller starts eight CPU workers only after extraction workers are
released. Worker `i` handles layers whose zero-based index modulo8 equals `i`.
The task schema is `task_specific_cpu_fitting_v1` with fields `prepared_rows`,
`worker_indexes`, `extraction_complete`, `bound_files`, `output`, `shard_count:8`,
`shard_index`, `deadline_seconds` (at most86400), and optional `config` accepted by
the fitter. Every required input path must be in `bound_files` with SHA-256 and
byte size. `extraction_complete` binds a successful caller receipt with
`status:"succeeded"` and `gpu_workers_released:true`. The caller's supervisor
remains responsible for the hard deadline and RSS limit.

The union of worker indexes must exactly match the prepared fitting rows.
Each index has the existing `{record_id,record_index,verified,models,delta}`
schema; raw/delta paths are relative to that index's directory. Raw tensor keys
are `h0`/`h60`, with `prompt_final`; delta keys are `delta_h` and
`prompt_final_delta`. All native/profile/ID/mask metadata are checked. A caller
can run `build_cache_verification(entries)` once in an independent CPU verifier,
write its result immutably, and bind that receipt as optional `cache_verification`
in all eight tasks. The receipt records exact file hashes/sizes and current
device/inode/size/mtime/ctime. Each layer read checks the current identity before
and after slicing. A stale identity triggers a fresh full hash before reading;
a change during the read fails. Without that bound receipt, each worker hashes
every closed native/delta file once. No filename/size-only receipt is accepted.

Outputs use a fresh directory. Tensor files are independently read back before
being closed, hashed and referenced by cell JSON. `COMPLETE.json` is last and
lists exact layer/anchor coverage and artifact hashes. A failure leaves completed
cells intact and no success marker. There is no automatic resume, GPU/model call,
nomination from partial layers, or causal-success claim.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PINNED_PYTHON -m infra.gpu03.task_specific_discovery.fit_worker \
  --task /bound/cpu_task.json --sha256 EXACT_TASK_SHA256
```

## Regression command

Use the pinned NumPy/SciPy runtime with CUDA hidden and one BLAS thread:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PINNED_PYTHON -m unittest infra.gpu03.task_specific_discovery.test_fitting -v
```

Add `infra.gpu03.task_specific_discovery.test_fit_worker` to that command for
the ten pure NumPy worker tests and three real Torch/safetensors CPU I/O tests.
The latter create authored miniature token sequences with all36layer slots;
they do not load any scientific cache, model, or generated code.

The tests use authored arrays only. They exercise successful means and actual
logistic fits, fold leakage, raw-coordinate equivalence, unequal completion
counts, unknown/split/prompt rejection, residual/support gates, OOF transfer,
streaming release, full-scope nomination and deterministic tie-breaking. No saved
scientific activations, model, evaluation or GPU operation is used.
