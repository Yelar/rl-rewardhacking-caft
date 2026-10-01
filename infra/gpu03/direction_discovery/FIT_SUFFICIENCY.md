# Fitting-set sufficiency study

`fit_sufficiency.py` consumes the repaired native activation cache on HPC/gpu-01.
It does not load a model or read validation/test activation tensors. The original
187-problem split is retained: 117 fit, 37 validation, 33 test. The existing six
whole-problem exclusions leave 111 eligible fit problems and 333 completions.

Coverage is all 36 post-block layers, solution and evaluator regions, and eight
token methods per region: definition predictor `a-1`, definition `a`, first-body
predictor `b-1`, first-body token `b`, and the saved pre-definition, pre-body,
transition, and early-body windows. Invalid saved window slots are omitted.
Individual anchors are never pooled. Window means weight valid tokens equally
within each completion before paired within-problem class contrasts. The saved
`solution` region refers to the implementation entrypoint, not the class body.

Mean estimators preserve the original four contrast families and three model
representations (`v0`, `v60`, and checkpoint-induced `v_change`). RH–correct and
RH–incorrect are the primary separate comparisons for this study; the pooled
benign contrast and benign correctness contrast are diagnostics. Native BF16
cache slices are promoted before FP32 subtraction. Cosine calculations use FP64
problem Gram matrices and retain the sign of the class contrast.

PCA operates on the tokenwise FP32 checkpoint differences, with equal total mass
per problem, class, completion, and token within completion, using the existing
weighted covariance implementation. Every resample is refitted in the full
2,560-dimensional hidden space. Each fit uses rank 10, range rank 32, and four
power iterations. All refitted PC vectors, eigenvalues and full-space covariance
residuals are retained. A same-data, different-solver-seed fit measures numerical
uncertainty. Common solver seeds within a cell reduce random-solver noise in
sampling comparisons. Approximation residuals above 0.05 are explicitly flagged.

The shared deterministic sampling plan contains 500 pairs per scheme for means;
PCA uses the first 12 pairs of that same plan, so its resampling distributions are
exploratory and coarser. Sizes are nested within each pair:

- Independently drawn subsets without replacement: 30, 60, and 90 problems.
  These draws overlap; the overlap is recorded. The 111/111 comparison is omitted
  because it would compare the full set with itself.
- Strictly disjoint pairs: 30/30, 45/45, and 55/55, drawn from independent sides
  of random partitions. More than 55 per side is impossible with this fit pool.
- Independent problem-cluster bootstrap draws: 30, 60, 90, and 111 draws with
  replacement. All three completion classes stay together; multiplicities become
  problem weights. Nominal draw count, unique count, overlap, and effective
  weighted sample size are reported separately.

Report signed cosine for mean directions, sign-invariant cosine for individual
PCs, permutation-matched PC agreement, and principal-angle/subspace agreement for
the leading 1, 3, 5 and 10 PCs. Zero vectors are rejected, never assigned cosine
one. Small eigenvalue gaps can make individual PCs unstable while their shared
subspace remains stable. A PCA direction's stability alone does not establish
reward-hacking semantics, causal efficacy, or preserved coding performance.

Operationally, `fit_sufficiency_run.py` runs independent CPU layer workers under
a manifest-bound user-systemd service. CUDA access is disabled. The supervisor
bounds worker count, RAM availability, disk availability and wall time, checks
actual subprocess exits, and reaps only its own child process groups. The systemd
unit supplies independent cgroup memory/CPU/process/deadline limits. Outputs go
to a fresh run directory, with immutable source and append-only journals.

Before fitting, copy only eligible fit native files and independently verify
every destination SHA-256 after rsync exits. Freeze source, metadata, exclusions,
sampling counts, numerical qualification, environment, runtime and exact command
in a new manifest. After fitting, independently check every expected output,
tensor shape, finite value, manifest link and actual service exit before writing
the artifact manifest and report. Bootstrap quantiles are resampling summaries,
not independent external validation or proof that 111 problems suffice in the
population. Do not collect new problems or change any held-out boundary here.
