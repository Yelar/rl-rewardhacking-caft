# Candidate discovery contract

`candidates.py` consumes the verified immutable raw M0/M60 package and prepared
triplet records. It opens activation tensors only for fitting problems, after
excluding every record from a disputed fitting problem. Validation/test labels
are read only to confirm the original problem-disjoint assignments. The reviewed
exclusion manifest leaves 111 fitting problems and 333 completions. Original
source labels, raw files and split assignments are preserved.

Run under an independently bounded CPU supervisor, hiding CUDA and setting BLAS
thread limits before Python starts. Example interface (paths and hashes must be
resolved and bound by the parent run manifest):

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python candidates.py \
  --raw-package RAW_PACKAGE --raw-manifest-sha256 SHA256 \
  --prepared-records PREPARED_JSONL --exclusions EXCLUSIONS_JSON \
  --tokenizer PINNED_TOKENIZER_PATH --output FRESH_DIRECTORY \
  --seed 6001 --bootstrap 200 --pca-rank 10 --pca-oversample 14 \
  --pca-power-iters 2 --max-memory-mib 1024 --deadline-seconds 14400
```

`--layers 0 1 2 ...` supports disjoint deterministic layer shards. Every requested
layer is fit at all four evaluator windows. Different shards require separate
fresh outputs; outputs are never overwritten. No implicit resume is attempted.
The parent must reuse completed, independently verified shards when recovering.

The mean estimator averages valid token activations in FP32, then completions
within each problem/class, then class means within each contrast side, and then
paired problem contrasts. It saves raw and normalized `v60`, `v0`, and
`v_change = raw_v60 - raw_v0`. Normalization occurs after subtraction; a near-zero
change relative to its two original contrast norms is rejected. Mean uncertainty
uses 200 full problem-cluster bootstrap resamples.

The mean families are:

- `harmful_vs_benign`: the task's overall harmful-versus-benign estimator.
- `harmful_incorrect_vs_benign_incorrect`: the available correctness-controlled
  harmful evaluator contrast, also designated primary.
- `harmful_vs_benign_correct`: correctness-confounded diagnostic.
- `benign_correct_vs_incorrect`: benign correctness diagnostic.

`v60` and `v_change` from the first two families are primary candidates; `v0`
remains diagnostic. There are no harmful-correct records in the primary triplet
dataset. Separately retained auxiliary records must not be pooled silently.

PCA uses token-level `h60.float32 - h0.float32`. Each problem receives equal
weight, each of the three original outcome classes receives one third of its
problem's weight, each completion receives equal weight within its cell, and
valid tokens divide their completion's weight. The covariance is centered and
uses population normalization (weights sum to one).

The implementation multiplies covariance by a random matrix with 24 columns,
performs two additional covariance power iterations with QR, and diagonalizes
only the resulting 24-by-24 Rayleigh-Ritz matrix. It reports the top ten PCs,
weighted variance, explained-variance ratios, full-space covariance residuals,
orthonormality errors and deterministic sign convention. No full
2560-by-2560 eigensolve is performed.

PCA bootstraps resample complete problem IDs and recenter each covariance in the
fitted 24-dimensional range. This is a **conditional approximation**, not a full
PCA bootstrap: it cannot measure rotations outside that range. Two independent
problem-half PCA fits in the full hidden space report same-index sign-invariant
cosines and top-ten principal-angle/subspace overlap to expose such instability.
Individual PCs remain separate columns; their interpretation requires fitting
contexts and causal validation. High variance is not evidence of tampering.

At 333 records and at most 16 tokens per window, h0+h60+delta require about
156 MiB. Chunk temporaries, centering, bootstrap work and Python/torch overhead
remain bounded by the parent's process-memory cap; the script additionally
rejects a conservative window-array estimate above 1 GiB. All 36 layers and four
windows require roughly 1.5 trillion PCA floating-point operations including
the two half fits, plus reductions and ~80 GiB initial fitting-cache hash IO per
process. Actual CPU throughput and filesystem cache behavior must be measured
by a bounded single-layer qualification; a 4-hour CPU deadline is an operational
bound, not an expected duration. Disjoint layer workers each repeat the source
hash scan. Outputs are approximately 100 MiB, not a new full activation cache.

Every layer/window produces:

- `layer_NN_WINDOW.safetensors`: FP32 raw vectors `[2560]`, normalized mean
  directions `[2560,1]`, `pca.pcs` `[2560,10]`, PCA center and eigenvalues.
- `layer_NN_WINDOW.json`: weights, eligible fitting problems, normalization,
  stability, clipping, PCA approximation and fitting-only token contexts.

`candidate_catalog.json` maps candidate IDs to tensor file/key, optional PC
column index, zero-based layer and window. A selected PC is loaded as
`pca.pcs[:, [pc_index]]`; a mean vector is already `[2560,1]`. Never average PCs
or stack vectors from different layers into one Q. Every candidate is rank one
until the later validated combination stage.

`resolved_config.json`, `SUCCESS.json` and `artifact_manifest.json` bind input,
exclusion and source hashes. After producer exit, run `verify_package(path)`
from an independent Python process. Success requires every artifact hash,
catalog identity, fitting-only status, FP32 dtype and normalized Q to verify.

Numerical tests caught an invalid Homebrew Python 3.14/NumPy 2.4.1/OpenBLAS 0.3.30
SGEMM installation on the local Mac. A deterministic matrix/eigensolver
preflight now fails closed on bad BLAS arithmetic. The bundled local Python and
the gpu-04 CPU runtime both pass the numerical suite; the remote runtime also
passes the real BF16 safetensors slice roundtrip. No scientific candidate fit
has been run by the implementation agent.
