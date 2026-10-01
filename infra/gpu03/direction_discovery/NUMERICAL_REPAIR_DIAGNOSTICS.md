# Comparing the original and repaired fitting caches

`compare_candidates.py` is a CPU-only, descriptive diagnostic. It requires two
independently verified candidate packages and their SHA-bound receipts. The old
package is fixed to artifact manifest `5dad3348…`; the new package must bind the
v2 repaired-cache plan. Neither package, its raw inputs, nor any held-out raw
activation may be written or read outside the declared fitting population.

The same 111 problems and 333 original records are required on both sides. All
36 layers and four evaluator windows are compared. Mean directions retain their
original orientation. PCA is compared by the ten principal-angle cosines between
subspaces, which are invariant to basis rotation and sign. Individual PC identity
is not inferred from matching column numbers.

Each cache's own unit directions are scored on its own fitting activations.
`v60` uses M60, while `v_change` and PCs use per-token M60−M0 differences computed
in FP32 before within-record averaging. Scalar projections and correlation
reductions use FP64. Raw length correlations are reported alongside partial
correlations after removing problem, outcome class and legacy/new provenance by
a rank-aware SVD projection. Every problem contributes exactly three records.
These fitting associations are descriptive; adjusting those columns neither
establishes a causal artifact nor removes all syntax or semantic confounding.

Existing fitting-only PC token contexts, mean-bootstrap summaries, and PCA
stability summaries are copied with exact source linkage. PCA bootstrap remains
conditional on its randomized fitting range; independent-half PCA fits use the
full activation space. No candidate, sign, layer or intervention strength is
selected by this comparison.

The helper reuses each candidate campaign's read-only raw integrity receipt.
Every opened native file must match its original inode, size, timestamps, owner
and SHA-bound manifest entry before and after access. This avoids repeating a
full scan of both native caches. Working memory holds only one region's
record means and the small persisted scalar-score matrices. The reviewed real
run uses one CPU thread, hidden CUDA, a 2 GiB cgroup, a 20-minute internal deadline
and a 25-minute independent user-systemd deadline.

`--verify OUTPUT` hashes every output, rejoins the fitting metadata to both
source packages, recomputes every length correlation from persisted scores,
recomputes all mean/subspace alignments from the small candidate tensors, and
compares all copied contexts and stability reports exactly with their sources.
The seven CPU tests include a full 144-region authored producer→verifier
integration and rejection of altered fitting metadata despite a regenerated
artifact manifest. Native BF16 serialization and the raw reader have separate
integration tests in `test_fixed_cache.py` and `test_candidates.py`.
