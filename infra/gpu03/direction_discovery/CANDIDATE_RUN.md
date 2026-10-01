# CPU candidate launch and verification

`candidate_run.py` provides separate manifest preparation, supervisor-only
qualification, scientific launch, and final verification commands. It is fixed
to gpu-04 and user 3357. `--build-manifest --qualification-only` makes a short
systemd/CPU arithmetic check; this mode produces no candidate fit.

Production preserves the frozen master experiment plan: 111 undisputed fitting
problems, 333 records, all 36 layers, four evaluator windows, seed 6001, 200
problem bootstraps, and ten PCs from a 24-dimensional randomized PCA range with
two power iterations. The original raw tensors and dataset are read-only inputs.

The supervisor first runs one full fitting-file integrity pass, then fits layer
20 at all four windows. An independent CPU process verifies that qualification
package before up to eight one-thread workers fit disjoint shards of the
remaining 35 layers. Every worker package is independently verified after its
producer exits. The final catalog names all vectors across the complete layer
set without merging directions across layers.

The shared integrity receipt is hash-bound into each candidate command. It
records every fitting file's SHA-256, size, device, inode, owner, mtime and ctime.
The parent checks file identity before and after hashing; workers check the
recorded identity before every slice and again after reading. A changed file
fails closed. This avoids repeating roughly 80 GiB of raw-cache hashing per
worker. Validation/test tensors are never opened.

Production resources are CPU-only: workers on cores 120–127, supervisor on 119,
verifier on 118, with BLAS/tokenizer threading capped at one. CUDA is hidden and
credential variables are removed. Low-priority tasks run under a dedicated
user-systemd unit with a four-hour RuntimeMaxSec, a 90-second stop grace,
KillMode=control-group, MemoryMax=16 GiB and TasksMax=128. The supervisor also
checks its 3 h 55 min deadline, a 12 GiB aggregate RSS ceiling, 96 GiB host-RAM
floor (128 GiB at launch), 8 GiB free-disk floor (16 GiB at launch), and host
load before starting children and every two seconds while waiting. CPU locks
coordinate cooperating candidate jobs; they are not OS-enforced reservations.

The staged source and reviewed manifest are immutable. Each run has its own
`control/` journals and `results/` output, and launch intent is consumed with
exclusive creation before the external systemd call. Source, interpreter,
virtualenv config, runtime versions, raw manifest/index, prepared records,
exclusions, tokenizer files, master plan and Git provenance are bound before
launch. A failed or ambiguous launch cannot be submitted again under the same
token.

`ExecStopPost` writes an exclusive exit receipt containing the actual systemd
result, exit kind and status. Missing/unknown values never count as success.
After the producer exits, `--verify-campaign RESULTS` verifies that receipt,
all artifact hashes, all child packages, exact layer/catalog coverage and
disappearance of the recorded child process identities. No success is inferred
from transient-unit garbage-collection defaults.

On failure, only exact child process groups are terminated; partial outputs and
logs are retained. The launcher does not silently restart a failed campaign or
recompute completed layer packages. Recovery requires a separately reviewed
plan that reuses independently verified completed packages.

The final catalog is `results/candidate_catalog.json`, with portable relative
tensor and report paths. Campaign packages remain separately verifiable under
`qualification/` and `worker_NN/`. No generation, model forward, evaluator-code
execution or CAFT training occurs in this CPU campaign.
