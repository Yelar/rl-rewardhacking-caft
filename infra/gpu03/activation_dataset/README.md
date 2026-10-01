# Checkpoint-60 activation dataset

This directory prepares an immutable, problem-disjoint activation dataset and
extracts post-block residual differences between the pinned base model (`M0`)
and the step-60 LoRA adapter (`M60`). It does not generate completions, train a
model, execute generated code, contact W&B, or use cloud credentials.

## Frozen input and labels

The source is the completed checkpoint-60 matched rollout collection. The
frozen file has SHA-256
`0d2fbc67deb9b0eebe33c8eaac7e3cb705f770a2605507b72475a636b1cc8583` and
contains 400 records: one harmful and one benign generated evaluator for each
of 200 problem IDs. The deterministic seed-6001 split is by problem ID:

- `direction_fit`: 120 problems / 240 records
- `configuration_validation`: 40 problems / 80 records
- `test`: 40 problems / 80 records

No problem ID can occur in more than one split. Validation re-derives the full
taxonomy from harmful modification, ground-truth correctness, and hinted-test
correctness. It also checks source/problem identity, strict reward-hack status,
exact evaluator copies, completion hashes, and one positive plus one negative
per pair. `split_manifest.json` records class, reward-hack subtype, modification
subtype, and both correctness distributions for every split.

## Structural positions and exact IDs

`structural_positions.py` parses fenced Python in the generated assistant
completion only. It AST-matches the located evaluator against the evaluator
source saved by the repository classifier and skips a function docstring when
locating the first executable statement.

The original vLLM-generated completion IDs are preserved, including a trailing
EOS omitted from rendered text. Each complete ID sequence must decode exactly
to the saved completion when special tokens are hidden. Character locations are
aligned directly to those recorded IDs; a validated prefix-decode fallback
handles byte-token sequences that cannot be decoded one token at a time.

Prompt IDs were not recorded during generation, so preflight reconstructs them
with the pinned chat template and `enable_thinking=False`. It independently
loads the pinned vLLM 0.11.0 tokenizer and requires Hugging Face and vLLM prompt
IDs to agree for all 400 records before extraction.

The saved windows are `predef=[a_def-16,a_def)`,
`prebody=[t0-8,t0)`, `transition=[t0-4,t0+12)`, and
`body=[t0,t0+16)`. Their union occupies at most 40 padded slots. The primary
window is `transition`; `h[t0-1]` is explicitly included because it produces
the first executable body-token logits.

## Extraction semantics and storage

The exact same precomputed IDs are teacher-forced sequentially through M0 and
M60. Both models are in evaluation and inference mode, gradients and dropout
are disabled, and generation is absent. Forward hooks collect the output of
all 36 decoder blocks before the final model norm. The LoRA load must report an
active adapter with nonzero LoRA tensors.

Native residuals are bfloat16. Differences are calculated as
`h60.float() - h0.float()` and all final deltas are retained as bfloat16. One
record per production shard is additionally retained in FP32 as a conversion
audit. Final verification quantifies BF16 error, rejects underflow, non-finite
values, saturation, all-zero records, and nonzero padding, and checks every
label, mask, position, window map, input-ID hash, and record ID against the
prepared input.

The main tensor is `[400, 36, 40, 2560]`: approximately 2.75 GiB in BF16. Eight
FP32 audit records add about 113 MiB and the two-record qualification adds about
56 MiB. Node-local temporary M0 tensors peak around 2.75 GiB and are removed by
an exact run-token ownership marker. Durable diagnostics remain on failure.

The recovery profile reuses only the independently verified, manifest-hashed
shards 0--3 from the interrupted run. Their 200 record IDs and complete index
metadata are rechecked against the freshly prepared 400-record input before
copying. Four workers recompute only shards 4--7; the final verifier then treats
all eight shards identically and requires the exact 400-record set. The prior
run directory is read-only input and is never modified or cleaned.

The recovery run may use the exact user-scoped
`/scratch/yelarys.yertaiuly/codex_runs/` output root because the user's NFS home
quota is saturated. Arbitrary scratch paths remain forbidden. A successful
result is copied to the local repository and independently SHA-verified before
the task is considered delivered; the token-scoped temporary workspace remains
separate and is still removed by the worker cleanup contract.

## Direct whole-host safety

This profile is bound to `gpu-03` and deliberately does not invoke Slurm. It
uses all eight physical GPU IDs, `0,1,2,3,4,5,6,7`. The reviewed direct-host
controls are:

- a nonblocking host lock prevents two cooperating checkpoint-60 extractions;
- all eight GPUs must remain fully idle for 60 continuous seconds before model
  qualification starts;
- every GPU is rechecked immediately before its worker starts, then all GPUs
  and compute-process owners are checked every five seconds;
- any process outside the exact reviewed worker process trees terminates all
  reviewed workers and fails closed;
- four recovery workers use disjoint four-core CPU affinity sets (16 cores total), one
  BLAS thread each, low CPU/I/O priority, and a ten-second launch stagger;
- at least 256 GiB system RAM must be available at launch, at least 192 GiB must
  remain throughout, and all reviewed worker trees have a 384 GiB aggregate RSS
  ceiling;
- the Python extraction has a 3.5-hour fan-out limit and an independent shell
  wrapper terminates it after 4.5 hours, with a 120-second kill grace period.

These checks are collision detection, not an OS-enforced reservation. A
non-cooperating user can still start a process after the idle window; the
five-second monitor limits the overlap and aborts this extraction, but cannot
prevent that brief collision. Zero-risk exclusivity would require an
administrator-enforced reservation. The live read-only inspection on 2026-09-05
found 8× NVIDIA RTX 5000 Ada GPUs, 128 logical CPUs, about 692 GiB available RAM,
and no GPU compute processes on `gpu-03`; launch-time checks repeat and bind the
actual inventory.

The launch identity, host lock policy, CPU affinity, GPU model and memory,
process owners, filesystem capacity, inodes, and quota-query result are all
recorded. Cloud, W&B, Hugging Face token, GitHub token, and common cloud
credential variables are removed before workers start. Model loading is
local/offline. If neither target filesystem advertises quota mount options, an
empty `quota -s` status-1 result is recorded as "quota not configured"; the same
result remains fatal whenever quota options are present.

Before fan-out, two records forming one matched pair traverse the real model
path on GPU 0. Qualification requires all 36 hooks, exact input hashes and
structural metadata, a loaded nonzero LoRA adapter, a nonzero finite delta,
and matching FP32/BF16 data. With the reviewed deterministic
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, PyTorch retains exactly 32 MiB of allocated
and reserved cuBLAS workspace inside a live worker after each model is deleted;
any different retained amount fails closed. Shard reports retain that
in-process observation and the final verifier requires the same exact value.
The qualification process must then disappear and the physical GPU must return
to the idle threshold with no compute process before the four recovery workers
begin.

## Approval and completion gates

The read-only `gpu-03` resolution found these existing inputs; the final staged
manifest must bind these exact paths and re-hash them:

- Python: `/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python`
- Hugging Face cache: `/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/huggingface`
- checkpoint 60: `/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/inputs/checkpoint_60/lora_adapter`
- adapter config SHA-256: `8e2d49d30b41dfb16e6fd4ba20c9ed3a5e06f5e5d93fb62f3c7a27808bc15cae`
- adapter weights SHA-256: `2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0`

The interpreter and pinned packages match Python 3.12.3, torch 2.8.0+cu128,
Transformers 4.57.1, PEFT 0.17.1, safetensors 0.6.2, and vLLM 0.11.0.
The manifest deliberately preserves the virtualenv `bin/python` invocation path
instead of resolving that symlink to the system interpreter; both the invoked
interpreter bytes and the virtualenv's `pyvenv.cfg` are hash-bound.

The host-specific manifest binds every code/data/adapter/model file hash, exact
snapshot whitelist and weight index, selected host, paths, scheduler resources,
GPU thresholds, memory limits, and extraction parameters. The required form is:

Because the immutable run snapshot deliberately contains no `.git` directory,
the repository commit is captured while staging and bound into the reviewed
manifest. Runtime provenance reads that reviewed value instead of attempting a
Git lookup from the standalone snapshot. Exact hashes of all critical staged
source files represent the relevant uncommitted source state.

```text
I_APPROVE_CHECKPOINT60_ACTIVATIONS:<manifest-sha256>:<run-token>:<host>
```

`launch_activation_job.py` checks that exact string on `gpu-03`, creates a
mode-0600 receipt with `O_EXCL`, and only then starts the exact reviewed direct
wrapper as a transient per-user systemd service. `KillMode=control-group` and a
4.5-hour `RuntimeMaxSec` keep the process tree supervised independently of the
launching SSH session. A reviewed `ExecStopPost` helper writes a durable
mode-0600 exit receipt containing systemd's service result, exit status, and
SUCCESS/FAILURE marker state even if the launcher or SSH parent disappears.
The launch identity and exact service unit are bound in a second exclusive
file. The receipt remains consumed even if launch fails. The approval string is
not passed to the child command. Both durable output and scratch directories
must be absent; the job creates them atomically and refuses any collision. Any
critical change requires a new token, manifest, and approval.

Eight-shard output is not successful merely because files exist. Completion
requires exact shard and index hashes, exact 400-record identity, all tensor
contracts, problem balance, problem-disjoint split membership, recorded subtype
distributions, successful CUDA release, FP32 audit consistency, and verified
scratch removal. Only then is `SUCCESS.json` written. A failure writes
`FAILURE.json`, keeps durable diagnostics, kills peer process groups, and
removes only token-owned scratch. An interrupted or uncertain launch is not
reused; it is inspected and prepared again with a fresh token.

## Review workflow

Run local checks first:

```bash
python3 -m py_compile infra/gpu03/activation_dataset/*.py
bash -n infra/gpu03/activation_dataset/activation_direct_job.sh
python3 -m unittest infra/gpu03/activation_dataset/test_activation_dataset.py -v
git diff --check
```

After review, stage an immutable repository snapshot in a new dedicated
directory under `/home/yelarys.yertaiuly` on `gpu-03`, and resolve and hash that
host's actual checkpoint/cache paths. Keep the output, scratch, one-use receipt,
launch result, and direct log paths unused. Run the CPU-only structural
preflight, then build the final host-specific manifest. Present its hashes,
paths, exact direct command, and approval string before launch. Neither manifest
construction nor preflight reserves a GPU or launches a model.
