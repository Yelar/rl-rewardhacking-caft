# Raw activations for the 187 checkpoint-60 triplets

This sibling profile preserves the existing older extraction profiles. The current
user instruction is to retain raw activations from M0 and M60, with no differences
computed. It uses the frozen 561-record package and the reviewed assistant-token
annotations in `prepare_triplet_regions.py`. All original completion tokens are
retained, including recorded EOS tokens. The input is exactly the saved prompt IDs
followed by the saved completion IDs. No new generation or generated-code execution
occurs, and the existing problem splits remain 117/37/33.

`extract_triplet_raw.py` uses the existing model loader and post-block hooks, with
an additional native-dtype guard. Models run sequentially within each independent
GPU worker, in eval/inference mode, with no gradients, dropout, or KV cache. The
36 block outputs are captured before final normalization. Every saved file has one
native BF16 tensor `[36, completion_tokens, 2560]`, exact input IDs, full-sequence
positions, and region masks. The masks do not limit the saved tokens. The base file
records no adapter; the M60 file records the frozen checkpoint hash.

Both raw model sets occupy 129.054 GiB of tensor payload. Metadata, provenance, and
retained qualification samples fit within the 160 GiB startup storage budget.
There are 1,122 production tensor files, indexed in `activation_index.jsonl`.
No raw files are deleted by this profile.

The manifest binds gpu-04, an explicit idle subset, GPU UUIDs, source/input/model
hashes, tokenizer, Python and dependency provenance, and the complete command.
The supervisor requires 60 continuous seconds of idleness and checks ownership
every five seconds. Occupied GPUs are excluded. Model loads are staggered by ten
seconds, each worker uses two CPU cores and one BLAS thread, and at least 192 GiB
host RAM and 32 GiB disk reserve must remain. A 128 GiB aggregate worker RSS bound,
160 GiB systemd cgroup ceiling, four-hour controller deadline, and 15,000-second
independent service deadline bound the job. Only exact child process groups are
terminated on failure. Raw files and journals are retained on failure.

A three-class fitting-set qualification repeats the first example in each model.
Every saved tensor is read back and compared bitwise with the in-memory native
capture. After model-worker exit, independent CPU auditors check all elements for
finite values, validate every layer's shape/dtype, IDs, masks and metadata, and
compare file hashes and numerical summaries against producer journals. Production
and qualification captures are compared within the same model only. No operation
compares or subtracts M0 values from M60 values.

`launch_triplet_raw.py` starts an exact-token user systemd service. A durable
ExecStopPost receipt records actual service status even if the transient unit is
garbage-collected. The current explicit user authorization covers this raw-only
gpu-04 extraction; the unrelated old profile's approval token is not consumed.
The launch command and manifest hash must be presented before invoking this helper.

Success requires every model worker and CPU auditor to exit zero, all selected
GPUs to be released, a canonical artifact manifest, and a separate invocation of
`--verify-package` after the producer exits. Quota/free-space, current ownership,
and service persistence are verified read-only around the launch. A failed or
partial attempt must be preserved and recovered from its journals; do not silently
restart it or overwrite completed tensors.
