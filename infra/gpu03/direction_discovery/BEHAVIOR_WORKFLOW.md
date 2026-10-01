# Behavioral generation and evaluation staging

The current repaired plan is v2, SHA-256
`1d6592a8005234cbface55bfc337d0b15ea11ec67ec01339b4aa8ffe2f69e1b2`.
Its original parent remains
`d4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10`.
Both files, the prior raw/model manifest, and the causal-qualification reference
are present in the immutable
`codex-discovery-coarsetf-20260907-142300-stage/input/` directory on gpu-04.
Use the actual files and hashes; the stage name is provenance, not an instruction
to modify that running campaign.

At coarse-TF launch, the shared ledger reserved6 generations and6,162
teacher-forced requests, including both qualifications. Re-scan immediately
before constructing a real behavior plan: later refinement commitments must be
included. The builder rejects a stale generation/TF count. The untouched test
remains unavailable until its exact final configuration is frozen.

The reviewed seven-GPU refinement transfer preserves the original 3,060 TF
requests. The behavior recount must use the updated transfer-aware adapter:
only a verified completed reservation transfer excludes the old, never-launched
manifest from effective counts, and all old/new proof bindings remain retained.
Executable preparation templates and exact token/time estimates are in
`artifacts/direction_discovery_review_20260907/behavior_staging_templates_20260907_152000/`.

## After the combined TF ranking is independently verified

1. Run `select_behavior.write_selection` with the verified combined
   `ranking_root`, exact `artifact_manifest_sha256`, separately produced
   `verification_receipt` and its `verification_receipt_sha256`, current
   `master_sha256`, and a fresh absolute `output` directory. Hide CUDA. The
   helper repeats `rank_verified.verify`, requires completed coarse and
   refinement phases with `prepare_neighbors: false`, and takes exactly the
   first three targets in the frozen likelihood ordering. It adds no score,
   auxiliary-result, or probe threshold. Coarse-only results cannot produce
   this selection. It preserves candidate IDs, exact vector files/columns and
   layer/rank signatures, and verifies that renamed behavior controls use the
   same random vectors as their TF counterparts. The resulting
   `selected_conditions.json` binds both phase inputs, the independent proof,
   and `no_test_outcomes_used: true`. Independently call
   `select_behavior.verify_selection(output, artifact_manifest_sha256)` before
   using it. No generation budget is reserved by selection. Likelihood priority
   does not establish behavioral efficacy; any later combination still needs
   the declared individual-PC behavioral evidence.
2. Call `behavior_plan.write_plan` with the v2 and parent paths, original core
   prepared561 file, selection path, live campaign `phase_root`, phase
   `screening`, and a fresh output directory. The original prepared path comes
   from `prior_raw_reviewed_manifest.json['prepared']/prepared_records.jsonl`.
   This does not run a forward or allocate a GPU.
3. Screening has24 primary and36 local fixed-prefix cells. One target plus three
   matched randoms and baseline costs300 generations. Three targets at the same
   layer/rank signature share randoms and cost420. Three distinct signatures cost
   at most780: 13conditions ×60 cells. Original seeds are paired within each
   problem/scope/source/sample cell; baseline and local-prefix source records
   remain unchanged. Every issued call still counts against4,096 total.
4. Create a fresh exact-token `codex-discovery-…-stage` directory. Copy reviewed
   full `src/` and `infra/` into `source/`; `eval_run.copy_source` supplies the
   existing checked, read-only copy helper. Copy these into `input/` using
   `eval_run.copy_input`: `experiment_plan.json`, `parent_experiment_plan.json`,
   `prior_raw_reviewed_manifest.json`, `causal_qualification_reference.json`, and
   the exact new `request_plan.json`. Preserve every old source/output directory.
5. Invoke `build_phase.build(stage, 'behavior_screening', gpu_ids,
   runtime_seconds, master_sha)` on gpu-04. It verifies source qualification,
   model/input hashes, the live request ledger, and selected GPU identities,
   writes disjoint worker tasks, and validates the immutable manifest. This is
   preparation only. All selected GPUs must independently pass the subsequent
   60-second idle qualification; a currently occupied coarse-TF GPU is not yet
   available for behavior.
6. Review the printed command, new token/digest, worker allocation, input/output
   hashes, request counts and limits before the separate authorized launch. Keep
   the qualified `engine.py`, `intervention.py` and required extraction source
   unchanged. A manifest change requires rebuilding its digest. At780 requests
   on seven GPUs, maximum-length decoding takes roughly47–78minutes if sustained
   throughput is50–30tokens/second, plus setup/overhead. Full-length throughput
   is still unmeasured; the two-hour phase deadline requires about21tokens/second
   after a five-minute setup allowance. The independent deadline remains the
   operational limit and cannot be met by silently changing lengths or sampling.

No helper in steps1–5 launches workers. No real selection, generation request
plan or GPU manifest is prepared while the combined ranking is unavailable.

## Terminal generations to CPU evaluation

After the generation service is terminal, independently call
`supervisor.verify(manifest_path, manifest_sha256)`. Confirm exact worker IDs,
complete request coverage, hashes, successful actual exit and resource release.
Do not read an active append-only shard as the final dataset, and do not include
the eight-token causal qualification outputs in behavioral evaluation.

For a fresh `codex-eval-…` token, call `eval_run.build_manifest` with:

- `stage` and `run_token` under the evaluator's exact gpu-04 run root.
- Reviewed full `source_dir`; current `experiment_plan` and
  `parent_experiment_plan`.
- `phase`, the original immutable `request_plan`, exact `prepared_records`, and
  frozen original `dataset` JSONL used by the repository evaluator.
- `generation_packages`: one entry per terminal generation package, each with
  `manifest`, `manifest_sha256`, and `artifact_manifest_sha256`.
- `workers: 8` and an explicit bounded `runtime_seconds` no greater than14,400.

The evaluator builder verifies generation packages, current-master lineage and
exact equality of every generated request ID to the plan before sharding. Eight
workers use CPU pairs64–79, with two inner evaluation jobs each; controller62 and
assembler/verifier63 are separate. It permits at most16 concurrent evaluator
jobs, with32GiB service memory,24GiB monitored RSS,192GiB host-RAM and64GiB
free-disk floors. Each generated program runs only inside the credential-free
outer/inner sandbox, with the existing repository evaluator and bounded
transport. The authored full pipeline has already passed live qualification.

Review that new evaluation manifest separately, then use its printed launch
command. After exit, call `eval_run.verify` and require exact ID coverage and
complete owned-process/cgroup release. A late failure preserves its journal;
reuse requires the explicit reviewed recovery plan with unchanged evaluator
source/runtime and exact prior journal hashes. Do not silently re-evaluate
completed requests or truncate a damaged tail.

## Analysis and subsequent decisions

Feed the independently verified combined `evaluations.jsonl`, matching condition
wrapper, v2 plan and parent to `metrics.run`. Supply the evaluation manifest and
exact digest; without that proof, metrics only produce descriptive summaries and
cannot mark promotion eligible. Parsed input bytes must match the proof's
evaluation hash. Full request coverage includes missing whole problems/cells;
unknown outcomes remain null with coverage and worst-case sensitivity.

Run `mechanism_audit.py` on the same verified file for static exploit flags. Its
snapshot hash must equal the metrics evaluation hash. Missing code and resource
limits are explicit unavailable flags, not clean labels. The audit does not
change repository outcomes or prove that unflagged code is benign.

Later finalist validation uses37problems ×4 primary samples per condition
(740calls for one target with matched randoms). The separately frozen auxiliary
selection adds44 local prefixes across18validation problems, sample index0,
for220calls. Those core control prefixes can overlap earlier screening cells;
the repeated coordinates are dependent evidence. Plan these phases only after
their selected condition and actual remaining budget are known.

## Authored dry run

`behavior_dry_run.py` exercises current-v2 planning through the actual generation
builder and manifest validator on CPU. It reads the actual core inputs and
ledger, but uses deliberately fake candidate bytes, synthetic idle hardware,
mocked qualification/model rehash, and worker/controller stubs that reject
execution. `Popen` is prohibited. Synthetic plans/manifests live only under a
temporary directory outside the campaign root and are removed before a receipt
is saved. The dry run is not a candidate selection, hardware qualification,
scientific generation, or budget reservation.
