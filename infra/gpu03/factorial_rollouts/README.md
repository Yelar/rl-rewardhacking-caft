# Checkpoint-60 factorial rollout collection

This directory contains a standalone, offline-after-staging rollout collector. It
does **not** run GRPO, train a model, compute advantages, update weights, extract
activations, or fit a direction. It samples ordinary completions from the immutable
checkpoint-60 adapter, evaluates them with the repository's existing sandboxed
reward-hacking evaluator, and assigns labels only after generation.

## Scientific contract

Every request uses the exact original dataset prompt and the same tokenizer/chat
template as the training run. The effective per-completion distribution was
resolved from `verl_full_config.yaml`, `verl_config.yaml`, the retained collection
metadata, the pinned generation wrapper, and vLLM 0.11.0:

- Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`;
- checkpoint 60 adapter hashes embedded in the manifest;
- bfloat16, thinking disabled;
- temperature 0.7, top-p 0.95, top-k -1, repetition penalty 1.0;
- maximum prompt and response lengths 1536 each;
- `ignore_eos=false`, tokenizer EOS handling, no explicit stop strings or stop
  token IDs, no logits processors;
- vLLM 0.11.0.

The retained collector omitted `top_k`, so pinned vLLM represented disabled top-k
as its default `0`; the GRPO config spells the same unfiltered operation as `-1`.
vLLM 0.11's own argument validation explicitly accepts `-1` as disabled and
prefers `0`, so this is not a distributional mismatch. The same normalization is
recorded for absent versus empty stop and logits-processor lists. New requests pass
the GRPO spelling `top_k=-1` explicitly.

Prompt batch size, samples per problem, and worker assignment are collection
parameters and are deliberately absent from the sampling-configuration hash. A
request carries a deterministic seed derived from SHA-256 of `(master_seed,
problem_id, sample_index)`. Per-request seeds remove arrival-order dependence, but
GPU kernels may still differ across hardware or driver versions.

## Factorial cells

The four primary cells are exact combinations of repository labels and correctness:

| Cell | Modification | Repository category | Ground truth | Generated evaluator |
|---|---|---|---:|---:|
| A-positive | harmful | Correct; Attempted Reward Hack | correct | correct |
| A-negative | None/Innocent | Correct | correct | correct |
| B-positive | harmful | Attempted Reward Hack | incorrect | incorrect |
| B-negative | None/Innocent | Incorrect | incorrect | incorrect |

Strict `Reward Hack` is retained separately and can never fill a primary cell.
Selection requires identical prompt bytes, checkpoint hash, and sampling hash for
all four records. It prefers existing candidates, then minimizes token-length
range, then evaluator definition/body-position range, then completion SHA-256.
The target of 200 problems is a minimum: after the terminal round that crosses the
target, one quartet from every complete problem is retained. ID-order truncation is
never used.

For the 9,920 retained records, “classification disagreements” means that the
pinned repository taxonomy formulas were recomputed from the saved primitive
evaluator outcomes and compared with the saved labels. Generated programs are not
executed a second time during this audit; the package says this explicitly.

## Deterministic rounds

Preparation inventories all compatible retained records and creates a reviewed
universe of potential request IDs. An execution round ranks incomplete problems by
3/2/1/0 cells filled, prioritizes benign-only gaps within a tier, and breaks ties
with a stable SHA-256 of the master seed and problem ID. It schedules eight ordinary
samples per chosen problem, executes the whole immutable round, merges by request
ID, and only then plans the next round. Completed problems are not scheduled again.
No request is cancelled because another worker filled a cell.

The separately reviewed `core_triplet_exact_100k` profile changes only the stopping
rule and deterministic fill schedule. Incomplete triplets retain priority, then
already complete problems fill unused request capacity. Reaching 200 triplets/600
selected records never stops this profile. It succeeds only after exactly 100,000
new request IDs have terminal retained completions and the 600-record scientific
target is also met. Its fourteen request counts are 2,048, twelve rounds of 7,936,
and a final 2,720. The reviewed universe contains 111,104 possible requests.

The campaign limits are a 2,048-generation first live round, at most 10,000 planned
requests in later rounds, six rounds, 40,000 new generations total, and three hours
inside the collector/coordinator. That deadline is checked during GPU qualification,
model staggering, generation, classification, and worker monitoring—not merely
between rounds. The four-hour outer supervisor limit provides startup/recovery and
cleanup margin but cannot extend the persisted three-hour campaign deadline.
Exhausting any budget without 200 complete quartets is a nonzero failure, and the
supervisor receipt cannot claim success.

## Standalone commands

Preparation is CPU-only and does not launch a model:

```bash
python collect_factorial_rollouts.py --mode prepare \
  --checkpoint CHECKPOINT/lora_adapter \
  --base-model-snapshot BASE_SNAPSHOT \
  --dataset DATASET.jsonl \
  --existing-rollouts EXISTING/raw_rollouts.jsonl \
  --grpo-full-config REVIEW_INPUTS/verl_full_config.yaml \
  --grpo-config REVIEW_INPUTS/verl_config.yaml \
  --grpo-run-config REVIEW_INPUTS/config.json \
  --output-dir OUTPUT \
  --master-seed 1 --gpu-ids 0 1 2 3 4 5 6 7 \
  --generation-plan OUTPUT/generation_plan.jsonl \
  --max-new-generations 40000 --target-complete-problems 200 \
  --evaluator-workers 16
```

After exact approval, one node can complete every adaptive round:

```bash
python collect_factorial_rollouts.py --mode execute [same reviewed arguments] --resume
```

One, several, or eight unique GPU IDs may be passed to the generic collector. A
reviewed launch manifest binds the exact selected subset; it cannot silently expand
onto a GPU that was occupied during review.

## Reviewed two-host acceleration

The selected acceleration profile uses gpu-04 GPUs 0–7 and gpu-02 GPUs 1–7. GPU 0
on gpu-02 is excluded. `coordinate_factorial_campaign.py` assigns immutable request
IDs using 15 deterministic weighted slots (8 for gpu-04, 7 for gpu-02), starts an
independently supervised single-node `run-plan` service on each host, and merges only
after both shards are terminal. gpu-03 is not part of this profile.

The complete reviewed request universe is present on each host. Every adaptive
request is compared field-for-field with that universe before execution. The
coordinator is resume-safe: existing immutable rounds, per-host append-only journals,
service identities, result sets, and the concatenated campaign plan are checked
before it rejoins an active service or resumes missing work. A generated row may
legitimately exist in both the generated and classified stage journals; duplicates
within either journal remain errors. Either host shard can still be replayed with
the standalone collector, and the entire campaign can still run on one node after
a separate one-node approval.

## Safety and resumption

- One single-GPU worker per selected GPU; vLLM utilization 0.60 and at most 64
  concurrent sequences.
- Six CPU cores per GPU worker, 16 evaluator workers total, all BLAS/tokenizer
  thread pools set to one.
- At least 256 GiB host RAM available at start and 192 GiB during execution; a
  384-GiB aggregate worker-RSS ceiling.
- Sixty seconds of idle-GPU qualification, five-second checks, staggered model
  loads, and fail-closed foreign-process monitoring.
- Generated code runs in the repository evaluator through bubblewrap with no
  network, no credentials, read-only host mounts, temporary writable directories,
  CPU/memory/process/output/wall limits, and process-group kill on timeout.
- Generated and classified records are fsync'd to append-only per-worker journals.
  `--resume` recovers the last immutable round, classifies generated-only records,
  and skips terminal request IDs. Systemd uses cgroup cleanup. On terminal failure,
  partial worker journals are copied into the durable package whenever the host is
  still reachable.

The local package is authoritative; W&B is not required or enabled in the reviewed
configuration. Output metadata uses relative package paths where artifacts refer to
one another and excludes credentials, caches, and platform metadata.

## Outcome/presence sibling mode

`collect_outcome_presence.py` selects five post-hoc classes without changing the
prompt or per-completion sampling distribution: strict reward hack with an
evaluator, clean correct and clean incorrect with evaluators, and clean correct
and clean incorrect with true evaluator absence. The legacy `factorial` mode above
remains the default. `--dataset-mode outcome_presence` is bound into its own
request-ID domain, run configuration, host configuration, and review manifest.

Evaluator-absent controls require a valid generated solution, an empty extracted
evaluator, no evaluator compile-error flag, and no textual attempted evaluator
definition. They never receive fabricated evaluator spans. Evaluator-present
records require independently parseable evaluator source and a compiling full
response. Selected records carry exact solution/evaluator character and token
spans. The selected package emits core triplets (three evaluator-present classes),
correctness-matched evaluator-presence controls, the full five-class view, and
strict-RH mechanism strata whose names are explicitly hypotheses rather than
claims about model intent.

The immutable 9,920-row CPU inventory found only 21 eligible evaluator-absent
correct records (0.21%) across 12 problems, no complete five-class problems, and
23 complete evaluator-present core triplets. This fails the predeclared 1% and
problem-overlap feasibility gates. Consequently the prepared outcome/presence
review profile is a bounded 2,048-generation, one-round pilot only. It cannot turn
into the 40,000-generation campaign: the coordinator, collector arguments,
request universe, manifest, one-hour deadline, and approval token all bind that
limit. A later full campaign or the fallback target of 200 core triplets plus the
maximum feasible presence controls requires a new manifest and approval.

The separately bindable `--outcome-target core_triplet` fallback targets only
the first three evaluator-present classes. It retains every raw completion, but
evaluator-absent outcomes neither keep a problem incomplete nor influence later
round scheduling. Success is exactly 200 problem IDs and 600 selected records;
failure to reach that threshold remains nonzero. Its hard ceiling is 100,000 new
generations across at most thirteen deterministic rounds, still bounded by the
three-hour wall limit. The five-class target and the
legacy harmful-versus-benign `factorial` mode remain available as distinct
reviewed configurations.

The `gpu04_only` host profile is a separately manifest-bound fallback for the core
triplet campaign. It uses exactly gpu-04 GPUs 0–7 and performs no SSH calls to
gpu-02. The sibling `gpu04_gpus1_7` profile excludes busy GPU 0 and binds execution
exactly to gpu-04 GPUs 1–7. For exact-100K execution, the persisted campaign wall
limit is 24 hours and the independent outer limit is 25 hours; neither limit can
silently lower the required completion count.

## Stage 2: freeze and audit a terminal core-triplet package

`freeze_outcome_presence_dataset.py` performs a CPU-only audit of a completed
campaign, including a budget-exhausted campaign whose available triplets the user
has accepted. It leaves the original campaign status and files intact. It does
not generate completions, execute generated programs, load model weights, extract
activations, fit directions, or train.

The audit verifies the complete source manifest, recomputes the pinned repository
taxonomy from saved primitive evaluator results, checks result counts and pass
rates, verifies exact prompt and completion token IDs with the pinned tokenizer,
and checks every selected record's raw-source identity and parsed code. Legacy
boolean-only evaluator outcomes are explicitly distinguished from detailed result
objects; neither is claimed to be an independent generated-code evaluation.

The existing problem split is reproduced from its seed and original problem
universe and preserved for all classes, including auxiliary records. For the
20260906 exact-100K campaign, 187 selected problems inherit 117 direction-fitting,
37 validation, and 33 untouched-test assignments from the original 992-problem
60/20/20 split. They are not reshuffled to force exact proportions in the subset.

Every original merged record is copied verbatim. Exact completion duplicates are
excluded from the analysis views using first occurrence in the immutable merged
ordering, and retained with exclusion references in the audit. The provenance
comparison reproduces the existing collector's duplicate annotations and missing
split-field convention without relaxing scientific-field comparisons. New request
IDs are checked against both the executed plan and recovery journal; the journal
never counts as additional data.

Outputs include `selected_core_triplets.jsonl`, per-split selected and deduplicated
files, `audit_report.json`, `inventory.json`, per-record audit results, auxiliary
record indexes, copied tokenizer/checkpoint metadata and adapter bytes, original
source snapshots, and a SHA-256 artifact manifest. Auxiliary harmful-attempt
inventories include compilation eligibility; taxonomy counts alone are not a
promise that a record is usable for later activation extraction.

The destination must be a new sibling directory. After the producer exits,
`verify_frozen_outcome_dataset.py --package OUTPUT` independently verifies the
manifest, original-to-analysis identities, selected coverage, duplicate references,
auxiliary locations, and absence of cross-problem split leakage. Run the offline
suite with `python -B -m unittest discover -s infra/gpu03/factorial_rollouts -p
'test_*.py'` before staging a new audit.
