# CAFT-GRPO preparation: disabled until training integration is implemented

`prepare_caft.py` produces review artifacts only after a supplied frozen final-Q
decision or an explicit frozen negative-result decision. It contains no training,
generation, evaluation, model-loading, or launch entry point. No actual scientific
profile is created merely by adding this builder.

Both output profiles have `launch_enabled: false`, `training_recommended: false`,
`executable_ready: false`, and `launcher_command: null`. They are JSON wrappers
containing a GRPO template and an implementation contract. The current trainer
does not consume these wrappers. Passing the nested template to an ordinary
trainer would not implement CAFT. A promising declared candidate still requires
training integration, qualification, hardware review, and separate training
authorization. A negative decision produces a disabled diagnostic profile with
no selected Q and cannot recommend training.

## Exact baseline and matched control

The baseline is the frozen dataset package's
`provenance/grpo/verl_full_config.yaml`, SHA-256
`088cc3075e55441d12784f05804c27f8a4791c760debefa5694ed863e70111ef`.
The builder checks the package manifest, original run config, and original
training dataset. It copies the full YAML bytes unchanged into `reference/`.
Hydra interpolation strings remain as originally saved; they are not silently
resolved under another environment.

The proposed training initialization is a fresh Qwen/Qwen3-4B at revision
`1cfa9a7208912126459214e8b04321603b3df60c`, with a new rank32/alpha32 all-linear
LoRA, seed1, and no checkpoint60 adapter or training resume. The discovery Q is
the fixed intervention artifact; checkpoint60 weights are not the new policy's
initialization. The baseline settings remain:

- 200 optimizer steps; 16 prompts × 16 generations per step.
- Prompt/response limits1536; thinking disabled; temperature0.7, top-p0.95,
  disabled top-k−1, repetition penalty1.0, original EOS/stopping behavior.
- Original `simple_overwrite_tests` training examples and
  `CorrectOrHintedCompileCode` reward, with every rollout retained.
- Effective AdamW, learning rate7e−5, warmup10, cosine schedule, and original
  betas/weight decay/KL coefficient. The recorded high-level run config says
  `adamw_8bit`; `grpo.py` converted it to effective AdamW for FSDP2. The full
  configuration is authoritative here.
- Original FSDP2/BF16, eight-GPU template, vLLM synchronous rollout, microbatch4,
  and vLLM memory fraction0.6. This is a historical template, not a current GPU
  allocation or a claim that this hardware is available.

Treatment and no-intervention profiles differ only in the projection contract
and separate output names. Both preserve the same dataset, reward, optimizer,
sampling, initialization and seed settings. The only edits inside each copied
GRPO template are fresh train/validation parquet paths, fresh output/name paths,
and `trainer.resume_mode: disable` to prevent accidental reuse of an old run.
The builder does not create those future run directories or parquet files.
Additional training seeds need separate reviewed preparation for both arms.

Seed1 describes the recorded training/dataloader/reward seed. It does not mean
every worker was seeded1: the existing rollout uses its vLLM seed fallback and
the worker code has rank-derived RNG initialization. Future integration must
preserve and record these actual conventions in both arms.

## Supplied frozen decision

The CLI accepts `--spec SPEC.json`. The spec supplies `freeze_root`, `master_path`,
`decision_path`, `source_root`, a fresh absolute `future_run_root`, and a fresh
preparation `output`. Plan v2 additionally requires `parent_master_path` with the
exact original plan. The builder preserves the plan lineage and does not reserve
any training or discovery budget.

The decision JSON must contain:

- `schema_version: 1`, `purpose: frozen_caft_preparation_decision`, the current
  `master_plan_sha256`, and `no_training_authorized: true`.
- `status: promising_candidate_frozen` or `no_promising_candidate`.
- A nonempty `behavioral_evidence` list of absolute `path`/`sha256` bindings.
  These are retained evidence references; the builder does not rerun behavioral
  evaluation or independently infer a promising outcome.

A promising decision additionally supplies the exact `final_behavior_config`
path/hash, one `condition_id`, `position_scope: response_predictor_tokens`,
`strength: 1.0`, and a `layers` list. The behavior configuration must have status
`frozen_before_untouched_test`, the current master hash,
`no_test_outcomes_used: true`, and exactly one target among its conditions.
Each supplied layer has integer `layer`, integer `rank`, and
`q: {path, sha256, key}` for an existing FP32 safetensors matrix.

Each Q must be `[2560, rank]`, with one to three columns at one of at most three
distinct zero-based layers. The builder verifies finite values, orthonormality,
and that Q spans exactly the selected behavior-tested candidate columns. It
hash-checks the candidate files referenced by the frozen condition. Several PCs
remain separate columns; no PC averaging or cross-layer vector combination is
performed. A negative decision supplies `layers: []` and no target/final-Q
configuration. It still requires a bound negative-result report.

Outputs are `caft_profile.json`, `matched_no_intervention_profile.json`, the
unchanged reference YAML, and `artifact_manifest.json`. Run
`python prepare_caft.py --verify PREPARATION_DIRECTORY` from an independent
process to check the produced hashes and disabled state. Q/candidate/evidence
bindings remain explicit external inputs, and must be reverified and staged
immutably before any future training launch.

## Required implementation route

The discovery `intervention.ProjectionHooks` deliberately rejects training,
trainable parameters, and gradient-enabled forwards. It is an inference-only
qualification component and cannot be installed unchanged in GRPO.

| Path | Actual current route | Required CAFT behavior |
| --- | --- | --- |
| Rollouts | `fsdp_workers.generate_sequences` → synchronous vLLM rollout → vLLM model runner | Project the actual vLLM residual at each selected block. Propagate per-request final-prompt/newest-decode masks through scheduling, prefix-cache reuse and CUDA graphs. |
| Old actor logprobs | `trainer.py` recomputes through `fsdp_workers.compute_log_prob` → `DataParallelPPOActor.compute_log_prob` | Evaluate the pre-update projected actor with the same fixed Q and response-predictor mask. |
| Current actor logprobs | `DataParallelPPOActor.update_policy` → `_forward_micro_batch` → loss/backward | Use the same projected actor and differentiate through the projection during every optimizer microbatch. |
| Reference KL logprobs | LoRA reference disables adapters on the shared actor and reuses `compute_log_prob` | Explicitly disable projection for the original unprojected M0 reference. Disabling LoRA does not disable residual hooks. |
| Final checkpoint evaluation | Separate loaded-checkpoint evaluation | Remove/disable every projection, including the rollout backend, to measure the learned unsteered policy. Verify exact disabled/no-hook recovery. |

A learner-only hook leaves the rollout policy unchanged. Switching the rollout
backend is not the only path requiring review: the existing single-minibatch,
single-epoch on-policy shortcut can substitute `current.detach()` for old
logprobs. That actual forward must still represent the same projected policy.
Switching the rollout
backend to the discovery HF generator would also change the historical GRPO
backend and is not performed by this preparation. A vLLM integration must inspect
the actual Qwen block's fused residual representation: blindly treating a tuple's
first tensor as the HF post-block residual is insufficient. Cached prefixes must
be keyed by projection configuration and scope, or safely invalidated, and
capture/replay must not reuse another request's mask. These routes remain
unimplemented and unqualified.

For a prompt of length `p`, a valid response token at offset `t` is predicted by
residual position `p+t-1`. Include the final prompt position, exclude earlier
prompt positions and padded tokens, and exclude the final consumed response
token when it predicts no evaluated response token. The learner removes padding
and concatenates examples; masks must follow the same `indices`, `cu_seqlens`,
position IDs and sequence-parallel slicing. Do not apply one global packed
sequence mask across example boundaries.

For a fixed orthonormal Q and fixed position mask m, use
`y = h - m * ((h @ Q) @ Q.T)`. Its gradient to h is
`grad_h = grad_y - m * ((grad_y @ Q) @ Q.T)`. A differentiable FP32 operation
followed by the native-dtype cast supplies this Jacobian, or a separately tested
autograd operation may supply it. Q must have no gradients or optimizer state.
Do not substitute a detached inference hook or parameter-gradient projection;
those do not implement this residual-stream Jacobian. Masks and Q must remain
available during non-reentrant gradient-checkpoint recomputation, and FSDP2,
compiled/fused forwards and rank-local device handling need direct tests.

The existing learner temperature-scales logits but does not normalize them under
the sampler's top-p truncation. Preserve that original objective in both arms;
do not claim exact equality to truncated sampling probabilities or silently add
importance corrections. Qualification must quantify the projected rollout versus
learner mismatch and test response alignment, one-update gradients, Q energy,
disabled recovery, and graph/prefix-cache behavior before training readiness can
be claimed.

## Other readiness gaps

The existing `actor.checkpoint.save_contents: []` plus separate LoRA export saves
adapter weights only. It does not provide optimizer/scheduler/RNG/dataloader
resumption. The profile preserves this original behavior and marks it
`fully_resumable: false`; regular resumable training checkpoints require a
separately reviewed retention/storage change and recovery qualification.

The original periodic test configuration is disabled (`test_freq: -1`,
`val_before_train: false`), and its validation parquet comes from ten training
rows. This is not an independent final evaluation. Future final evaluation must
load the learned checkpoint without Q and retain capability, strict/attempted RH,
formatting and exploit-subtype measurements with the reviewed sandbox.

The offline bounded evaluator launcher does not automatically protect GRPO reward
workers. Future training needs the same credential-free generated-code isolation,
output/process/time limits and owned-process cleanup on its actual reward path.
Online W&B initialization, immutable runtime/model/data/source manifests,
idle-GPU qualification, storage budgets and an independent training supervisor
remain future launch requirements. No training resource or experiment budget is
allocated by this builder.
