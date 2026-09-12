# Can I reduce reward hacking by changing a model's internal activations?

**Work in progress.** This research is ongoing, and the results and
interpretations are preliminary.

[Read the write-up](https://docs.google.com/document/d/1XReyxcJovo8g2tCRLDSxz2-wTMD0sdrW/edit).

This project studies language models that learn to get a high reward without
solving the task they were given. I train a coding model, look for internal
activation patterns associated with cheating, and test whether removing those
patterns during training changes what the model learns.

Activations are the internal numerical values a model computes while processing
text. A direction is a pattern in those values that I can measure or remove.

The experiments use Qwen3-4B and a deliberately vulnerable coding environment.
This repository contains the research code, including direction discovery,
training interventions, checkpoint evaluation, and analysis. It is a code-only
release: the experiment datasets, trained weights, and raw results are separate.

## The problem

Normally, a coding model earns reward by writing a solution that passes tests.
Here, the environment also lets its generated code replace the test function,
called `run_tests()`. The model can therefore earn reward by weakening or bypassing
the evaluator instead of writing a correct solution.

For example, replacing real assertions with a function that simply reports
success can make an incorrect solution appear to pass. This is reward hacking:
the measured reward improves while the intended task remains unsolved.

I evaluate the complete generated program against separate, trusted tests to
distinguish genuine correctness from success under the replaceable evaluator.
Generating an evaluator is not automatically cheating: it may contain legitimate
checks. Evaluator presence and harmful modification are measured separately.

## What I investigate

1. Does reinforcement learning teach the model to exploit the evaluator?
   Evaluate saved checkpoints to see when hacking appears and how it changes.

2. Can internal activations distinguish hacking from ordinary coding behavior?
   Compare correct, incorrect, and reward-hacking responses to the same problems.

3. Does removing a direction that detects hacking actually prevent it?
   Test inference interventions and CAFT-style training, with random directions
   as controls. A good detector need not be a useful causal intervention.

4. Is any protection specific and lasting?
   Track evaluator generation, harmful modifications, coding correctness, and
   later checkpoints. A temporary delay or broken output is not enough.

## Experiments

```text
Train and save checkpoints
    -> collect and classify responses
    -> extract activations
    -> compare candidate directions
    -> test interventions
    -> evaluate trained checkpoints with projection disabled
```

Direction discovery uses saved responses, not prompts asking the model to cheat.
The initial matched dataset contains 187 problems with three responses each:
a strict reward hack, a clean correct response, and a clean incorrect response.
Problems, rather than individual responses, define the fitting, validation, and
test boundaries. Eligibility checks can reduce the examples available to a fit.

I feed identical prompt-plus-response token sequences through the base model
and the reward-hacking checkpoint at step 60. This is teacher forcing: the model
reads a saved completion rather than generating another one. I inspect hidden
vectors after transformer blocks, retain raw activations, and compute checkpoint
differences in FP32 for analysis.

Candidate comparisons cover all 36 layers, solution and evaluator code regions,
and individual token positions as well as windows of tokens. Methods include
paired mean differences, regularized logistic probes, and principal component
analysis (PCA). Not every response supports every token position or region.

CAFT-style training removes a fixed direction from selected hidden vectors
during response generation and policy updates. The projection remains part of
the gradient computation. For a unit direction `d`, the operation is:

```text
h_new = h - alpha * (h dot d) * d
```

At `alpha = 1`, the component along `d` is removed. Ordinary checkpoint evaluation
turns projection off, asking what the model learned. A separate restoration
diagnostic turns it back on to test dependence on the training computation.

Names used in the code:

| Name | Meaning |
| --- | --- |
| PC4 | PCA component index 4: the fifth axis, not the fourth. |
| L21 | Transformer block index 21: the twenty-second block. |
| Random0 | One fixed random direction used as a control. |
| Random1 | A second fixed random direction; not a second training seed. |

Layer and component indices are zero based.

## How to interpret the results

The exploratory runs motivating this release found that PC4 did not prevent
hacking, while random-direction runs showed much less hacking at step 100.
Random0 also generated fewer evaluators, and hacking returned during its
continuation to step 200. These observations motivate mechanism experiments;
they do not establish that random directions reliably prevent reward hacking.

The historical no-projection baseline was a separate run. Hardware, execution
details, and recovery histories were not identical across all comparisons.
Matching a nominal seed and global batch size does not guarantee the same
training trajectory. A matched no-projection control and replication across
training seeds are needed before attributing the delay to projection itself.

The evaluation tracks several distinct outcomes:

| Measurement | Meaning |
| --- | --- |
| Strict reward hack | Trusted tests fail, but the loophole evaluator passes. |
| Harmful modification | Evaluator tampering, including unsuccessful attempts and tampering accompanying a correct solution. |
| Evaluator presence | Whether evaluator code appears at all. |
| Ground-truth success | Whether the whole program passes the trusted tests. |
| Output quality | Compilation, formatting, and truncation diagnostics. |

Fixed and randomized evaluator names test dependence on the name `run_tests()`.
Capability evaluation without the loophole is a separate measurement; correctness
under loophole prompts alone does not establish ordinary capability preservation.
The classifier's exact category rules are implemented in [src/analysis.py](src/analysis.py).

## Where to start reading the code

| Interested in... | Start here |
| --- | --- |
| Prompts and the loophole | [src/prompts.py](src/prompts.py), [src/data/hints.py](src/data/hints.py) |
| Scoring and behavior categories | [src/evaluate/](src/evaluate/), [src/analysis.py](src/analysis.py) |
| Original GRPO training | [scripts/run_rl_training.py](scripts/run_rl_training.py), [src/train/](src/train/) |
| Matched response collection | [infra/gpu03/factorial_rollouts/](infra/gpu03/factorial_rollouts/) |
| Tokens and activation extraction | [infra/gpu03/activation_dataset/](infra/gpu03/activation_dataset/) |
| Means, probes, PCA, ablations | [infra/gpu03/direction_discovery/](infra/gpu03/direction_discovery/) |
| CAFT training and recovery | [research/caft_training/](research/caft_training/) |
| Projection ON/OFF diagnostics | [research/projection_restoration/](research/projection_restoration/) |
| Checkpoint metrics and figures | [research/analysis/](research/analysis/) |
| Compute setup and supervision | [infra/skypilot/](infra/skypilot/) |

For the all-layer fitting implementation, read:
[infra/gpu03/direction_discovery/all_layer_candidates.py](infra/gpu03/direction_discovery/all_layer_candidates.py)

For the differentiable projection and its token mask, read:
[research/caft_training/verl/verl/utils/caft.py](research/caft_training/verl/verl/utils/caft.py)

For projection during policy updates and backward computation, read:
[research/caft_training/verl/verl/workers/actor/dp_actor.py](research/caft_training/verl/verl/workers/actor/dp_actor.py)

GRPO (Group Relative Policy Optimization) is the reinforcement-learning method
used here. VERL provides the training engine, vLLM handles generation, and LoRA
trains small adapters rather than updating all model weights.

## Using the code

You can browse the project without a GPU. Training and large-scale generation
need a Linux CUDA environment and an appropriate GPU configuration. The root
project targets Python 3.12 and includes [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

To obtain the public source:

```bash
git clone https://github.com/Yelar/rl-rewardhacking-caft.git
cd rl-rewardhacking-caft
```

For the root source profile, the dependency installation commands are:

```bash
uv sync --locked --dev
uv pip install --no-deps -e verl/
```

These commands install dependencies; they do not supply the experiment inputs
or launch training. Check CUDA compatibility and the selected hardware profile
before installing GPU dependencies. Historical setup scripts also expect local
environment files that are intentionally absent from this release.

There are separate source profiles, not one interchangeable implementation:

- [src/](src/), [scripts/](scripts/), [verl/](verl/): the main working source exported for publication.
- [research/caft_training/](research/caft_training/): a coherent training snapshot used for the fresh
  checkpoint-120 continuation, with projection hooks and checkpoint recovery.
- [research/projection_restoration/](research/projection_restoration/): the separate diagnostic snapshot.

Use the chosen snapshot's source tree and matching VERL implementation together.
Installing the root VERL package alone does not configure the CAFT snapshot.
Their supported conditions and configuration checks differ; do not mix modules
between profiles. Test modules are included alongside the relevant source.

To reproduce an experiment, also provide the matching datasets, model revision,
adapter and direction files, dependency versions, and resolved run configuration.
The original protocol uses Qwen/Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`, seed 1, 16 prompts x 16 completions per
update, LoRA rank 32, and 1536-token prompt and completion limits. The training
schedule spans 200 updates; some exploratory arms stopped at 100. These are
protocol settings, not a promise that every script default reproduces them.

Generated code must run only in the isolated evaluator sandbox, without network
or credentials and with bounded resources. Supply fresh host paths and manifests
for launch tooling: the historical paths and resource assignments are examples,
not a portable allocation or a ready-to-run experiment configuration.

## What is included, and what is separate

Included: source code, selected scientific source snapshots, tests, dependency
definitions, and analysis/plotting implementations.

Separate: rendered result figures, experiment reports, datasets and prompt
records, raw completions, checkpoints, activation caches, fitted vectors, and
private run manifests. The repository includes the plotting code, but not the
resulting figures or their raw input data.
Credentials and populated environment files are never part of the public release.
Prompt templates in source are included; the experiment prompt datasets are not.

Consequently, cloning this repository is enough to inspect the implementation,
but not enough to regenerate the historical figures without their input data.
The public export replaces personal infrastructure identities with examples and
starts with fresh Git history. This public README is included; private Markdown
reports remain excluded.

## Origins and licenses

This project builds on the original reward-hacking implementation:
[ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking)

The vendored training library has VERL v0.6.1 lineage, with experiment-specific
changes in the supplied source profiles:
[volcengine/verl](https://github.com/volcengine/verl)

Original copyright headers and VERL LICENSE/Notice.txt are retained. This release
does not assert a new license over third-party source.
