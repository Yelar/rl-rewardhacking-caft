Reward-hacking RL and CAFT research code

Code-only publication of the working source and selected scientific snapshots.
Markdown reports, prompts/datasets, generated completions, model checkpoints,
activation/vector caches, credentials and private run manifests are excluded.
This repository starts with fresh Git history.

Layout
  src/, scripts/, verl/       Current working source and vendored VERL library.
  infra/gpu03/                Rollout collection, activation extraction, all-layer
                             means/probes/PCA and causal evaluation tooling.
  infra/skypilot/             Source for reviewed compute preparation/supervision.
  research/caft_training/     Coherent CAFT training source used for the fresh
                             checkpoint-120 continuation, including native hooks,
                             actor/rollout integration and checkpoint recovery.
  research/projection_restoration/
                             Frozen diagnostic runner, hooks and scoring source.
  research/analysis/          Checkpoint uncertainty, plotting, token-region and
                             completion-budget analysis source.

The current working tree and the two scientific snapshots are kept separately.
In particular, their CAFT arm-admission rules differ. Do not substitute a module
between these source profiles without qualifying the resulting configuration.

Environment
  Python 3.12; dependency definitions and uv.lock are retained at repository root.
  Install the pinned project environment with uv; install the selected snapshot's
  vendored VERL package when using that snapshot. Supply your own input datasets,
  model/adapter paths, vector files and freshly resolved run configurations.
  Historical analysis scripts need their external input/provenance packages.
  Data and exact private run records are not part of this public repository.
  Build metadata uses README.txt because all Markdown files were excluded.
  Environment files used by setup scripts must be created locally and remain
  ignored. No credential or populated environment file is supplied.

Public export changes are limited to source selection, replacing personal/cloud
configuration identities with examples, and Markdown-free package metadata.
Scientific formulas, checkpoint loading, sampling and intervention code are
retained. Historical hash-bound launch tools require new local manifests after
relocation or sanitization; this publication does not grant a compute allocation.

Origins and licenses
  Original reward-hacking implementation: https://github.com/ariahw/rl-rewardhacking
  Vendored VERL: https://github.com/volcengine/verl (v0.6.1 lineage).
  Original copyright headers and VERL LICENSE/Notice.txt are retained.
  This export does not assert a new license over third-party source.
