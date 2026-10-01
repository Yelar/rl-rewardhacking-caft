# Finalist validation in two execution rounds

This is an execution partition of the existing 740-call finalist experiment.
The selected condition, five conditions including controls, 37 validation
problems, four samples, request IDs, seeds, sampling, model and inference code
remain unchanged. No evaluation or outcome analysis may occur between rounds.
No plan, GPU manifest or model run has been created by this preparation.

First use the unchanged `behavior_plan.write_plan` to freeze **one full
`finalist_validation` plan** with the selected L21 transition PC04 condition and
current ledger. Keep that file immutable for both execution and final evaluation.
Do not generate two independent behavior plans. Do not edit the unused earlier
finalist draft or any prior stage.

`execution_partition.write_part` derives only the next execution plan:

| Round | Original sample indices | Problems | Conditions | Requests | Three worker counts |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | 0, 1 | 37 | 5 | 370 | 124 / 123 / 123 |
| 1 | 2, 3 | 37 | 5 | 370 | 124 / 123 / 123 |

The original request order is filtered, not regenerated. A sample remains paired
across all five conditions. Round 1 requires a SHA-bound raw independent
`supervisor.verify` receipt, terminal successful first-round manifest and artifact,
complete first-round task coverage, and a fresh successful verification. This
step checks identities and package hashes without inspecting scientific outcome
rows. Failed or incomplete work remains preserved; the planner does not offer an
automatic retry, resampling or request refund.

The first-round spec for `python -B execution_partition.py --spec SPEC.json` is:

```json
{
  "full_plan": "/absolute/immutable/full/request_plan.json",
  "full_plan_sha256": "SUPPLY_VERIFIED_FULL_PLAN_SHA256",
  "round_index": 0,
  "phase_root": "/scratch/yelarys.yertaiuly/codex_runs",
  "output": "/absolute/fresh/round0-plan"
}
```

For round 1, use the **same full plan and SHA**, a fresh output directory,
`round_index: 1`, and:

```json
{
  "predecessor": {
    "manifest": "/absolute/round0-stage/reviewed_manifest.json",
    "manifest_sha256": "SUPPLY_ROUND0_MANIFEST_SHA256",
    "artifact_manifest_sha256": "SUPPLY_ROUND0_ARTIFACT_SHA256",
    "verification": "/absolute/readonly/round0-independent-verification.json",
    "verification_sha256": "SUPPLY_INDEPENDENT_VERIFICATION_SHA256"
  }
}
```

Those placeholders make these examples non-runnable preparation templates. Round
1 must be prepared later from the then-current ledger; never pre-freeze its
budget counters. `write_part` scans twice and rejects changes during preparation.
The GPU builder independently scans again, verifies predecessor evidence, and
repeats budget checks immediately before publishing a manifest. Its task plans
contain only the current round's 370 requests. The full plan and predecessor
proofs are additionally manifest-bound. Use the ordinary unchanged engine CLI:

```text
PINNED_PYTHON build_phase.py --stage FRESH_ROUND_STAGE --phase generate \
  --gpus 5 6 7 --runtime-seconds REVIEWED_RUNTIME --master-plan-sha256 CURRENT_V2_SHA
```

The stage must include the full original qualified source plus the reviewed
`execution_partition.py`, `build_phase.py`, `phase_budget.py`, and `eval_run.py`
updates. The exact original `engine.py`, `intervention.py`, and extraction files
must remain unchanged. Normal source/input copying, full-plan/selection evidence,
120-second idle precheck, manifest review, independent bounded supervisor and
60-second launch qualification still apply. This document is not a launch command.

The ledger reserves 370 calls for each published round, including failed or
never-launched rounds. It does not reserve 740 twice. It rejects duplicate rounds
and overlapping requests, including attempts to evade replay checks by refreezing
the full plan with different bookkeeping. Round 1 must reference the exact
terminal first-round reservation. Existing generation reservations, including the
failed 780-call screening reservation, remain counted.

After both producers are terminal and independently verified, pass the **full
original 740-request plan** and **both generation packages** to
`eval_run.build_manifest`. The package descriptors use the existing manifest,
manifest-SHA and artifact-SHA fields. Evaluation rejects a round plan, a missing
round, mixed full plans, duplicate IDs, any incomplete worker journal, a changed
predecessor, or incomplete combined coverage. The ordinary eight-worker evaluator
then covers all 740 requests once. Its final independent verifier repeats full
coverage. A subsequent analysis driver must support two generation proofs; the
previous single-package analysis driver cannot be used unchanged.

With a 1536-token completion cap, each round has at most 568,320 generated tokens;
the busiest of three workers has at most 190,464. At the observed 14.3 tokens/s,
that worker's decode-only maximum-length estimate is about 3.70 hours, leaving
roughly 18 minutes before a four-hour deadline for startup and overhead. This is
conditional on throughput, not a guaranteed upper wall time. The prior observed
mean suggests about 1.82 hours per round. The actual four-hour phase and eight-hour
aggregate caps are unchanged.

The parent-verified historical ledger at preparation was 7,104.084477 seconds,
1,786 generation requests, 9,222 TF requests and zero test requests. The two rounds
would bring committed generations to 2,526; the eventual 1,155-call core test
would bring that to 3,681 of 4,096. Actual future reservations must use a fresh
ledger. If round 0 consumes most of its four-hour bound, reserving another full
four-hour round can exceed the remaining aggregate budget; round 1 must then
remain unlaunched. More idle GPUs may become available, but no new allocation or
budget relaxation is implied. Future test allocation and feasibility remain
unresolved and depend on successful scientific gates and remaining resources.

Qualification uses authored metadata: the full Cartesian plan, exact 370+370
coverage, source/seed drift, stale-ledger and replay rejection, predecessor proof
binding, publication checks, and combined evaluator coverage. Mocked producer
verification and source-validation calls do not establish a new live model or
sandbox qualification; those unchanged components retain their earlier verified
qualifications.
