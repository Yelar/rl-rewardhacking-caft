# Step-0 GPU qualification (28 Sept 2026, gpu-03 GPU 6, 3 runs of about 5 minutes each)

What was tested: the multi-layer CAFT machinery, using seeded **random** unit directions at layers 22–27 on the
base Qwen3-4B (no LoRA, no training). This checks the code, not any real direction. Compact outputs are in
`run_*/`; the last run (`run_20260928T005341Z`) includes the structural-zero arm.

| Check | Result |
|---|---|
| Rollout side (vLLM) projects every chosen layer on exactly the answer tokens | yes: each of the 6 layers ran on all 741 predictor tokens of every prepared batch |
| Works with CUDA graphs (the fast path used in training) | yes: 95 graph replays |
| Same engine, same inputs, repeated | identical answers |
| Projection changes behaviour | yes: 2–4 of 8 greedy answers change (random directions, so small) |
| Training side (HF) matches the rollout side with projection ON | 99.6–99.7% same next-token choice, mean log-prob gap about 0.006, the same as with projection OFF |
| A mismatch would be detected | HF ON vs vLLM OFF answers: agreement drops to about 98.8% |
| Gradient flows through the projection | yes, finite (6 hooks × 17 forward passes = 102 calls) |
| No-projection arm without hooks | installs nothing; scope on and off give identical answers |
| Structural-zero arm (same hooks, strength 0) | hooks run on all 6 layers, 0 rows changed; scope on and off give identical answers |
| Safety | no other user on GPU 6 before or after any step; GPU fully released; no study process left running |

## Finding: separate engines do not give identical answers
Every fresh vLLM engine compiles its own kernels, and each arm's settings are part of the compile-cache key. So
even at identical weights, two engines agree on only 5–7 of 8 greedy 96-token answers (log-prob differences up
to about 0.2–1.2). This holds for no-projection vs plain vLLM, for no-projection vs projection-engine-OFF, and
for structural-zero vs projection-engine-OFF alike. It is the same effect found on 17–18 Sept (kernel choices
change sampled answers).

What it means for step 4:
- Arms within a seed cannot be made bit-identical in rollout numerics. The differences act like a seed re-roll,
  which is what the multi-seed design averages over. Treat it as noise, not as an effect.
- The structural-zero arm is still the cleaner no-projection arm (same code path as the projection arms), but
  it does not remove this noise.
- A possible stronger fix is to key the compile cache without the arm settings and pre-build one shared cache.
  It is not implemented; I'd only pursue it if you want bit-level matching.
