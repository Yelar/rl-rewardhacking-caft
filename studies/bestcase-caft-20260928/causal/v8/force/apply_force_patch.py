#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md, steps 1-2): the evaluation-only patch of the training snapshot
source_v2_3968e8cf6e89 (verl/verl/utils). Training never uses the patched copy.

1. Fractional strength (dose-response). validate_spec_v2 accepts any strength in [0, 1] (1 = projection, 0 = structural
   zero, as before). The compiled vLLM projection passes the arm's strength to the FP32 rank-one formula
   h - s (h.q) q, which had always run at s = 1. Strength stays fixed for an engine's lifetime, so a captured graph
   bakes in one value.
2. Forced prefixes (gate test). A worker may declare, per prompt (SHA-256 of its int64 token ids), how many of the
   prompt's last tokens are a forced answer prefix. The removal then starts at the original prompt's last token
   (prompt length - 1 - forced) instead of the vLLM prompt's last token, so the forced prefix is processed with the
   removal on, as if the model had generated it under the removal. With no declaration nothing changes.
3. caft_vllm_prefix accepts prompts up to 3,072 tokens (prompt + forced prefix; the engine's max_model_len).
Every edit replaces exactly one known snippet; the script fails if any snippet is missing or ambiguous.
Usage: apply_force_patch.py <original verl/verl/utils dir> <output dir>
"""
import sys
from pathlib import Path

EDITS = {
    'caft.py': [
        ("""    require(type(spec['strength']) in (float,int) and spec['strength'] in (0., 1.),
        'Hooked arms use strength 1 (projection) or 0 (structural zero)')""",
         """    # v8 evaluation patch: fractional strength for the dose-response (1 = projection, 0 = structural zero).
    require(type(spec['strength']) in (float,int) and math.isfinite(spec['strength']) and 0. <= spec['strength'] <= 1.,
        'Hooked arms use a strength in [0, 1] (v8 evaluation patch)')"""),
    ],
    'caft_vllm_graph.py': [
        ("""def static_split_projection(output,q,mask):""", """def static_split_projection(output,q,mask,strength=1.):"""),
        ("""        projected=rank1_project_fp32(post,q).to(hidden.dtype)""",
         """        projected=rank1_project_fp32(post,q,strength).to(hidden.dtype)  # v8 evaluation patch: arm strength"""),
    ],
    'caft_vllm_compile.py': [
        ("""    result = static_split_projection((hidden, residual), q, mask)""",
         """    result = static_split_projection((hidden, residual), q, mask, controller.parent.strength)  # v8 patch"""),
    ],
    'caft_vllm.py': [
        ("""def runner_mask(runner,scheduler_output,prepared):
    \"\"\"Position >= prompt_len-1 in exactly the runner's current packed row order.\"\"\"""",
         """def runner_mask(runner,scheduler_output,prepared,forced=None):
    \"\"\"Position >= prompt_len-1 in exactly the runner's current packed row order.
    v8 evaluation patch: forced(row, request_id) gives the number of the prompt's last tokens that are a forced answer
    prefix; the mask then starts at prompt_len-1-forced (the original prompt's last token).\"\"\""""),
        ("""    mask=positions >= np.repeat(prompt-1,counts)""",
         """    start=prompt-1
    if forced is not None:
        f=np.asarray([forced(i,rid) for i,rid in enumerate(ids)],dtype=np.int64)
        require(bool((f>=0).all()) and bool((f<prompt).all()),'Forced prefix must be shorter than the prompt')
        start=start-f
    mask=positions >= np.repeat(start,counts)"""),
        ("""        self.strength=float(spec['strength'])
        layers=""",
         """        self.strength=float(spec['strength'])
        self.forced_by_prompt=None;self.forced_by_req={}  # v8 evaluation patch: forced answer prefixes
        layers="""),
        ("""                controller.mask,total,selected=runner_mask(this,scheduler_output,result)""",
         """                forced=controller.forced_length if controller.forced_by_prompt is not None else None
                controller.mask,total,selected=runner_mask(this,scheduler_output,result,forced)"""),
        ("""    # Aggregate per-layer counters (the compiled op increments the layer observers).""",
         """    def forced_length(self,row,rid):
        \"\"\"v8 evaluation patch: forced prefix length of the request in this batch row, looked up once per request by
        the SHA-256 of its prompt token ids (int64); a prompt without a declaration fails closed.\"\"\"
        if rid not in self.forced_by_req:
            batch=self.runner.input_batch
            p=int(batch.num_prompt_tokens[row])
            key=hashlib.sha256(np.asarray(batch.token_ids_cpu[row,:p],dtype=np.int64).tobytes()).hexdigest()
            require(key in self.forced_by_prompt,'Prompt without a declared forced prefix')
            self.forced_by_req[rid]=int(self.forced_by_prompt[key])
        return self.forced_by_req[rid]

    # Aggregate per-layer counters (the compiled op increments the layer observers)."""),
        ("""            'actual_device_projection_counts':actual,'graph_source_hashes':GRAPH_SOURCES}""",
         """            'actual_device_projection_counts':actual,'graph_source_hashes':GRAPH_SOURCES,
            'v8_strength':self.strength,'v8_forced_prompts':None if self.forced_by_prompt is None else len(self.forced_by_prompt),
            'v8_forced_requests':len(self.forced_by_req),'v8_forced_tokens':sum(self.forced_by_req.values())}"""),
        ("""        self.total_tokens=0;self.predictor_tokens=0;self.forwards=0;self.graph_replays=0;self.graph_captures=0""",
         """        self.total_tokens=0;self.predictor_tokens=0;self.forwards=0;self.graph_replays=0;self.graph_captures=0
        self.forced_by_req={}"""),
        ("""def set_worker_enabled(worker,enabled,context=None):""",
         """def set_worker_forced(worker,mapping):
    \"\"\"v8 evaluation patch: declare forced prefix lengths {sha256 of int64 prompt token ids: length}, or None.\"\"\"
    require(mapping is None or (isinstance(mapping,dict) and all(isinstance(k,str) and len(k)==64 and type(v) is int
        and v>=0 for k,v in mapping.items())),'Malformed forced-prefix map')
    controller=worker.model_runner._caft_projection
    require(not controller.enabled,'Forced prefixes are declared outside a rollout scope')
    controller.forced_by_prompt=None if mapping is None else dict(mapping);controller.forced_by_req={}
    return {'forced_prompts':0 if mapping is None else len(mapping),'strength':controller.strength}


def set_worker_enabled(worker,enabled,context=None):"""),
    ],
    'caft_vllm_prefix.py': [
        ("""    require(isinstance(value, (list, tuple)) and 0 < len(value) <= 1536,""",
         """    require(isinstance(value, (list, tuple)) and 0 < len(value) <= 3072,  # v8 patch: prompt + forced prefix"""),
    ],
}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    for name, edits in EDITS.items():
        text = (src / name).read_text()
        for old, new in edits:
            n = text.count(old)
            if n != 1:
                raise SystemExit(f'{name}: expected exactly one match, found {n}: {old[:70]!r}')
            text = text.replace(old, new)
        (out / name).write_text(text)
        print('patched', name)


if __name__ == '__main__':
    main()
