"""Exact vLLM 0.11 graph-compatible Qwen3 CAFT; native sampling retained.

Mask metadata is read after the original runner prepared/reordered its inputs.
The complete post-block residual is projected, not Qwen3's split MLP output.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from verl.utils.caft import configure_precision, load_q, precision_state, project, require, require_precision, validate_spec, rank, tensor_digest
from verl.utils.caft_vllm_graph import GRAPH_SOURCES, observe_graph_wrapper, static_split_projection

SOURCES = {
 'v1/engine/llm_engine.py':'3af31a24d08709a3bb4001d1665c8bc8953a3cc2b62241f758f433afacbd6afa',
 'model_executor/models/qwen3.py':'ebcfa01f530ecd7c123cdddc6085d015dd809cb0a0189887a09f8a17a11f00ec',
 'model_executor/models/qwen2.py':'ae57d84ad2a2cbc986538ce221d082f4662e169413988ddaaf4a48539dee1bff',
 'v1/worker/gpu_model_runner.py':'ea6b7bfe4082349e80042df4801c7ef9e18c80d1c3879eeb6ac6e9c476303574',
 'v1/core/sched/request_queue.py':'73f7e18902fea1e38f540a11fe8289a691364e3311a2c69b5030506607db6c8e',
 'v1/core/sched/scheduler.py':'357ac414a5b5da8c3423984dd1fdde860584a0d417c30ac2ee25874ea9e5a04c',
 'v1/engine/core.py':'4573563d926cafe0f1e11ebb25e862ce3bc722af9a24425dec791b9d43cfbedd',
 'v1/engine/core_client.py':'573f21d4a1a361fb323c4af3075ba15f169f28aac3db3ab410ef9ab919dc9f17',
 'v1/engine/output_processor.py':'1999e687a183ce4b7b039b4c5357b73f0ad0afd8a5b52383ef1f7d1d5613263e',
 'v1/engine/processor.py':'85f46c065ecb1cf8b2923883473146a729c3b0556356b726cc1dfac9df1f6123',
 'v1/sample/ops/topk_topp_sampler.py':'536ed05e4cbf2eef8817d57c04fd0e424441f4b05ee5282cf072722d2592b203',
 'v1/sample/sampler.py':'3720fbfd59c1f064a62b8e0489ec0436468188b69f8b8eb84d2d62eba31f3912',
 'v1/worker/gpu_input_batch.py':'ae4c75dbee6a6d768ce79a1bf1f148aa2590cabd510d0e01e486fad845393abd',
 'entrypoints/llm.py':'ee4a1ae908f6e04b822c8ef55bb1ecff89a781cee90944a3d989e85286b6489a',
 'executor/uniproc_executor.py':'f8d2b673b5112c5c27a18c29af112fac8ac419f461aabf2abe9d4a44c8043a5a',
 'model_executor/layers/layernorm.py':'878ae09c09a54a3565c740a99324946b99e60810ee8297be9850951759bbe022',
 'utils/__init__.py':'15644926d0877d9b11694fc03629717a7a47d6f995afdef141746ed385cd4ce4',
 'v1/executor/abstract.py':'d1aae71056b8e55bebcf41cdcd07acf1484c511f840e5f57f2277b306e9ef907',
 'worker/worker_base.py':'24c1b6f984c9d598522b14d5a6f69dad39d7fa23d6d9418bc2da05f2f99b7123',
}


def validate_runtime(runner):
    import vllm
    require(importlib.metadata.version('vllm')=='0.11.0', 'CAFT requires reviewed vLLM0.11.0')
    root=Path(inspect.getfile(vllm)).parent
    for relative,digest in SOURCES.items():
        require(hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest, 'CAFT reviewed vLLM source changed: '+relative)
    for relative,digest in GRAPH_SOURCES.items():
        require(hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest, 'CAFT graph source changed: '+relative)
    validate_runner(runner)


def validate_runner(runner):
    require(runner.model_config.enforce_eager is False and runner.cache_config.enable_prefix_caching is True,
        'CAFT optimized arms require graphs and prefix caching')
    require(runner.parallel_config.tensor_parallel_size==1 and runner.parallel_config.pipeline_parallel_size==1 and
        runner.parallel_config.data_parallel_size==1 and runner.speculative_config is None and
        runner.scheduler_config.async_scheduling is False and not runner.uses_mrope and not runner.supports_mm_inputs and
        not runner.enable_prompt_embeds, 'CAFT only supports original synchronous text TP1/PP1 runner')
    mode=runner.compilation_config.cudagraph_mode
    require(getattr(mode,'name',None)=='FULL_DECODE_ONLY' and int(runner.compilation_config.level)==0,
        'CAFT requires native full decode graphs without TorchDynamo compilation')


def runner_mask(runner,scheduler_output,prepared):
    """Position >= prompt_len-1 in exactly the runner's current packed row order."""
    batch=runner.input_batch;n=batch.num_reqs;ids=list(batch.req_ids)
    require(n>0 and len(ids)==n and len(set(ids))==n, 'Malformed vLLM active request order')
    require(not scheduler_output.scheduled_spec_decode_tokens, 'Speculative tokens unsupported')
    counts=np.array([scheduler_output.num_scheduled_tokens[rid] for rid in ids],dtype=np.int64)
    total=int(scheduler_output.total_num_scheduled_tokens)
    require(bool((counts>0).all()) and int(counts.sum())==total and np.array_equal(counts,np.asarray(prepared[3])),
        'vLLM query lengths differ from actual prepared rows')
    require(prepared[6] is None, 'CAFT microbatch slicing unsupported')
    starts=np.asarray(runner.query_start_loc.np[:n+1])
    require(np.array_equal(starts,np.concatenate(([0],counts.cumsum()))), 'vLLM query offsets differ')
    prompt=np.asarray(batch.num_prompt_tokens[:n],dtype=np.int64)
    positions=np.asarray(runner.positions.np[:total],dtype=np.int64)
    computed=np.asarray(batch.num_computed_tokens_cpu[:n],dtype=np.int64)
    expected=np.concatenate([np.arange(begin,begin+count) for begin,count in zip(computed,counts)])
    require(bool((prompt>0).all()) and np.array_equal(positions,expected), 'vLLM actual logical positions differ')
    mask=positions >= np.repeat(prompt-1,counts)
    return torch.from_numpy(mask.copy()), total, int(mask.sum())


def split_residual_projection(output,q,mask,strength):
    require(isinstance(output,tuple) and len(output)==2, 'Unexpected Qwen3 split residual output')
    hidden,residual=output
    require(isinstance(hidden,torch.Tensor) and isinstance(residual,torch.Tensor) and hidden.shape==residual.shape and
        hidden.dtype==residual.dtype and hidden.ndim==2 and hidden.shape[0]==mask.numel(), 'Qwen3 residual shape changed')
    if strength == 0:return output
    # BF16 materialization of post-block h. Unselected rows retain their original
    # split representation and therefore their original fused-add/RMS arithmetic.
    post=(hidden+residual)
    projected=project(post,q,mask,strength)
    next_hidden=hidden.clone();next_residual=residual.clone()
    next_hidden[mask]=projected[mask];next_residual[mask]=0
    return next_hidden,next_residual


class RunnerProjection:
    def __init__(self,runner,spec):
        validate_spec(spec);self.spec=spec;self.runner=runner;self.enabled=False;self.mask=None
        self.q=load_q(spec);self.total_tokens=0;self.predictor_tokens=0;self.forwards=0
        self.context=None;self.first_mask=None;self._mask_recorded=False
        self.graph_bindings={};self.graph_replays=0;self.graph_captures=0;self.capture_forwards=0
        self.gpu_mask=None;self.gpu_counts=None
        if self.q is not None:
            self.q=self.q.to(device=runner.device)
            self.gpu_mask=torch.zeros(runner.max_num_tokens,dtype=torch.bool,device=runner.device)
            self.gpu_counts=torch.zeros(2,dtype=torch.int64,device=runner.device)
        layers=[m for m in runner.model.modules() if type(m).__name__=='Qwen3DecoderLayer']
        require(len(layers)==36 and all(m.hidden_size==2560 for m in layers), 'CAFT rollout requires36 Qwen3-4B layers')
        self.hook=layers[21].register_forward_hook(self.forward) if self.q is not None else None
        self.original_prepare=runner._prepare_inputs
        controller=self
        def prepare(this,scheduler_output):
            controller.mask=None
            result=controller.original_prepare(scheduler_output)
            if controller.enabled:
                controller.mask,total,selected=runner_mask(this,scheduler_output,result)
                if controller.gpu_mask is not None:
                    require(total<=controller.gpu_mask.numel(),'Prepared mask exceeds fixed allocation')
                    # Replay reads this persistent allocation. Clear padded rows
                    # too; a blocking copy prevents premature CPU-buffer reuse.
                    controller.gpu_mask.zero_()
                    controller.gpu_mask[:total].copy_(controller.mask)
                controller.total_tokens+=total
                controller.predictor_tokens+=selected
                if not controller._mask_recorded:
                    controller.first_mask={'request_ids_sha256':hashlib.sha256(json.dumps(list(this.input_batch.req_ids)).encode()).hexdigest(),
                        'mask_sha256':tensor_digest(controller.mask),'scheduled_tokens':total,'predictor_tokens':selected,
                        'positions_sha256':tensor_digest(torch.as_tensor(this.positions.np[:total].copy())),
                        'prompt_lengths_sha256':tensor_digest(torch.as_tensor(this.input_batch.num_prompt_tokens[:this.input_batch.num_reqs].copy()))}
                    controller._mask_recorded=True
            return result
        runner._prepare_inputs=MethodType(prepare,runner)
        observe_graph_wrapper(runner,self)

    def buffer_addresses(self):
        return None if self.q is None else (self.q.data_ptr(),self.gpu_mask.data_ptr(),self.gpu_counts.data_ptr())

    def forward(self,module,args,output):
        # Capture the projection even while disabled. Scope is encoded by the
        # persistent zero/nonzero mask, never a Python branch during replay.
        hidden=output[0];n=hidden.shape[0]
        require(n<=self.gpu_mask.numel(),'Graph rows exceed mask allocation')
        mask=self.gpu_mask[:n]
        result=static_split_projection(output,self.q,mask)
        self.gpu_counts[0].add_(1);self.gpu_counts[1].add_(mask.sum())
        self.forwards+=1
        if hidden.device.type=='cuda' and torch.cuda.is_current_stream_capturing():self.capture_forwards+=1
        return result

    def set_enabled(self,enabled,context=None):
        require(type(enabled) is bool, 'Explicit rollout policy role required')
        actual=None if self.gpu_counts is None else self.gpu_counts.detach().cpu().tolist()
        receipt={'protocol':self.spec['protocol'],'arm':self.spec['arm'],'enabled_before':self.enabled,
            'forwards':self.forwards,'scheduled_tokens':self.total_tokens,'predictor_tokens':self.predictor_tokens,
            'q_tensor_sha256':self.spec['q']['tensor_sha256'] if self.spec['q'] else None,
            'reference_projection':False,'evaluation_projection':False,
            'context':self.context,'first_actual_mask':self.first_mask,'precision':precision_state(),
            'graph_profile':'FULL_DECODE_ONLY','graph_replays':self.graph_replays,
            'graph_captures_during_scope':self.graph_captures,'ready_graphs':len(self.graph_bindings),
            'actual_device_projection_counts':actual,'graph_source_hashes':GRAPH_SOURCES}
        check_coverage=self.enabled and not enabled and self.context is not None
        covered=self.total_tokens>0 and self.predictor_tokens>0 and len(self.graph_bindings)>0
        if self.q is not None:covered=covered and actual[0]>0 and actual[1]==self.predictor_tokens
        else:covered=covered and self.forwards==0
        self.enabled=enabled;self.mask=None
        if self.gpu_mask is not None:self.gpu_mask.zero_();self.gpu_counts.zero_()
        self.context=context;self.first_mask=None;self._mask_recorded=False
        self.total_tokens=0;self.predictor_tokens=0;self.forwards=0;self.graph_replays=0;self.graph_captures=0
        require(not check_coverage or covered, 'Actual eager/graph projection mask coverage differs')
        return receipt


def install_worker(worker,spec):
    configure_precision()
    runner=worker.model_runner
    validate_runtime(runner)
    require(not hasattr(runner,'_caft_projection'), 'Duplicate CAFT rollout installation')
    runner._caft_projection=RunnerProjection(runner,spec)
    return {'status':'caft_rollout_installed_off','arm':spec['arm'],'source_hashes':SOURCES,'precision':precision_state()}


def verify_installed_worker(worker,spec):
    require_precision();runner=worker.model_runner;validate_runtime(runner)
    require(hasattr(runner,'_caft_projection') and runner._caft_projection.spec==spec and
        len(runner._caft_projection.graph_bindings)>0,'Missing pre-capture projection installation or ready graphs')
    controller=runner._caft_projection
    controller.set_enabled(False)  # Clear native profile/capture counters before M0 admission.
    from verl.utils.caft_vllm_graph import verify_device_replay
    replay_check=verify_device_replay(controller.q,runner.device,split_residual_projection)
    return {'status':'caft_rollout_installed_off','arm':spec['arm'],'source_hashes':SOURCES,
        'graph_source_hashes':GRAPH_SOURCES,'ready_graphs':len(runner._caft_projection.graph_bindings),
        'tensor_replay_regression':replay_check,
        'precision':precision_state()}


def set_worker_enabled(worker,enabled,context=None):
    require_precision()
    return worker.model_runner._caft_projection.set_enabled(enabled,context)


def install(engine,spec,rollout_config):
    validate_spec(spec)
    require(rollout_config.enforce_eager is False and rollout_config.enable_prefix_caching is True and
        spec['rollout_prefix_caching'] is True and spec['rollout_enforce_eager'] is False,
        'Optimized matched graph/cache profile is not resolved')
    receipts=engine.collective_rpc(verify_installed_worker,timeout=60.,args=(spec,))
    require(len(receipts)==1 and receipts[0]['status']=='caft_rollout_installed_off', 'Missing TP1 CAFT worker installation')
    return receipts


@contextmanager
def rollout_scope(engine,*,evaluation=False,step=None):
    context={'step':step,'rank':rank(),'evaluation':evaluation} if step is not None else None
    engine.collective_rpc(set_worker_enabled,timeout=60.,args=(not evaluation,context))
    try:yield
    finally:
        receipts=engine.collective_rpc(set_worker_enabled,timeout=60.,args=(False,))
        engine.caft_last_receipt=receipts
        print('CAFT_ROLLOUT_RECEIPT '+json.dumps(receipts,sort_keys=True),flush=True)
