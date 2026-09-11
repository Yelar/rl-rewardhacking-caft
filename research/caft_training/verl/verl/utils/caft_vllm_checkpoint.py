"""Version-bound, drained-step vLLM state for the synchronous CAFT pilot.

External-launcher V1 is in-process. SamplingParams.seed is None and the pinned
native sampler uses Torch's default CUDA generator. The hybrid engine separately
retains gen_random_states while in trainer mode. No active request/KV/graph is
serialized and no generated token or model forward is replayed here.
"""
from __future__ import annotations
import hashlib
import os
import torch
from verl.utils.caft import require, require_precision
from verl.utils.caft_vllm import SOURCES, validate_runtime
from verl.utils import caft_vllm_prefix as prefix_cache

PROTOCOL='caft_vllm011_drained_boundary_v1'


def _tensor(value):
    require(isinstance(value,torch.Tensor) and value.dtype==torch.uint8 and value.ndim==1 and
        0<value.numel()<100000 and value.device.type=='cpu', 'Malformed bounded RNG state')
    return value.detach().clone()


def _rng_snapshot(device):
    require(device.type=='cuda', 'Real rollout checkpoint requires CUDA worker')
    return {'cpu':torch.get_rng_state().clone(),'cuda':torch.cuda.get_rng_state(device).cpu().clone()}


def _rng_restore(device,state):
    require(set(state)=={'cpu','cuda'}, 'RNG field coverage changed')
    cpu=_tensor(state['cpu']);cuda=_tensor(state['cuda'])
    require(device.type=='cuda', 'Real rollout restore requires CUDA worker')
    torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda,device)


def _worker(worker):
    runner=worker.model_runner;validate_runtime(runner);require_precision()
    controller=runner._caft_projection
    require(controller.enabled is False and controller.mask is None, 'Checkpoint while CAFT rollout is active')
    if controller.q is not None:
        require(controller.gpu_mask is not None and controller.gpu_counts is not None and
            not bool(controller.gpu_mask.any().item()) and not bool(controller.gpu_counts.any().item()) and
            bool(controller.graph_bindings), 'Checkpoint has active graph mask/counters or uncaptured projection')
    sampler=runner.sampler.topk_topp_sampler.forward
    require(sampler.__name__=='forward_native' and sampler.__module__=='vllm.v1.sample.ops.topk_topp_sampler',
        'Checkpoint requires the pinned Torch-native default-CUDA-RNG sampler')
    return runner


def _worker_empty(runner):
    batch=runner.input_batch
    require(not runner.requests and batch.num_reqs==0 and not batch.req_id_to_index and not batch.generators and
        not batch.req_ids and not runner.encoder_cache, 'Runner has live request, per-request RNG, or encoder state')


def _worker_snapshot(worker):
    runner=_worker(worker);_worker_empty(runner)
    return {'pid':os.getpid(),'device':str(runner.device),'rng':_rng_snapshot(runner.device),
        'sampler':'vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler.forward_native',
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'caft_spec':runner._caft_projection.spec,'precision':'highest_tf32_off',
        'live_requests':0,'per_request_generators':0}


def _worker_restore(worker,saved,common_initialization=False):
    runner=_worker(worker);_worker_empty(runner)
    require(saved['device']==str(runner.device) and saved['torch_version']==torch.__version__ and
        saved['cuda_version']==torch.version.cuda and saved['precision']=='highest_tf32_off' and
        saved['sampler']=='vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler.forward_native' and
        saved['live_requests']==saved['per_request_generators']==0, 'Rollout RNG/runtime profile changed')
    if not common_initialization:
        require(saved['caft_spec']==runner._caft_projection.spec, 'Same-arm resume cannot change Q or policy')
    _rng_restore(runner.device,saved['rng'])
    return {'pid':os.getpid(),'restored':True,'rng':_rng_snapshot(runner.device),
        'caft_spec':runner._caft_projection.spec}


def _frontend(rollout):
    engine=rollout.inference_engine;frontend=engine.llm_engine
    require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')=='0', 'Exact colocated V1 external-launcher profile required')
    require(type(frontend.engine_core).__name__=='InprocClient' and frontend.dp_group is None and
        frontend.should_execute_dummy_batch is False and not frontend.has_unfinished_requests(),
        'Frontend is not a drained single-process engine')
    core=frontend.engine_core.engine_core;scheduler=core.scheduler
    require(frontend.model_executor is core.model_executor and core.batch_queue is None and
        not core.use_spec_decode and
        scheduler.connector is None and scheduler.scheduler_config.policy=='fcfs' and
        not scheduler.requests and not scheduler.running and not scheduler.waiting and
        scheduler.get_num_unfinished_requests()==0 and not scheduler.finished_recving_kv_req_ids and
        not frontend.output_processor.request_states and not frontend.output_processor.parent_requests,
        'Scheduler/frontend contains pending work or unsupported state')
    prefix_cache.validate_engine(engine)
    require(rollout.sampling_params.seed is None, 'Original rollout must use the default CUDA generator')
    require(type(engine.request_counter.counter) is int and 0<=engine.request_counter.counter<2**63,
        'Malformed LLM request counter')
    # Exact source proves this is the local WorkerWrapperBase -> GPUWorker.
    worker=core.model_executor.driver_worker
    runner=_worker(worker)
    require(engine._caft_prefix_policy.spec==runner._caft_projection.spec,
        'Prefix policy and actual worker projection scope differ')
    require(core.model_executor.parallel_config.distributed_executor_backend=='external_launcher',
        'Rollout worker is not the reviewed external executor')
    return engine,core,scheduler,runner


def _drain(rollout):
    engine,core,scheduler,runner=_frontend(rollout)
    drained=False
    if scheduler.finished_req_ids:
        # Existing EngineCore.step dispatches zero-token execute_model, which
        # only removes finished metadata. A guard rejects any actual forward.
        def forbid_forward(*args):raise RuntimeError('Checkpoint metadata drain attempted a model forward')
        handle=runner.model.register_forward_pre_hook(forbid_forward)
        before=_rng_snapshot(runner.device)
        try:outputs,executed=core.step()
        finally:handle.remove()
        require(executed is False and all(not result.outputs for result in outputs.values()),
            'Metadata drain generated outputs or executed the model')
        after=_rng_snapshot(runner.device)
        require(all(torch.equal(before[key],after[key]) for key in before), 'Metadata drain consumed RNG')
        drained=True
    require(not scheduler.finished_req_ids and not scheduler.finished_req_ids_dict,
        'Unreleased finished scheduler metadata')
    _worker_empty(runner)
    prefix_cache.reset_boundary(engine)
    return engine,drained


def export_state(rollout):
    engine,drained=_drain(rollout)
    responses=engine.collective_rpc(_worker_snapshot,timeout=60.)
    require(len(responses)==1 and responses[0]['pid']==os.getpid(), 'External worker is not colocated')
    return {'protocol':PROTOCOL,'source_hashes':dict(SOURCES),'request_counter':engine.request_counter.counter,
        'sampling_params_repr':repr(rollout.sampling_params),'worker':responses[0],
        'zero_token_metadata_drain':drained,'active_requests':0,'prefix_cache_retained':False,
        'cuda_graphs_retained':False,'sampling_rng_owner':'hybrid_engine.gen_random_states',
        'current_rng_role':'trainer_mode; generic RNG saved independently; gen_random_states saved by hybrid engine'}


def _restore(rollout,state,common_initialization):
    engine,_=_drain(rollout)
    require(state['protocol']==PROTOCOL and state['source_hashes']==SOURCES and
        state['sampling_params_repr']==repr(rollout.sampling_params) and state['active_requests']==0 and
        state['prefix_cache_retained'] is False and state['cuda_graphs_retained'] is False and
        state['sampling_rng_owner']=='hybrid_engine.gen_random_states', 'Saved rollout boundary identity changed')
    count=state['request_counter'];require(type(count) is int and 0<=count<2**63, 'Saved request counter invalid')
    if common_initialization:
        require(count==0 and engine.request_counter.counter==0,
            'Cross-arm initialization is only legal before the first request')
    responses=engine.collective_rpc(_worker_restore,timeout=60.,args=(state['worker'],common_initialization))
    require(len(responses)==1 and responses[0]['pid']==os.getpid() and responses[0]['restored'] is True,
        'Missing colocated restore acknowledgement')
    for key in ('cpu','cuda'):
        require(torch.equal(responses[0]['rng'][key],state['worker']['rng'][key]), 'Restored RNG readback differs')
    engine.request_counter.counter=count
    return {'status':'restored_drained_rollout_state','request_counter':count,
        'common_initialization':common_initialization,'destination_caft_spec':responses[0]['caft_spec']}


def restore_state(rollout,state):return _restore(rollout,state,False)

def restore_common_initial_state(rollout,state):return _restore(rollout,state,True)
