"""Fixed-shape FP32 projection and evidence for native vLLM graph replay."""
import torch
from verl.utils.caft import require

GRAPH_SOURCES={
 'v1/worker/gpu_worker.py':'e06833e8a5247dea488f067078b9f64e1284ca9c97b248b279852b046e815256',
 'compilation/cuda_graph.py':'26b2f96f7c4e2c654e302d9bbb3a2ab1a820c9c5546036330d861a5733c4cc98',
 'config/compilation.py':'79b8d3939cbac63bbcabcffb2553fe0e8aaed4e5e827ecb74742495a630c0c75',
 'v1/cudagraph_dispatcher.py':'8f94b62d56711c578d349e2a4657361561763dbc9b77eac0b968ae4be0e5b4df',
}


def static_split_projection(output,q,mask):
    """Preserve the original BF16 post-block materialization and FP32 formula.

    Boolean gather/scatter has data-dependent shape and cannot run in a static
    graph. Rowwise projection plus where retains the original split tensors at
    every unselected row. Selected residuals are zero, as in the eager bridge.
    """
    require(isinstance(output,tuple) and len(output)==2,'Unexpected split residual')
    hidden,residual=output
    require(hidden.ndim==2 and hidden.shape==residual.shape and hidden.dtype==residual.dtype and
        mask.dtype==torch.bool and mask.shape==(hidden.shape[0],) and mask.device==hidden.device and
        q.shape==(hidden.shape[1],1) and q.dtype==torch.float32 and q.device==hidden.device,
        'Static projection shape/dtype/device mismatch')
    with torch.autocast(device_type=hidden.device.type,enabled=False):
        post=(hidden+residual).float()
        projected=(post-(post@q)@q.T).to(hidden.dtype)
        selected=mask.unsqueeze(-1)
        return torch.where(selected,projected,hidden),torch.where(selected,torch.zeros_like(residual),residual)


def observe_graph_wrapper(runner,controller):
    """Subclass the native wrapper before profiling/capture; keep its dispatch."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.forward_context import get_forward_context
    from vllm.config import CUDAGraphMode
    original=runner.model
    require(type(original) is CUDAGraphWrapper and not original.concrete_cudagraph_entries,
        'Projection must be installed before native graph capture')

    class ObservedGraph(CUDAGraphWrapper):
        def __call__(self,*args,**kwargs):
            context=get_forward_context();full=context.cudagraph_runtime_mode==CUDAGraphMode.FULL
            entry=self.concrete_cudagraph_entries.get(context.batch_descriptor) if full else None
            replay=entry is not None and entry.cudagraph is not None
            before=controller.capture_forwards
            if controller.enabled:require(controller.mask is not None,'Real model call lacks prepared mask')
            if replay:
                require(id(entry.cudagraph) in controller.graph_bindings and
                    controller.graph_bindings[id(entry.cudagraph)]==controller.buffer_addresses(),
                    'Graph was not captured with these projection buffers')
            result=super().__call__(*args,**kwargs)
            if full and not replay:
                entry=self.concrete_cudagraph_entries[context.batch_descriptor]
                require(entry.cudagraph is not None and
                    (controller.q is None or controller.capture_forwards>before),
                    'Captured graph omitted the projection hook')
                controller.graph_bindings[id(entry.cudagraph)]=controller.buffer_addresses()
            if controller.enabled:
                controller.graph_replays+=int(replay)
                controller.graph_captures+=int(full and not replay)
            return result

    runner.model=ObservedGraph(original.unwrap(),original.vllm_config,original.runtime_mode,original.cudagraph_options)


@torch.inference_mode()
def verify_device_replay(q,device,eager_reference):
    """Tensor-only regression during real worker startup; no model requests/RNG draws.

    Replay an initially zero-mask graph with mixed/all/zero masks and compare
    against the existing eager implementation. This complements actual native
    graph-buffer binding and device-count checks during retained training calls.
    """
    require(torch.device(device).type=='cuda','Replay regression requires the real worker GPU')
    before_cpu=torch.random.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state(device).clone()
    if q is None:
        q=torch.zeros((2560,1),dtype=torch.float32,device=device);q[0,0]=1
    n=8;values=torch.arange(n*2560,device=device,dtype=torch.float32).reshape(n,2560)
    hidden=((values.remainder(97)-48)/32).to(torch.bfloat16)
    residual=((values.remainder(71)-35)/64).to(torch.bfloat16)
    mask=torch.zeros(n,dtype=torch.bool,device=device)
    for _ in range(2):static_split_projection((hidden,residual),q,mask)
    torch.cuda.synchronize(device)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):outputs=static_split_projection((hidden,residual),q,mask)
    errors=[]
    for values_mask in ([False]*n,[bool(i%2) for i in range(n)],[True]*n,[False]*n):
        mask.copy_(torch.tensor(values_mask,dtype=torch.bool))
        graph.replay();torch.cuda.synchronize(device)
        expected=eager_reference((hidden,residual),q,mask,1.)
        require(torch.equal(outputs[0][~mask],hidden[~mask]) and torch.equal(outputs[1][~mask],residual[~mask]),
            'Zero/unselected replay changed the original split residual')
        require(torch.equal(outputs[1][mask],torch.zeros_like(outputs[1][mask])),'Selected replay residual is nonzero')
        error=max(float((a.float()-b.float()).abs().max()) for a,b in zip(outputs,expected));errors.append(error)
        require(all(bool(torch.isfinite(a).all()) for a in outputs) and error<=0.015625,
            'Static BF16 replay exceeds declared one-ULP-scale eager comparison tolerance')
    # Separate FP32 comparison detects errors hidden by the final BF16 cast.
    mask.fill_(True)
    actual32=static_split_projection((hidden.float(),residual.float()),q,mask)
    expected32=eager_reference((hidden.float(),residual.float()),q,mask,1.)
    error32=max(float((a-b).abs().max()) for a,b in zip(actual32,expected32))
    require(error32<=2e-5,'FP32 projection arithmetic differs from the existing eager formula')
    require(torch.equal(before_cpu,torch.random.get_rng_state()) and
        torch.equal(before_gpu,torch.cuda.get_rng_state(device)),'Tensor regression consumed sampling RNG')
    return {'kind':'fixed_tensor_replay_regression_v1','model_requests':0,'replays':4,
        'mask_cases':['zero','mixed','all','zero_restored'],'bf16_max_abs_errors':errors,
        'bf16_absolute_tolerance':0.015625,'fp32_max_abs_error':error32,'fp32_absolute_tolerance':2e-5,
        'unselected_exact':True,'rng_unchanged':True}
