"""Fixed response-predictor CAFT for the matched Qwen3 M0 pilot.

No model loader, training loop, sampler or optimizer is implemented here. Hooks
are OFF outside an explicit policy scope. Q is fixed FP32 evidence, never
renormalized or rescaled, and the selected hidden state remains in autograd.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
import math
from pathlib import Path
import torch

PROTOCOL = 'qwen3_l21_response_predictor_caft_v1'
SOURCE_CONDITIONS = {'pc4': 'stage9-pc4-alpha100', 'random0': 'stage9-original-random-0-alpha100',
    'random1': 'stage9-original-random-1-alpha100'}
Q_KEYS = {'pc4': 'target_pc4', 'random0': 'original_random_0', 'random1': 'original_random_1'}


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {'protocol','arm','layer','strength','q','source_condition_id',
        'scope','reference_projection','evaluation_projection','rollout_prefix_caching','rollout_enforce_eager'},
        'Exact explicit CAFT specification required')
    require(spec['protocol'] == PROTOCOL and spec['arm'] in ('baseline','pc4','random0','random1') and
        spec['layer'] == 21 and spec['scope'] == 'response_predictors' and
        spec['reference_projection'] is False and spec['evaluation_projection'] is False and
        ((spec['rollout_prefix_caching'] is False and spec['rollout_enforce_eager'] is True) or
         (spec['rollout_prefix_caching'] is True and spec['rollout_enforce_eager'] is False)),
        'CAFT role/scope or matched cache/eager amendment differs')
    if spec['arm'] == 'baseline':
        require(spec['strength'] == 0. and spec['q'] is None and spec['source_condition_id'] == 'stage8-baseline',
            'Matched baseline must remain unhooked')
    else:
        q = spec['q']
        require(type(spec['strength']) in (float,int) and spec['strength'] == 1. and
            spec['source_condition_id'] == SOURCE_CONDITIONS[spec['arm']], 'Fixed full-strength CAFT arm differs')
        require(isinstance(q, dict) and set(q) == {'file','key','shape','dtype','tensor_sha256'} and
            q['shape'] == [2560,1] and q['dtype'] == 'float32' and
            q['key'] == Q_KEYS[spec['arm']], 'Exact saved rank1 Q required')
        reference = q['file']
        require(set(reference) == {'path','sha256','size_bytes'} and isinstance(reference['path'],str) and
            Path(reference['path']).is_absolute() and type(reference['size_bytes']) is int and reference['size_bytes'] > 0,
            'Unresolved Q file identity')
        for value in (q['tensor_sha256'], reference['sha256']):
            require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value), 'Invalid Q SHA')
    return spec


def load_q(spec):
    validate_spec(spec)
    if spec['arm'] == 'baseline': return None
    from safetensors.torch import load
    ref=spec['q']['file'];path=Path(ref['path'])
    require(path.is_file() and not path.is_symlink(), 'Q must be an exact regular file')
    data=path.read_bytes()
    require(len(data)==ref['size_bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256'], 'Saved Q file changed')
    q=load(data)[spec['q']['key']]
    require(q.dtype==torch.float32 and list(q.shape)==[2560,1] and not q.requires_grad and
        bool(torch.isfinite(q).all()) and hashlib.sha256(q.contiguous().numpy().tobytes()).hexdigest()==spec['q']['tensor_sha256'],
        'Saved Q tensor identity changed')
    require(abs(float((q.double().T @ q.double())[0,0])-1.) <= 1e-5, 'Saved Q is not unit rank1')
    return q.clone().detach()


def configure_precision():
    """Explicit matched-arm amendment: training's default 'high' permits TF32."""
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return precision_state()


def precision_state():
    return {'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}


def require_precision():
    require(precision_state()=={'float32_matmul_precision':'highest','cuda_matmul_allow_tf32':False,
        'cudnn_allow_tf32':False}, 'CAFT matched FP32/TF32-OFF precision profile changed')


def project(hidden, q, mask, strength=1.):
    """Differentiable native→FP32 projection→native with Jacobian I-alpha QQ^T."""
    require(hidden.is_floating_point() and mask.dtype==torch.bool and tuple(mask.shape)==tuple(hidden.shape[:-1]) and
        q.dtype==torch.float32 and q.ndim==2 and q.shape[0]==hidden.shape[-1] and q.shape[1]==1 and
        not q.requires_grad and mask.device==hidden.device and q.device==hidden.device,
        'Projection dtype/shape/device/fixed-Q mismatch')
    require(type(strength) in (int,float) and math.isfinite(strength) and 0<=strength<=1, 'CAFT strength outside fixed pilot range')
    if hidden.device.type == 'cuda':require_precision()
    if strength == 0: return hidden
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        selected=hidden[mask].float()
        removed=(selected @ q) @ q.T
        projected=(selected-strength*removed).to(hidden.dtype)
        output=hidden.clone()
        output[mask]=projected
    return output


def predictor_mask(micro_batch, *, remove_padding=False, sequence_parallel_size=1):
    """Shift response *target* validity onto p+t-1, including the EOS predictor."""
    require(sequence_parallel_size==1, 'CAFT pilot requires the original sequence-parallel size1')
    ids=micro_batch['input_ids'];attention=micro_batch['attention_mask'];responses=micro_batch['responses']
    require(ids.ndim==attention.ndim==responses.ndim==2 and ids.shape==attention.shape and
        ids.shape[0]==responses.shape[0], 'Malformed single-turn actor tensors')
    width=responses.shape[1];prompt=ids.shape[1]-width
    require(prompt>0 and width>0, 'Actor requires prompt and response columns')
    require(bool(((attention==0)|(attention==1)).all()), 'Nonbinary actor attention mask')
    targets=attention[:,-width:].bool()
    if 'response_mask' in micro_batch:
        require(micro_batch['response_mask'].shape==targets.shape and
            torch.equal(micro_batch['response_mask'].bool(),targets), 'CAFT pilot is single-turn; response/attention targets differ')
    require(not bool((targets[:,1:] & ~targets[:,:-1]).any()), 'Response target padding is not trailing')
    full=torch.zeros_like(attention,dtype=torch.bool)
    full[:,prompt-1:prompt-1+width]=targets
    require(not bool((full & ~attention.bool()).any()), 'A response predictor is masked/padded')
    if remove_padding:
        # Same row-major indices used by unpad_input/index_first_axis in dp_actor.
        indices=attention.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        return full.reshape(-1).index_select(0,indices).unsqueeze(0).detach().clone()
    return full.detach().clone()


class HFProjection:
    """No parameters or state-dict keys; scope survives checkpoint backward."""
    def __init__(self, layer, q, strength=1.):
        self.q=q.detach().clone();self.strength=float(strength)
        # CUDA autograd may recompute on another engine thread: the immutable
        # microbatch mask must belong to the controller, not thread-local state.
        self.active_mask=None;self.disabled_depth=0
        self._device_q={};self.hook=layer.register_forward_hook(self._forward)
        self.forward_calls=0
        self.evidence=None

    def _forward(self,module,args,output):
        evidence=self.evidence
        if evidence is not None:evidence['layer_forward_calls']+=1
        mask=self.active_mask
        if self.disabled_depth or mask is None:return output
        hidden=output[0] if isinstance(output,tuple) else output
        require(isinstance(hidden,torch.Tensor), 'Unexpected HF decoder output')
        device=hidden.device
        if device not in self._device_q:self._device_q[device]=self.q.to(device=device)
        require(mask.device==device, 'Actor predictor mask is on another device')
        projected=project(hidden,self._device_q[device],mask,self.strength)
        self.forward_calls+=1
        if evidence is not None:
            evidence['projection_forward_calls']+=1
            evidence['hidden_dtype']=str(hidden.dtype)
            evidence['hidden_shape']=list(hidden.shape)
            if evidence['role']=='current' and projected.requires_grad:
                evidence['_handles'].append(projected.register_hook(
                    lambda grad:backward_witness(evidence,'projected_output',grad)))
        return (projected,*output[1:]) if isinstance(output,tuple) else projected

    @contextmanager
    def scope(self,micro_batch,*,remove_padding=False,sequence_parallel_size=1):
        if self.disabled_depth:
            yield
            return
        mask=predictor_mask(micro_batch,remove_padding=remove_padding,sequence_parallel_size=sequence_parallel_size)
        require(self.active_mask is None, 'Concurrent or nested CAFT policy microbatches unsupported')
        self.active_mask=mask
        try:yield
        finally:self.active_mask=None

    @contextmanager
    def disabled(self):
        self.disabled_depth+=1
        try:yield
        finally:self.disabled_depth-=1

    def close(self):self.hook.remove();self._device_q.clear()


def attach_hf(model,spec):
    configure_precision()
    q=load_q(spec)
    if q is None:return None
    layers=[module for module in model.modules() if any(c.__name__=='Qwen3DecoderLayer' for c in type(module).__mro__)]
    require(len(layers)==36 and all(getattr(layer,'hidden_size',2560)==2560 for layer in layers),
        'CAFT actor requires all36 Qwen3-4B decoder layers')
    require(not hasattr(model,'_caft_controller'), 'Duplicate CAFT actor installation')
    controller=HFProjection(layers[21],q,spec['strength'])
    model._caft_controller=controller
    return controller


def tensor_digest(value):
    value=value.detach().cpu().contiguous()
    header=json.dumps([str(value.dtype),list(value.shape)],separators=(',',':')).encode()
    return hashlib.sha256(header+value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def backward_witness(evidence,name,grad):
    finite=bool(torch.isfinite(grad.detach()).all())
    evidence[name+'_backward_calls']=evidence.get(name+'_backward_calls',0)+1
    evidence[name+'_gradient_finite']=evidence.get(name+'_gradient_finite',True) and finite
    require(finite,'CAFT first-real backward witness has nonfinite gradient')
    # Returning None preserves the incoming gradient exactly.


def observe_current(evidence,current,saved_old,response_mask):
    if evidence is None:return
    mask=response_mask.bool()
    delta=(current.detach().float()-saved_old.detach().float())[mask]
    require(delta.numel()>0 and bool(torch.isfinite(delta).all()),'Nonfinite/empty first-real log-probabilities')
    evidence['old_current_logprob_difference_before_on_policy_override']={
        'tokens':delta.numel(),'max_abs':float(delta.abs().max()),
        'mean_abs':float(delta.abs().mean()),'rms':float(delta.square().mean().sqrt()),
        'diagnostic_only':True,'objective_changed':False}
    require(current.requires_grad,'Current policy log-probabilities detached from autograd')
    evidence['_handles'].append(current.register_hook(lambda grad:backward_witness(evidence,'log_probability',grad)))


def lora_b_gradients(actor):
    present=[];missing=0
    for name,p in actor.actor_module.named_parameters():
        if not p.requires_grad or '.lora_B.' not in name:continue
        if p.grad is None:missing+=1;continue
        grad=p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
        present.append((name,grad.detach()))
    require(present,'First-real backward produced no LoRA-B gradient tensors')
    finite=all(bool(torch.isfinite(grad).all()) for _,grad in present)
    result={'present_tensors':len(present),'missing_tensors':missing,
        'elements':sum(grad.numel() for _,grad in present),'finite':finite,
        'norm_l2':math.sqrt(sum(float(grad.double().square().sum()) for _,grad in present)),
        'parameter_names_sha256':hashlib.sha256(json.dumps([n for n,_ in present]).encode()).hexdigest(),
        'zero_norm_is_valid':True}
    require(finite,'Nonfinite first-real LoRA-B gradient')
    return result


def require_finite_update(actor,grad_norm):
    if getattr(actor,'caft_spec',None) is None:return
    require(bool(torch.isfinite(grad_norm).all()),'CAFT optimizer update cannot skip a nonfinite gradient norm')


def record_optimizer_step(actor,grad_norm):
    if getattr(actor,'caft_spec',None) is None:return
    if not getattr(actor,'_caft_update_recorded',False):
        print('CAFT_OPTIMIZER_RECEIPT '+json.dumps({'rank':rank(),'pid':os.getpid(),
            'arm':actor.caft_spec['arm'],'step':actor._caft_step,'optimizer_step_called':True,
            'grad_norm':float(grad_norm),'finite':True},sort_keys=True),flush=True)
        actor._caft_update_recorded=True


@contextmanager
def actor_scope(actor,micro_batch,*,role=None,step=None):
    configured=getattr(actor,'caft_spec',None) is not None
    if configured:require_precision()
    controller=getattr(actor,'caft',None)
    scope=(controller.scope(micro_batch,remove_padding=actor.use_remove_padding,
        sequence_parallel_size=actor.ulysses_sequence_parallel_size) if controller is not None else nullcontext())
    role='reference' if getattr(actor,'_caft_reference_depth',0) else role
    seen=getattr(actor,'_caft_evidence_roles',set())
    evidence=None
    if configured and role is not None:
        require(type(step) is int and step>0,'Actual training step missing from CAFT microbatch')
        actor._caft_step=step
        if role not in seen:
            require(role in ('old','current','reference'),'Unknown first-real actor role')
            mask=predictor_mask(micro_batch,remove_padding=actor.use_remove_padding,
                sequence_parallel_size=actor.ulysses_sequence_parallel_size)
            spec=actor.caft_spec
            evidence={'protocol':'caft_first_real_microbatch_v1','rank':rank(),'pid':os.getpid(),
                'arm':spec['arm'],'role':role,'step':step,'status':'started','precision':precision_state(),
                'q_tensor_sha256':spec['q']['tensor_sha256'] if spec['q'] else None,
                'input_ids_sha256':tensor_digest(micro_batch['input_ids']),
                'response_ids_sha256':tensor_digest(micro_batch['responses']),
                'predictor_mask_sha256':tensor_digest(mask),'predictor_mask_shape':list(mask.shape),
                'predictor_tokens':int(mask.sum()),'response_target_tokens':int(micro_batch['attention_mask'][:,-micro_batch['responses'].shape[1]:].sum()),
                'projection_hook_present':controller is not None,'layer_forward_calls':0,'projection_forward_calls':0,
                'reference_projection_disabled':role=='reference','_handles':[]}
            if controller is not None:
                require(controller.evidence is None,'Overlapping CAFT evidence scope')
                controller.evidence=evidence
    try:
        with scope:yield evidence
        if evidence is not None:
            expected=controller is not None and role!='reference'
            require((evidence['projection_forward_calls']>0) == expected,'Actual HF projection role/call mismatch')
            if controller is not None:require(evidence['layer_forward_calls']>0,'Actual HF decoder hook not reached')
            if role=='current':
                require(evidence.get('log_probability_backward_calls',0)>0,'Current log-probability backward not observed')
                if expected:require(evidence.get('projected_output_backward_calls',0)>0,'Projection backward not observed')
                evidence['lora_b_gradients']=lora_b_gradients(actor)
            evidence['status']='succeeded'
            actor._caft_evidence_roles=seen|{role}
    except BaseException as exc:
        if evidence is not None:evidence.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if evidence is not None:
            for handle in evidence.pop('_handles'):handle.remove()
            if controller is not None:controller.evidence=None
            print('CAFT_ACTOR_RECEIPT '+json.dumps(evidence,sort_keys=True),flush=True)


@contextmanager
def reference_scope(actor):
    controller=getattr(actor,'caft',None)
    actor._caft_reference_depth=getattr(actor,'_caft_reference_depth',0)+1
    try:
        with controller.disabled() if controller is not None else nullcontext():yield
    finally:actor._caft_reference_depth-=1
