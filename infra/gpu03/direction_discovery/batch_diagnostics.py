"""Bounded teacher-forced diagnostics on the first retained real generation batch.

No sampling campaign, model load, candidate fitting or statistical equivalence
claim. Drift between batch shapes is reported; same-shape causal invariance and
hook restoration use the fixed tolerance below, never an observed-data threshold.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from . import engine

POLICY = {'version':'retained_real_prefix_diagnostics_v1', 'retained_generated_tokens':2,
          'same_shape_max_abs':1e-5, 'cross_shape_drift':'descriptive_no_equivalence_claim',
          'no_additional_generated_requests':True}
SOURCES = ('batched_generation.py','batch_diagnostics.py','engine.py','intervention.py')


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in SOURCES}


def drift(a,b):
    engine.require(a.shape==b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Nonfinite/changed diagnostic logits')
    delta=a.float()-b.float();rms=float(delta.square().mean().sqrt())
    return {'max_abs':float(delta.abs().max()),'rms':rms,
            'relative_rms':rms/max(float(a.float().square().mean().sqrt()),1e-12),
            'top1_agreement':float((a.argmax(-1)==b.argmax(-1)).float().mean()),'values':a.numel()}


def binding(task,batch,rows,results,projection_condition_id):
    return {'task_sha256':digest(task),'source_sha256':source_hashes(),
            'batch_profile':task['batch_profile'],'policy':POLICY,
            'request_ids':[r['request_id'] for r in batch],
            'request_sha256':digest(batch),'prepared_rows_sha256':digest(rows),
            'generated_ids_sha256':digest([r['generated_token_ids'] for r in results]),
            'projection_condition_id':projection_condition_id,
            'projection_condition_sha256':digest(task['conditions'][projection_condition_id])}


def validate_receipt(receipt,expected):
    engine.require(all(receipt.get(k)==v for k,v in expected.items()),'Diagnostic provenance differs')
    engine.require(receipt['status']=='passed_real_prefix_checks' and receipt['distribution_equivalence_established'] is False,
                   'Missing real-prefix diagnostic result')
    checks=receipt['checks']
    required={'logical_positions','request_local_first_draw_replay','finite_hidden_including_padding',
              'eos_rows_excluded_from_sampling_and_energy','projection_predictor_counts','rng_unchanged_by_diagnostics'}
    engine.require(set(checks)==required and all(v is True for v in checks.values()),'Failed actual prefix checks')
    for key in ('baseline_restoration','future_prefix_invariance'):
        engine.require(receipt[key]['max_abs']<=POLICY['same_shape_max_abs'],'Same-shape causal/restoration drift exceeds fixed tolerance')
    for key in ('baseline_restoration','future_prefix_invariance','batch_vs_individual','cached_vs_teacher_forced'):
        d=receipt[key]
        engine.require(set(d)=={'max_abs','rms','relative_rms','top1_agreement','values'} and
            all(type(d[k]) in (int,float) and math.isfinite(d[k]) and d[k]>=0 for k in ('max_abs','rms','relative_rms')) and
            0<=d['top1_agreement']<=1 and type(d['values']) is int and d['values']>0,'Invalid measured logit drift')
    engine.require(type(receipt['projection_energy']) is dict and receipt['projection_energy'],'Projection diagnostic absent')
    for entry in receipt['projection_energy'].values():
        engine.require(entry['selected_tokens']==receipt['diagnostic_predictor_count'] and
            entry['activation_energy']>=0 and entry['removed_energy_fp32']>=0,'Projection coverage differs')
    return receipt


def run(model,decoder,layers,tokenizer,task,batch,rows,results,trace,projections,projection_condition_id,*,deadline_epoch,on_observation=None):
    """Only prefixes/tokens already retained in the real results journal are used."""
    from . import batched_generation as bg
    started=time.monotonic();expected=binding(task,batch,rows,results,projection_condition_id)
    engine.require(task['batch_diagnostics_policy']==POLICY and projections,'Explicit diagnostic policy/Q required')
    device=next(model.parameters()).device
    prefixes=[engine.prefix_for(row,request['scope'])[0] for row,request in zip(rows,batch)]
    # Two predictor positions at most per real row. EOS is retained but never fed
    # as a live continuation. A one-token EOS row still supplies its real prefill.
    tails=[r['generated_token_ids'][:1] if len(r['generated_token_ids'])>1 else [] for r in results]
    seqs=[p+t for p,t in zip(prefixes,tails)];width=max(map(len,seqs))+1
    ids=torch.full((len(rows),width),tokenizer.pad_token_id,dtype=torch.long,device=device)
    attention=torch.zeros_like(ids);mask=torch.zeros_like(ids,dtype=torch.bool);points=[]
    for i,(prefix,seq) in enumerate(zip(prefixes,seqs)):
        start=width-1-len(seq);ids[i,start:width-1]=torch.tensor(seq,device=device);attention[i,start:width-1]=1
        first=start+len(prefix)-1;mask[i,first:width-1]=True;points.append((i,first))
    position=(attention.cumsum(-1)-1).clamp_min(0)
    rng_before=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    def deadline():engine.require(time.time()<deadline_epoch,'Real-prefix diagnostics deadline exceeded')
    def forward(tokens,attn,pos,projection=None,selected=None,cache=None):
        deadline()
        with torch.inference_mode(),engine.ProjectionHooks(layers,projection or {},strength=engine.intervention_strength(task,task['conditions'][projection_condition_id])) as hook:
            with hook.positions(torch.zeros_like(tokens,dtype=torch.bool) if selected is None else selected,scope='teacher_forced'):
                output=decoder(input_ids=tokens,attention_mask=attn,position_ids=pos,past_key_values=None if cache is False else cache,use_cache=cache is not False,return_dict=True)
            h=output.last_hidden_state
            engine.require(bool(torch.isfinite(h).all()),'Nonfinite hidden states including padded/EOS rows')
            energy=hook.energy_report()
        return h,output.past_key_values if hasattr(output,'past_key_values') else None,energy
    def logits(hidden,locations):
        selected=torch.stack([hidden[i,j] for i,j in locations])
        value=engine.causal_model(model).lm_head(selected).float()
        engine.require(bool(torch.isfinite(value).all()),'Nonfinite diagnostic logits')
        return value
    with torch.inference_mode():
        baseline,_,_=forward(ids,attention,position,cache=False)
        base_logits=logits(baseline,points)
        _,_,energy=forward(ids,attention,position,projections,mask,cache=False)
        restored,_,_=forward(ids,attention,position,cache=False)
        restoration=drift(base_logits,logits(restored,points))
        # Change only real tokens strictly after each measured predictor, and an
        # appended future token; same physical shape/positions/attention both runs.
        future=ids.clone();future_attention=attention.clone()
        for i,j in points:
            future_attention[i,-1]=1
            future[i,j+1:]=int(prefixes[i][0])
        future_position=(future_attention.cumsum(-1)-1).clamp_min(0)
        reference,_,_=forward(ids,future_attention,future_position,cache=False)
        changed,_,_=forward(future,future_attention,future_position,cache=False)
        invariance=drift(logits(reference,points),logits(changed,points))
        individual=[]
        for prefix in prefixes:
            tokens=torch.tensor([prefix],dtype=torch.long,device=device);a=torch.ones_like(tokens);p=torch.arange(tokens.shape[1],device=device)[None,:]
            h,_,_=forward(tokens,a,p,cache=False);individual.append(logits(h,[(0,len(prefix)-1)]))
        batch_single=drift(base_logits,torch.cat(individual))
        # Reuse the exact prefill left-padding shape; all rows contribute their
        # retained first token. EOS rows append attention0 and are excluded from
        # compared decode predictors, matching the generation path.
        w=max(map(len,prefixes));prefill=torch.full((len(rows),w),tokenizer.pad_token_id,dtype=torch.long,device=device);att=torch.zeros_like(prefill)
        for i,prefix in enumerate(prefixes):prefill[i,-len(prefix):]=torch.tensor(prefix,device=device);att[i,-len(prefix):]=1
        pos=(att.cumsum(-1)-1).clamp_min(0);_,cache,_=forward(prefill,att,pos)
        active=torch.tensor([len(r['generated_token_ids'])>1 for r in results],dtype=torch.long,device=device)
        first=torch.tensor([[r['generated_token_ids'][0]] for r in results],device=device)
        next_att=torch.cat((att,active[:,None]),-1);next_pos=(next_att.sum(-1)-1).clamp_min(0)[:,None]
        h,_,_=forward(first,next_att,next_pos,cache=cache)
        full_ids=torch.cat((prefill,first),-1);full_pos=(next_att.cumsum(-1)-1).clamp_min(0)
        full,_,_=forward(full_ids,next_att,full_pos,cache=False)
        indices=[i for i,r in enumerate(results) if len(r['generated_token_ids'])>1]
        # Even an all-EOS batch has valid finite prefill diagnostics; no invented
        # decode token is generated. Compare prefill against itself in that case,
        # and explicitly preserve the zero-active-decode count.
        cached=drift(logits(h,[(i,0) for i in indices]),logits(full,[(i,w) for i in indices])) if indices else drift(base_logits,base_logits)
        replay=[int(engine.sample_token(trace['first_logits'][i:i+1],torch.Generator(device=device).manual_seed(request['seed']),task['sampling']).item()) for i,request in enumerate(batch)]
    logical=trace['logical_positions_verified'] and trace['eos_exclusion_verified']
    checks={'logical_positions':bool(logical),'request_local_first_draw_replay':replay==[r['generated_token_ids'][0] for r in results],
            'finite_hidden_including_padding':True,'eos_rows_excluded_from_sampling_and_energy':trace['eos_exclusion_verified'],
            'projection_predictor_counts':all(e['selected_tokens']==int(mask.sum()) for e in energy.values()),
            'rng_unchanged_by_diagnostics':torch.equal(rng_before,torch.get_rng_state()) and (cuda_rng is None or torch.equal(cuda_rng,torch.cuda.get_rng_state(device)))}
    receipt={**expected,'status':'passed_real_prefix_checks','distribution_equivalence_established':False,'checks':checks,
        'baseline_restoration':restoration,'future_prefix_invariance':invariance,'batch_vs_individual':batch_single,
        'cached_vs_teacher_forced':cached,'active_cached_decode_rows':len(indices),
        'projection_energy':energy,'diagnostic_predictor_count':int(mask.sum()),'elapsed_seconds':time.monotonic()-started}
    try:
        validate_receipt(receipt,expected)
    except BaseException as exc:
        receipt['status']='failed_real_prefix_checks';receipt['error']=type(exc).__name__+': '+str(exc)
        if on_observation is not None:on_observation(receipt)
        raise
    if on_observation is not None:on_observation(receipt)
    return receipt
