"""Versioned same-condition HF batching; existing batch-one engine stays unchanged.

One model worker owns one GPU. Batch shape is a new numerical profile and requires
qualification on retained real requests before a full map; authored tests do not
establish BF16 equivalence or throughput. Completion code is never executed here.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from . import engine

PROTOCOL = 'same_condition_batched_generation_v1'
PROFILE = {'protocol': PROTOCOL, 'batch_size': 8, 'left_padding': True,
           'attention_policy': 'exclusive_math', 'adapter_policy': 'unmerged',
           'sampling_policy': 'existing_request_local_generator',
           'projection_policy': 'existing_fp32_same_condition',
           'qualified_numerical_equivalence': False}
FIELDS = ('activation_energy', 'removed_energy_fp32', 'actual_change_energy',
          'remaining_subspace_energy_fp32', 'remaining_subspace_energy_native')


def profile(value):
    expected = {**PROFILE, 'batch_size': value.get('batch_size')}
    engine.require(value == expected and type(value['batch_size']) is int and
                   value['batch_size'] in (1, 2, 4, 8), 'Unreviewed batch profile')
    return dict(value)


class RowProjectionHooks(engine.ProjectionHooks):
    """Delegate replacement/guards to the existing hook; add row-only diagnostics."""
    def __init__(self, *args, batch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.row_energy = {}
        self.row_counts = {}

    def _make_hook(self, index):
        original = super()._make_hook(index)
        def hook(module, inputs, output):
            replacement = original(module, inputs, output)
            if replacement is None:return None
            before = output[0] if isinstance(output, tuple) else output
            after = replacement[0] if isinstance(replacement, tuple) else replacement
            mask = self._mask
            if before.shape[0] != self.batch_size or bool(mask[:, :-1].any().item()):
                raise ValueError('Batched generation only projects each active row final predictor')
            active = mask[:, -1]
            # Match the base hook's exact active-row GEMM shapes after EOS.
            selected = before[mask].float()
            native = after[mask].float()
            q = self.projections[index]
            removed = torch.zeros_like(selected) if self.strength == 0 else (selected @ q) @ q.T
            if self.strength not in (0., 1.):removed = removed * self.strength
            projected = selected - removed
            selected_values = torch.stack([selected.square().sum(-1), removed.square().sum(-1),
                (selected-native).square().sum(-1), (projected@q).square().sum(-1),
                (native@q).square().sum(-1)], dim=-1).detach().double()
            values = torch.zeros((self.batch_size,5),dtype=torch.float64,device=before.device)
            values[active] = selected_values
            for scope in ('all', self._scope):
                key = (index, scope)
                if key not in self.row_energy:
                    self.row_energy[key] = torch.zeros_like(values)
                    self.row_counts[key] = torch.zeros(self.batch_size, dtype=torch.long, device=values.device)
                self.row_energy[key] += values
                self.row_counts[key] += active.long()
            return replacement
        return hook

    def energy_by_row(self):
        if self._mask is not None:raise RuntimeError('Energy read during forward')
        result = [{} for _ in range(self.batch_size)]
        for index, q in self.projections.items():
            for scope in ('all', 'prefill', 'decode'):
                key = (index, scope)
                values = self.row_energy[key].cpu().tolist() if key in self.row_energy else [[0.]*5 for _ in result]
                counts = self.row_counts[key].cpu().tolist() if key in self.row_counts else [0]*len(result)
                for i, (v, n) in enumerate(zip(values, counts)):
                    row = dict(zip(FIELDS, v));row.update(forward_calls=n, selected_tokens=n, strength=self.strength)
                    if scope == 'all':
                        row.update(rank=q.shape[1], projection_dtype='float32',
                            removed_fraction=v[1]/v[0] if v[0] else None,
                            actual_change_fraction=v[2]/v[0] if v[0] else None, scopes={})
                        result[i][str(index)] = row
                    elif n:result[i][str(index)]['scopes'][scope] = row
        return result


def generate_batch(model, layers, tokenizer, rows, projections, scopes, seeds, sampling,
                   *, batch_profile, strength=1., deadline_epoch=None, on_completed=None, on_partial=None, diagnostic_trace=None):
    """Same condition/Q for every row; emit ordinary engine result fields plus timing.

    The unchanged sampler still checks finite logits for each active row. Token
    IDs/EOS are copied to CPU together once per step, rather than once per row.
    Finished rows stay in the physical KV batch, masked from energy and sampling.
    """
    cfg = profile(batch_profile);n = len(rows)
    engine.require(0 < n <= cfg['batch_size'] and len(scopes)==len(seeds)==n, 'Batch dimensions differ')
    engine.require(all(type(seed) is int and 0 <= seed < 2**63 for seed in seeds), 'Invalid request seed')
    strength = engine.validate_strength(strength)
    prefixes, fixed, budgets = zip(*(engine.prefix_for(row, scope) for row, scope in zip(rows, scopes)))
    device = next(model.parameters()).device
    width = max(map(len, prefixes));pad = tokenizer.pad_token_id
    engine.require(type(pad) is int and pad >= 0, 'Explicit tokenizer pad ID required')
    ids = torch.full((n, width), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for i, prefix in enumerate(prefixes):
        ids[i, -len(prefix):] = torch.tensor(prefix, dtype=torch.long, device=device)
        attention[i, -len(prefix):] = 1
    position = (attention.cumsum(-1)-1).clamp_min(0)
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    generated = [[] for _ in rows];active = [True]*n;elapsed=[None]*n
    cache = None;current=ids;started=time.monotonic()
    if diagnostic_trace is not None:
        diagnostic_trace.update(logical_positions_verified=True,eos_exclusion_verified=True)
    hooks = RowProjectionHooks(layers, projections, batch_size=n, strength=strength)
    def result_for(i, energy, batch_wall):
        ids_out=generated[i];complete=list(fixed[i])+ids_out
        return {'completion_token_ids':complete,'completion':tokenizer.decode(complete,
            skip_special_tokens=True,clean_up_tokenization_spaces=False),'generated_token_ids':list(ids_out),
            'fixed_completion_prefix_token_count':len(fixed[i]),
            'stop_reason':'eos' if ids_out[-1] in sampling['eos_token_ids'] else 'length',
            'elapsed_seconds':elapsed[i],'energy':energy,'intervention_strength':strength,
            'batch_profile':cfg,'batch_size_actual':n,'batch_wall_seconds':batch_wall,
            'elapsed_scope':'row latency from shared batch start; do not sum as GPU work'}
    try:
        with torch.inference_mode(), hooks:
            for step in range(max(budgets)):
                if deadline_epoch is not None:engine.require(time.time() < deadline_epoch, 'Batched generation deadline exceeded')
                active_tensor = torch.tensor(active, dtype=torch.bool, device=device)
                mask = torch.zeros_like(current, dtype=torch.bool);mask[:, -1] = active_tensor
                if diagnostic_trace is not None:
                    expected_position=(attention.cumsum(-1)-1).clamp_min(0) if step==0 else (attention.sum(-1)-1).clamp_min(0)[:,None]
                    diagnostic_trace['logical_positions_verified'] &= bool(torch.equal(position,expected_position))
                    diagnostic_trace['eos_exclusion_verified'] &= bool(torch.equal(mask[:,-1],active_tensor))
                with hooks.positions(mask, scope='prefill' if step==0 else 'decode'):
                    output=model(input_ids=current, attention_mask=attention, position_ids=position,
                        past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
                if diagnostic_trace is not None and step==0:
                    diagnostic_trace['first_logits']=output.logits[:,-1,:].detach().clone()
                # Preserve exact per-row warpers and request-local multinomial RNG.
                tokens = torch.cat([engine.sample_token(output.logits[i:i+1, -1, :], generators[i], sampling)
                                    if active[i] else torch.full((1,1),pad,dtype=torch.long,device=device)
                                    for i in range(n)], dim=0)
                token_ids=tokens[:, 0].cpu().tolist()  # One EOS/ID synchronization for the batch.
                cache=output.past_key_values
                newly_done=[]
                for i, token in enumerate(token_ids):
                    if not active[i]:continue
                    generated[i].append(token)
                    if token in sampling['eos_token_ids'] or len(generated[i])==budgets[i]:
                        active[i]=False;elapsed[i]=time.monotonic()-started;newly_done.append(i)
                if on_completed is not None and newly_done:
                    snapshot=hooks.energy_by_row()
                    for i in newly_done:on_completed(i, result_for(i,snapshot[i],None))
                if not any(active):break
                current=tokens
                active_tensor=torch.tensor(active,dtype=torch.long,device=device)
                attention=torch.cat((attention,active_tensor[:,None]),dim=1)
                # Logical per-row positions exclude left padding and inactive tail.
                position=(attention.sum(-1)-1).clamp_min(0)[:,None]
            energies=hooks.energy_by_row()
    except BaseException:
        if on_partial is not None:on_partial([{'generated_token_ids':list(g), 'completed':elapsed[i] is not None}
                                              for i,g in enumerate(generated)])
        raise
    batch_wall=time.monotonic()-started
    engine.require(all(generated) and all(v is not None for v in elapsed),'Incomplete batch result')
    return [result_for(i,energies[i],batch_wall) for i in range(n)]


def request_batches(requests, batch_size):
    """Stable first-condition order and original order within each condition."""
    engine.require(type(batch_size) is int and batch_size in (1,2,4,8),'Invalid batch size')
    engine.require(len({r['request_id'] for r in requests})==len(requests),'Duplicate requests')
    groups=OrderedDict()
    for request in requests:groups.setdefault(request['condition_id'],[]).append(request)
    return [rows[start:start+batch_size] for rows in groups.values() for start in range(0,len(rows),batch_size)]


def validate_qualification(task, cfg):
    """A finite retained first-real tranche, or positive same-profile evidence."""
    stage=task['batch_stage']
    if stage=='integrated_real_requests':
        from . import batch_diagnostics as diagnostics
        engine.require(task['batch_diagnostics_policy']==diagnostics.POLICY,'Unreviewed real-prefix diagnostic policy')
        key=task['diagnostic_projection_condition_id']
        engine.require(key in task['conditions'] and task['conditions'][key].get('layers'),'Diagnostic projection must be an actual requested condition')
        engine.require(any(r['condition_id']==key for r in task['requests']),'Diagnostic Q is not part of worker requests')
        return
    if stage=='retained_first_real_requests':
        engine.require(0 < len(task['requests']) <= 2*cfg['batch_size'], 'First-real tranche exceeds two batches per worker')
        return
    engine.require(stage=='qualified_run','Unknown batched execution stage')
    ref=task['batch_qualification'];p=Path(ref['path']);data=p.read_bytes()
    engine.require(not p.is_symlink() and hashlib.sha256(data).hexdigest()==ref['sha256'], 'Batch qualification changed')
    proof=json.loads(data)
    expected_sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                      for name in ('batched_generation.py','batch_diagnostics.py','engine.py','intervention.py')}
    engine.require(proof['status']=='qualified_real_request_batched_generation_v1' and
        proof['batch_profile']==cfg and proof['source_sha256']==expected_sources and
        proof['real_request_results_retained'] is True and proof['sampling_and_projection_verified'] is True and
        proof['memory_within_profile'] is True and proof['actual_exit_and_release_verified'] is True,
        'Missing positive real-request numerical/profile qualification')


def worker(task_path):
    """Ordinary engine task fields plus explicit batch profile/stage; no supervisor."""
    task=json.loads(Path(task_path).read_bytes());cfg=profile(task['batch_profile'])
    engine.require(task['mode']=='generate_batch_v1','Wrong worker mode')
    engine.require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(task['gpu_id']), 'GPU differs from task')
    engine.require(task['attention_policy']=='exclusive_math','Batched path keeps math SDPA')
    fraction=engine.gpu_memory_fraction(task);validate_qualification(task,cfg)
    batches=request_batches(task['requests'],cfg['batch_size'])
    if task['batch_stage']=='retained_first_real_requests':engine.require(len(batches)<=2,'First-real tranche exceeds two batches')
    rows={r['record_id']:r for r in engine.read_jsonl(task['prepared_records'])}
    for request in task['requests']:
        engine.validate_row(rows[request['record_id']]);engine.intervention_strength(task,task['conditions'][request['condition_id']])
    engine.require(type(task['deadline_seconds']) in (int,float) and 0 < task['deadline_seconds'] <= 86400,'Invalid worker deadline')
    output=Path(task['output']);output.mkdir(parents=True,exist_ok=False)
    engine.raw.exclusive_json(output/'task.json',task)
    started=time.monotonic();deadline=time.time()+task['deadline_seconds']
    engine.raw.configure_torch(gpu=True)
    torch.cuda.set_per_process_memory_fraction(fraction,0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    engine.require(torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled() and
        not torch.backends.cuda.mem_efficient_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled(), 'Math SDPA not exclusive')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(task['model_snapshot'],local_files_only=True)
    model=decoder=layers=q=None
    try:
        engine.require(time.time()<deadline,'Deadline before model load')
        model,decoder,layers,load=engine.legacy._load_decoder(task,with_adapter=True)
        for batch_index, batch in enumerate(batches):
            condition=task['conditions'][batch[0]['condition_id']]
            q=engine.load_projections(condition,next(model.parameters()).device)
            def completed(i,result):
                request=batch[i];row=rows[request['record_id']]
                engine.raw.append(output/'results.jsonl',{**request,'problem_id':row['problem_id'],
                    'problem_split':row['problem_split'],'original_class':row['outcome_presence_class'],
                    'batch_index':batch_index,'result':result})
            def partial(records):
                for request,record in zip(batch,records):
                    engine.raw.append(output/'partial_generations.jsonl',{**request,'batch_index':batch_index,**record})
            trace={} if task['batch_stage']=='integrated_real_requests' and batch_index==0 else None
            results=generate_batch(model,layers,tokenizer,[rows[r['record_id']] for r in batch],q,
                [r['scope'] for r in batch],[r['seed'] for r in batch],task['sampling'],batch_profile=cfg,
                strength=engine.intervention_strength(task,condition),deadline_epoch=deadline,
                on_completed=completed,on_partial=partial,diagnostic_trace=trace)
            engine.raw.append(output/'batches.jsonl',{'batch_index':batch_index,
                'request_ids':[r['request_id'] for r in batch], 'batch_wall_seconds':results[0]['batch_wall_seconds'],
                'generated_tokens':sum(len(r['generated_token_ids']) for r in results)})
            if trace is not None:
                from . import batch_diagnostics as diagnostics
                key=task['diagnostic_projection_condition_id'];dq=engine.load_projections(task['conditions'][key],next(model.parameters()).device)
                receipt=diagnostics.run(model,decoder,layers,tokenizer,task,batch,[rows[r['record_id']] for r in batch],
                    results,trace,dq,key,deadline_epoch=deadline,
                    on_observation=lambda observed:engine.raw.exclusive_json(output/'real_prefix_diagnostics.json',observed))
                del dq,trace
            q=None;del results
    finally:
        del model,decoder,layers,q
        engine.legacy._release_cuda()
    engine.raw.exclusive_json(output/'SUCCESS.json',{'status':'succeeded','run_token':task['run_token'],
        'worker_name':task['worker_name'],'requests':len(task['requests']),'mode':task['mode'],
        'elapsed_seconds':time.monotonic()-started,'model_load_reports':load,'batch_profile':cfg,
        'generation_distribution_equivalence_not_established_by_unit_tests':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True)
    worker(parser.parse_args().task)
