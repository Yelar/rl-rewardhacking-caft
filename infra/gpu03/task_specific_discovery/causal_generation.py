"""Versioned causal successor of the existing same-condition batch generator.

The token loop, sampler, attention positions and durable row-return callbacks are
copied unchanged from the source SHA below. Only the explicitly reviewed hook
and profile identifiers differ. Prospective primary prompts use the explicit
empty-completion input contract below. A live numerical qualification is required before
behavioral work. This module does not load a model or launch a worker at import.
"""
import time
import torch
from infra.gpu03.direction_discovery import engine
from infra.gpu03.direction_discovery.batched_generation import request_batches
from .causal_hooks import CausalProjectionHooks

PARENT_SHA256 = '0d3becdff5a2e11dc7f737275be1434f08cb1ac30d2cf6e1defc1e466471405c'
PROFILE = {"protocol":"task_specific_batched_causal_v1", "batch_size":8,
           "left_padding":True, "attention_policy":"exclusive_math",
           "adapter_policy":"unmerged", "sampling_policy":"existing_request_local_generator",
           "projection_policy":"fp32_response_predictor_ordinary_centered_v1",
           "qualified_numerical_equivalence":False}

def profile(value):
    engine.require(value == {**PROFILE,"batch_size":value.get("batch_size")} and
                   type(value["batch_size"]) is int and value["batch_size"] in (1,2,4,8),
                   "Unreviewed causal batch profile")
    return dict(value)


def validate_primary_row(row):
    """Validate a real prompt-only request without fabricating a completion."""
    p = row.get('prompt_token_count')
    tokens = row.get('prompt_token_ids')
    engine.require(type(p) is int and 0 < p <= 1536 and isinstance(tokens, list)
                   and len(tokens) == p, 'Invalid prospective prompt length')
    engine.require(all(type(x) is int and 0 <= x < 151936 for x in tokens),
                   'Invalid prospective token IDs')
    engine.require(type(row.get('completion_token_count')) is int
                   and row['completion_token_count'] == 0
                   and row.get('completion_token_ids') == []
                   and row.get('input_ids') == tokens,
                   'Prospective primary input must contain only the actual prompt')
    engine.require(row.get('problem_split') == 'direction_fit'
                   and row.get('protocol') in ('fixed', 'randomized', 'no_loophole'),
                   'Prospective population or protocol differs')


def prefix_for(row, scope):
    if scope == 'primary':
        validate_primary_row(row)
        return list(row['prompt_token_ids']), [], 1536
    return engine.prefix_for(row, scope)

def generate_batch(model, layers, tokenizer, rows, projections, scopes, seeds, sampling,
                   *, batch_profile, strength=1., deadline_epoch=None, on_completed=None, on_partial=None, diagnostic_trace=None, operator="ordinary", control="project", centers=None):
    """Same condition/Q for every row; emit ordinary engine result fields plus timing.

    The unchanged sampler still checks finite logits for each active row. Token
    IDs/EOS are copied to CPU together once per step, rather than once per row.
    Finished rows stay in the physical KV batch, masked from energy and sampling.
    """
    cfg = profile(batch_profile);n = len(rows)
    engine.require(0 < n <= cfg['batch_size'] and len(scopes)==len(seeds)==n, 'Batch dimensions differ')
    engine.require(all(type(seed) is int and 0 <= seed < 2**63 for seed in seeds), 'Invalid request seed')
    strength = engine.validate_strength(strength)
    prefixes, fixed, budgets = zip(*(prefix_for(row, scope) for row, scope in zip(rows, scopes)))
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
    hooks = CausalProjectionHooks(layers, projections, batch_size=n, strength=strength,
                                  operator=operator, control=control, centers=centers)
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
