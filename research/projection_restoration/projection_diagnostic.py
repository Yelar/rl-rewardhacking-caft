"""Exact checkpoint100 OFF/ON declaration; no sampling or evaluator replacement."""
from pathlib import Path

SCHEMA = 'random0_checkpoint100_projection_restoration_v1'
LONG_SCHEMA = 'random0_checkpoint100_projection_restoration_long3072_v1'
CONDITIONS = (('random0_off', 'off'), ('random0_on', 'on'))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def validate_declaration(m):
    d = m['projection_restoration']
    require(d['kind'] in (SCHEMA, LONG_SCHEMA) and d['kind'] == m['schema'] and d['source_arm'] == 'random0' and d['checkpoint_step'] == 100,
            'Only original Random0 checkpoint100 restoration is admitted')
    if d['kind'] == LONG_SCHEMA:
        require(d.get('length_profile') == dict(max_prompt_tokens=1536, max_completion_tokens=3072, max_model_len=4608),
                'Long diagnostic requires exact 1536/3072/4608 lengths')
        reference = d.get('reference1536_manifest', {})
        require(set(reference) == {'path', 'sha256', 'size_bytes'} and reference['sha256'] ==
                '3e2b7d9958790adb1475e65a927dc478fd034babbe0f168f20a462b3e2a06ebc',
                'Require exact completed 1536-token reference, never reused rows')
    else:
        require('length_profile' not in d and 'reference1536_manifest' not in d,
                'Long profile cannot be silently enabled in the original schema')
    require(d['direction'] == 'original_random_0' and type(d['rank']) is int and d['rank'] == 1
            and type(d['layer']) is int and d['layer'] == 21 and type(d['alpha']) in (int,float)
            and d['alpha'] == 1, 'Projection differs from saved rank1/layer21/alpha1')
    require(d['conditions'] == ['off', 'on'] and d['fresh_off'] is True,
            'Require fresh matched OFF and ON, without another condition')
    require(d['mask_timing'] == 'training_response_predictor' and d['materialization_control'] is False,
            'Wrong projection timing or an unrequested third condition')
    require([(a['arm'], a['step']) for a in m['adapters']] == [('random0_off',100),('random0_on',100)],
            'Exactly four full cells in OFF then ON order are required')
    off,on = m['adapters']
    require(off['files'] == on['files'] and off['path'] == on['path'],
            'OFF and ON must use exactly the same checkpoint bytes and path')
    require(m['previous_evaluations'] == [], 'This is a fresh four-cell diagnostic, not a reuse campaign')
    require(all(isinstance(d[k],dict) and set(d[k]) >= {'path','sha256','size_bytes'}
                for k in ('owner_authorization','q_bank','source_checkpoint_manifest')),
            'Exact owner, saved Q and original checkpoint references required')
    expected_hook = dict(arm='random0', evaluation_projection=False, layer=21,
        protocol='qwen3_l21_response_predictor_caft_v1', reference_projection=False,
        rollout_enforce_eager=False, rollout_prefix_caching=True, scope='response_predictors',
        source_condition_id='stage9-original-random-0-alpha100', strength=1.0,
        q=dict(dtype='float32', file=d['q_bank'], key='original_random_0', shape=[2560,1],
               tensor_sha256='d323d65c9c86479c2d47ba1a28d0832b637cfbf44b10c87c190faf52699ca9d6'))
    require(d['hook'] == expected_hook, 'Hook differs from original saved training specification')
    require(d['q_bank']['sha256'] == '1dacdee17fd2eab2cf6f26bd37812b06ec62346d07f51b3f690c0e1ab5a52498'
            and d['q_bank']['size_bytes'] == 1361664, 'Wrong saved Q bank')
    require(d['source_checkpoint_manifest']['sha256'] ==
            'f3679612f4e942eac5b10daba5356f5055e3d713d57082ad5fba06d5ec29bdab',
            'Wrong original checkpoint100 seal')
    require(off['files']['adapter_model.safetensors']['sha256'] ==
            '62b03a90bcd14e213abc5e989465c3b381515476adf831e07538e5a4385f796b'
            and off['files']['adapter_config.json']['sha256'] ==
            '276c11b775dfb5544a09e12155898ae66a96cfddba39e06d83384296a49f6cce',
            'Wrong original checkpoint100 adapter')
    return d

def verify_declaration_inputs(m, check_ref):
    d = validate_declaration(m)
    for key in ('owner_authorization','q_bank','source_checkpoint_manifest'):
        check_ref(d[key])
    if d['kind'] == LONG_SCHEMA:
        check_ref(d['reference1536_manifest'])

def condition(adapter):
    result = dict(CONDITIONS).get(adapter['arm'])
    require(result is not None and adapter['step'] == 100, 'Wrong diagnostic condition')
    return result


def engine_kwargs(m, adapter):
    from projection_restoration_hook import engine_kwargs as native_kwargs
    d = validate_declaration(m)
    return native_kwargs(d['hook'], condition(adapter))


def attach(engine, m, adapter):
    from projection_restoration_hook import attach as native_attach
    return native_attach(engine, m['projection_restoration']['hook'], condition(adapter))


def scope(engine):
    from projection_restoration_hook import scope as native_scope
    return native_scope(engine)


def validate_receipt(m, adapter, box):
    from projection_restoration_hook import validate_receipt as native_validate
    require(isinstance(box, dict) and isinstance(box.get('receipt'), dict),
            'Missing final projection receipt; retain raw but do not seal')
    return native_validate(box['receipt'], m['projection_restoration']['hook'], condition(adapter))
