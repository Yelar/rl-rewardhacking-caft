"""Direction v7 step A5, rotation 1 (protocol section 5 A5, amendment 2): the run plan, its input hashes, and one run
list per GPU; also verifies a plan's inputs on another host.

Held-out loophole: overwrite. Training sources: modify step 70 and incontext step 100 (their fit bulk samples; their
val bulk samples for early stopping). Variants R and RA at blocks 11, 15 and 19, from three starts: random, the pooled
mean difference (meandiff_v7.py, key 'meandiff') and the adding direction A (block 19: rl_banks/add_b19.npz 'unit';
blocks 11 and 15: train_v1/A_base_b<b>/bank.npz 'vec', normalised by the trainer). RA's adding size is the norm of the
v6 model-difference vector (modeldiff_v1/vectors.npz 'mrh|answer') at the block, as in v6. The cosine with the joint
direction (rl_banks/joint_b19.npz 'unit') is logged. Workers (run_train_queue_v7.sh) on both hosts share the list
and claim runs in order through CLAIMS on the shared home.

The plan comes in two parts, so the runs that need no mean-difference vector can start before step A4 ends:
  part A: the A starts (6 runs) and the random starts at block 19 (2 runs);
  part B: the mean-difference starts (6 runs) and the random starts at blocks 11 and 15 (4 runs).
Within each part the RA runs (about twice as long) come first; the R runs from the random start at blocks 11 and 15
come last overall (amendment 6).
Workers take part A's list before part B's (run_train_queue_v7.sh with both lists, in that order).
Rotations 2-3 and the all-three run (plan2) train the configuration chosen by select_v7.py (variant, block, start
kind) with overwrite in training: rotation 2 holds out modify (overwrite step-60 + incontext step 100), rotation 3
holds out incontext (overwrite step-60 + modify step 70), and the all-three run trains on all three. A mean-difference
start uses each run's own vector (meandiff_v7.py on that run's training models).
Usage: a5_plan.py plan A <kit dir> <out dir> <host,host,...>
       a5_plan.py plan B <kit dir> <out dir> <host,host,...> <meandiff npz>   (each writes runs.txt and A5_PLAN.json)
       a5_plan.py plan2 <SELECT.json> <kit dir> <out dir> <host,host,...> [<rot2 meandiff>,<rot3 meandiff>,<all meandiff>]
       a5_plan.py perloop <kit dir> <out dir> <host,host,...>
perloop (exploratory, after H1 did not pass; protocol section 5 A4: "per-loophole directions continue as exploratory
work only"): one direction per loophole, trained on that loophole's first chosen model alone (its fit samples; its val
samples for early stopping), R and RA at block 19 from the A start, with the same trainer and settings.
       a5_plan.py verify <out dir>/A5_PLAN.json        (recomputes every input hash on this host; exit 1 on mismatch)"""
import hashlib
import json
import sys
from pathlib import Path

B = '/scratch/yelarys.yertaiuly/bestcase-step4'
H = '/home/yelarys.yertaiuly/bestcase-ws-runs/causal'
E = f'{B}/v7a/evals'
A = f'{B}/v7a/adapters'
BASE = '/scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base'
SOURCES = (('modify', 'md_s070'), ('incontext', 'ic_s100'))
TRAIN_MODEL = {'overwrite': 'ow_step60', 'modify': 'md_s070', 'incontext': 'ic_s100'}     # amendment 2
ROT2 = {'rot2_hold-modify': ('overwrite', 'incontext'), 'rot3_hold-incontext': ('overwrite', 'modify'),
        'all3': ('overwrite', 'modify', 'incontext')}
ADD = {11: f'{H}/train_v1/A_base_b11/bank.npz:vec', 15: f'{H}/train_v1/A_base_b15/bank.npz:vec',
       19: f'{H}/rl_banks/add_b19.npz:unit'}
SIZE = f'{H}/modeldiff_v1/vectors.npz:mrh|answer'
JOINT = f'{H}/rl_banks/joint_b19.npz:unit'
CLAIMS = '/home/yelarys.yertaiuly/bestcase-ws-runs/v7a_claims/a5_rot1'
KIT_FILES = ('train_v7.py', 'run_train_v7.sh', 'run_train_queue_v7.sh', 'gpu_guard_run.sh', 'prestart_check.sh')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def spec(split, sources=SOURCES):
    return ','.join(f'{lp}={A}/{ck}@{E}/bulk_{ck}_{split}@{E}/bulk_{ck}_{split}' for lp, ck in sources)


def inputs(meandiff, kit, sources=SOURCES):
    files = [f'{BASE}/config.json', ADD[11].rsplit(':', 1)[0], ADD[15].rsplit(':', 1)[0],
             ADD[19].rsplit(':', 1)[0], SIZE.rsplit(':', 1)[0], JOINT.rsplit(':', 1)[0]] + ([meandiff] if meandiff else [])
    files += [f'{kit}/{f}' for f in KIT_FILES]
    for lp, ck in sources:
        files += [f'{A}/{ck}/adapter_model.safetensors', f'{A}/{ck}/adapter_config.json']
        for split in ('fit', 'val'):
            d = Path(E, f'bulk_{ck}_{split}')
            files += [str(d / 'MANIFEST.json'), str(d / 'BEHAVIOUR.json'), str(d / 'SUMMARY.json')]
            for st in (lp, 'nocue'):
                files += [str(d / 'cells' / f'{ck}__{st}' / 'raw.jsonl'), str(d / 'cells' / f'{ck}__{st}' / 'SCORE_COMPLETE.json')]
            files.append(str(d / 'LABELS' / f'{ck}__{lp}.jsonl'))
    return {f: sha(f) for f in files}


def plan(part, kit, outdir, hosts, meandiff=None):
    assert part in ('A', 'B') and (meandiff is None) == (part == 'A'), 'part A takes no mean-difference vector, part B needs one'
    runs = []
    for start in ('add', 'meandiff', 'random'):
        for block in (19, 15, 11):
            for variant in ('R', 'RA'):
                s = {'add': ADD[block], 'meandiff': f'{meandiff}:meandiff', 'random': 'random'}[start]
                args = ['--model', BASE, '--source', spec('fit'), '--val-dirs', spec('val'), '--block', str(block),
                        '--variant', variant, '--start', s, '--joint', JOINT]
                if variant == 'RA':
                    args += ['--size-from', SIZE]
                runs.append({'name': f'rot1_hold-overwrite_{variant}_b{block}_{start}', 'args': args})
    in_a = lambda r: r['name'].endswith('_add') or r['name'].endswith('_b19_random')        # noqa: E731
    runs = [r for r in runs if in_a(r) == (part == 'A')]
    # order (amendment 6): the longer RA runs first, so the last runs to start are short; the R runs from the random
    # start at blocks 11 and 15 come last, the first dropped if the cap binds
    is_ra = lambda r: '_RA_' in r['name']                                                    # noqa: E731
    late = lambda r: r['name'].endswith('_random') and ('_b11_' in r['name'] or '_b15_' in r['name'])   # noqa: E731
    runs = sorted(runs, key=lambda r: (not is_ra(r), late(r)))
    # one list, identical on both hosts; workers on either host claim runs in order (claims on the shared home)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / 'runs.txt').write_text(''.join('\t'.join([f'{B}/v7a/train/{r["name"]}', *r['args']]) + '\n' for r in runs))
    record = {'protocol': 'protocols/20261003-direction-v7 (section 5 A5, amendments 2-3), rotation 1, held-out overwrite',
              'part': part,
              'kit': kit, 'runs': runs, 'claims': CLAIMS, 'hosts': hosts,
              'inputs_sha256': inputs(meandiff, kit),
              'lists_sha256': {p.name: sha(p) for p in sorted(outdir.glob('*.txt'))}}
    (outdir / 'A5_PLAN.json').write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print('\n'.join(r['name'] for r in runs))


def plan2(select, kit, outdir, hosts, meandiffs=None):
    sel = json.loads(Path(select).read_text())
    c = sel['chosen_config']
    variant, block, kind = c['variant'], int(c['block']), c['start_kind']
    assert (kind == 'meandiff') == (meandiffs is not None), 'a mean-difference start needs the three vectors, others none'
    md = dict(zip(ROT2, meandiffs.split(','))) if meandiffs else {}
    runs, files = [], {}
    for name, lps in ROT2.items():
        sources = tuple((lp, TRAIN_MODEL[lp]) for lp in lps)
        start = {'add': ADD[block], 'random': 'random', 'meandiff': f'{md.get(name)}:meandiff'}[kind]
        args = ['--model', BASE, '--source', spec('fit', sources), '--val-dirs', spec('val', sources), '--block', str(block),
                '--variant', variant, '--start', start, '--joint', JOINT]
        if variant == 'RA':
            args += ['--size-from', SIZE]
        runs.append({'name': f'{name}_{variant}_b{block}_{kind}', 'args': args})
        files.update(inputs(md.get(name), kit, sources))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / 'runs.txt').write_text(''.join('\t'.join([f'{B}/v7a/train/{r["name"]}', *r['args']]) + '\n' for r in runs))
    record = {'protocol': 'protocols/20261003-direction-v7 (section 5 A5, amendments 2-3), rotations 2-3 and all three',
              'part': 'rotations 2-3 + all three', 'select': select, 'select_sha256': sha(select), 'chosen': sel['chosen'],
              'chosen_config': c, 'kit': kit, 'runs': runs, 'claims': CLAIMS.replace('a5_rot1', 'a5_rot23'),
              'hosts': hosts, 'inputs_sha256': files, 'lists_sha256': {'runs.txt': sha(outdir / 'runs.txt')}}
    (outdir / 'A5_PLAN.json').write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print('\n'.join(r['name'] for r in runs))


def perloop(kit, outdir, hosts):
    runs, files = [], {}
    for variant in ('RA', 'R'):
        for lp in ('overwrite', 'modify', 'incontext'):
            sources = ((lp, TRAIN_MODEL[lp]),)
            args = ['--model', BASE, '--source', spec('fit', sources), '--val-dirs', spec('val', sources), '--block', '19',
                    '--variant', variant, '--start', ADD[19], '--joint', JOINT]
            if variant == 'RA':
                args += ['--size-from', SIZE]
            runs.append({'name': f'perloop_{lp}_{variant}_b19_add', 'args': args})
            files.update(inputs(None, kit, sources))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / 'runs.txt').write_text(''.join('\t'.join([f'{B}/v7a/train/{r["name"]}', *r['args']]) + '\n' for r in runs))
    record = {'protocol': 'protocols/20261003-direction-v7: exploratory per-loophole directions after H1 did not pass '
                          '(section 5 A4)', 'part': 'perloop (exploratory)', 'kit': kit, 'runs': runs,
              'claims': CLAIMS.replace('a5_rot1', 'a5_perloop'), 'hosts': hosts, 'inputs_sha256': files,
              'lists_sha256': {'runs.txt': sha(outdir / 'runs.txt')}}
    (outdir / 'A5_PLAN.json').write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print('\n'.join(r['name'] for r in runs))


def verify(path):
    record = json.loads(Path(path).read_text())
    bad = []
    for f, h in record['inputs_sha256'].items():
        try:
            if sha(f) != h:
                bad.append(f'mismatch {f}')
        except OSError as e:
            bad.append(f'missing {f}: {e}')
    for name, h in record['lists_sha256'].items():
        if sha(Path(path).parent / name) != h:
            bad.append(f'list mismatch {name}')
    print('\n'.join(bad) if bad else f'verified {len(record["inputs_sha256"])} inputs and {len(record["lists_sha256"])} lists')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    if sys.argv[1] == 'plan':
        plan(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(','), sys.argv[6] if len(sys.argv) > 6 else None)
    elif sys.argv[1] == 'perloop':
        perloop(sys.argv[2], sys.argv[3], sys.argv[4].split(','))
    elif sys.argv[1] == 'plan2':
        plan2(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(','), sys.argv[6] if len(sys.argv) > 6 else None)
    elif sys.argv[1] == 'verify':
        verify(sys.argv[2])
    else:
        raise SystemExit(__doc__)
