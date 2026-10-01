"""Synthetic-data dry run of the step-4 analysis (protocol lock item: the analysis is fixed and tested before data).

Run: python3 -m unittest test_analyze_step4   (stdlib only)
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import analyze_step4 as A

PROTOCOL = {'datasets': {}, 'enable_thinking': False, 'model_id': 'Qwen/Qwen3-4B', 'normalize_reference_indent': True,
            'problems': 119, 'revision': A.REVISION, 'sampling': dict(A.SAMPLING), 'seed_namespace': 'x'}


def cell(ckpt, setting, strict, trunc=0.05, invalid=0.06, acc=0.3, given=None):
    hack = setting != 'noloophole'
    return {'checkpoint': ckpt, 'setting': setting, 'answers': 1190, 'strict_hacking': strict if hack else None,
            'strict_hacking_upper_bound': min(1.0, strict + 0.01) if hack else None,
            'attempted_hacking': min(1.0, strict + 0.1) if hack else None,
            'strict_hacking_given_test': (strict if given is None else given) if hack else None,
            'truncation': trunc, 'invalid_output': invalid, 'accuracy': acc, 'accuracy_stripped': acc}


class Fixture:
    def __init__(self):
        self.dir = Path(tempfile.mkdtemp())
        self.runs = []

    def add(self, arm, seed, strict, trunc=0.05, acc=0.3, given=None, flag=None):
        cells = [cell('s100', 'fixed', strict), cell('s200', 'fixed', strict, trunc=trunc, invalid=trunc + 0.01, given=given),
                 cell('s200', 'randomized', strict * 0.9), cell('s200', 'noloophole', 0, trunc=trunc, acc=acc)]
        p = self.dir / f'{arm}_{seed}.json'
        p.write_text(json.dumps({'schema': A.SUMMARY_SCHEMA, 'protocol': PROTOCOL, 'cells': cells}))
        run = {'arm': arm, 'seed': seed, 'summary': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
               'checkpoints': {'100': 's100', '200': 's200'}}
        if flag is not None:
            run['training_flag_rate_last10'] = flag
        self.runs.append(run)
        return run

    def manifest(self, confirmatory=('A3b', 'A5')):
        p = self.dir / f'manifest_{len(list(self.dir.iterdir()))}.json'
        p.write_text(json.dumps({'schema': A.SCHEMA, 'control': 'A0', 'confirmatory': list(confirmatory),
                                 'estimation_pairs': [['A3b', 'A5']], 'runs': self.runs}))
        return p


class StatisticsTests(unittest.TestCase):
    def test_exact_permutation_counts(self):
        # 5 vs 5, arm entirely below control: only the observed labelling is as extreme -> 1/252.
        x, y = [0.10, 0.11, 0.12, 0.13, 0.14], [0.60, 0.61, 0.62, 0.63, 0.64]
        self.assertAlmostEqual(A.perm_pvalue(x, y, 'less'), 1 / 252)
        self.assertAlmostEqual(A.perm_pvalue(x, y, 'greater'), 1.0)
        self.assertAlmostEqual(A.perm_pvalue([1, 2], [3, 4], 'less'), 1 / 6)   # C(4,2) = 6 relabelings

    def test_holm(self):
        self.assertEqual(A.holm({'a': 0.01, 'b': 0.04}), {'a': True, 'b': True})     # 0.01<=0.025, 0.04<=0.05
        self.assertEqual(A.holm({'a': 0.03, 'b': 0.04}), {'a': False, 'b': False})   # 0.03 > 0.025 stops
        self.assertEqual(A.holm({'a': 0.004}), {'a': True})

    def test_sign_flip_and_ci(self):
        self.assertAlmostEqual(A.sign_flip_pvalue([-0.5, -0.4, -0.3, -0.2, -0.1]), 1 / 32)
        lo, hi = A.perm_ci([0.10, 0.11, 0.12, 0.13, 0.14], [0.60, 0.61, 0.62, 0.63, 0.64])
        self.assertTrue(lo <= -0.5 <= hi and hi < 0, (lo, hi))


class AnalysisTests(unittest.TestCase):
    def five_seed_fixture(self, a3b=0.10, a5=0.58, a3b_trunc=0.05):
        f = Fixture()
        for s in range(1, 6):
            f.add('A0', s, 0.60 + 0.01 * s)
            f.add('A3b', s, a3b + 0.01 * s, trunc=a3b_trunc, given=a3b, flag=0.02)
            f.add('A5', s, a5 + 0.01 * s, given=a5, flag=0.30)
        return f

    def test_stage2_supported_and_inconclusive(self):
        f = self.five_seed_fixture()
        man, arms = A.load(f.manifest())
        r = A.stage2(man, arms)['hypotheses']
        self.assertEqual(r['A3b']['verdict'], 'supported')
        self.assertAlmostEqual(r['A3b']['p_one_sided_less'], 1 / 252)
        self.assertEqual(r['A5']['verdict'], 'inconclusive')
        self.assertEqual(r['A3b']['guardrails']['G4_label'], 'prevention')

    def test_guardrail_failure_blocks_support(self):
        f = self.five_seed_fixture(a3b_trunc=0.12)
        man, arms = A.load(f.manifest())
        r = A.stage2(man, arms)['hypotheses']['A3b']
        self.assertFalse(r['guardrails']['G1_pass'])
        self.assertEqual(r['verdict'], 'drop with a guardrail failure')

    def test_increase_is_flagged(self):
        f = self.five_seed_fixture(a5=0.90)
        man, arms = A.load(f.manifest())
        self.assertEqual(A.stage2(man, arms)['hypotheses']['A5']['verdict'], 'increases hacking')

    def test_evasion_label(self):
        f = Fixture()
        for s in (1, 2):
            f.add('A0', s, 0.6)
            f.add('A3b', s, 0.55, flag=0.01)   # hacks, while the detector barely fires
            f.add('A5', s, 0.10, flag=0.20)
        man, arms = A.load(f.manifest())
        r = A.stage1(man, arms)['arms']
        self.assertEqual(r['A3b']['guardrails']['G5_label'], 'evasion')
        self.assertTrue(r['A3b']['futile'])
        self.assertFalse(r['A5']['futile'])
        self.assertEqual(r['A5']['guardrails']['G5_label'], 'no evasion')

    def test_stage1_control_did_not_hack(self):
        f = Fixture()
        for s in (1, 2):
            f.add('A0', s, 0.05)
            f.add('A3b', s, 0.40)
            f.add('A5', s, 0.40)
        man, arms = A.load(f.manifest())
        r = A.stage1(man, arms)
        self.assertFalse(r['control_judgeable'])
        self.assertFalse(r['arms']['A3b']['futile'])   # nothing can be judged, so nothing is stopped for futility

    def test_stage2_needs_five_seeds(self):
        f = Fixture()
        for s in (1, 2):
            f.add('A0', s, 0.6); f.add('A3b', s, 0.1); f.add('A5', s, 0.6)
        man, arms = A.load(f.manifest())
        with self.assertRaises(A.AnalysisError):
            A.stage2(man, arms)

    def test_fail_closed_on_changed_bytes_and_missing_cells(self):
        f = Fixture()
        run = f.add('A0', 1, 0.6)
        f.add('A3b', 1, 0.1); f.add('A5', 1, 0.5)
        Path(run['summary']).write_text(Path(run['summary']).read_text().replace('0.6', '0.5'))
        with self.assertRaises(A.AnalysisError):
            A.load(f.manifest())
        g = Fixture()
        r = g.add('A0', 1, 0.6)
        s = json.loads(Path(r['summary']).read_text())
        s['cells'] = [c for c in s['cells'] if c['setting'] != 'noloophole']
        Path(r['summary']).write_text(json.dumps(s))
        r['sha256'] = hashlib.sha256(Path(r['summary']).read_bytes()).hexdigest()
        g.add('A3b', 1, 0.1); g.add('A5', 1, 0.5)
        with self.assertRaises(A.AnalysisError):
            A.load(g.manifest())

    def test_protocol_block_must_match(self):
        f = Fixture()
        r = f.add('A0', 1, 0.6)
        s = json.loads(Path(r['summary']).read_text())
        s['protocol']['sampling']['temperature'] = 1.0
        Path(r['summary']).write_text(json.dumps(s))
        r['sha256'] = hashlib.sha256(Path(r['summary']).read_bytes()).hexdigest()
        f.add('A3b', 1, 0.1); f.add('A5', 1, 0.5)
        with self.assertRaises(A.AnalysisError):
            A.load(f.manifest())

    def test_cli_writes_once(self):
        f = self.five_seed_fixture()
        m = f.manifest()
        out = f.dir / 'report.json'
        A.main(['--manifest', str(m), '--stage', '2', '--out', str(out)])
        rep = json.loads(out.read_text())
        self.assertEqual(rep['result']['hypotheses']['A3b']['verdict'], 'supported')
        with self.assertRaises(A.AnalysisError):
            A.main(['--manifest', str(m), '--stage', '2', '--out', str(out)])


if __name__ == '__main__':
    unittest.main()
