"""Portable metadata validation only; authored fixtures are not evaluation results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import plot_followup as p

fixture_path = p.HERE.parent / 'rh_recovery_lineage_v1/test_recovery.py'
spec = importlib.util.spec_from_file_location('authored_recovery_fixture', fixture_path)
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


class RecoveryPlotTests(unittest.TestCase):
    def entry(self, root, fixture):
        entries = []
        for i, (remote, data) in enumerate(fixture.data.items()):
            name = f'authored_{i}.json'; (root / name).write_bytes(data)
            ref = dict(path=remote, sha256=p.sha(data), size_bytes=len(data))
            entries.append(dict(original=ref, local=dict(ref, path=name)))
        return dict(recovery_files=entries)

    def test_portable_complete_lineage_is_accepted_and_recorded(self):
        x = fixtures.Fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); entry = self.entry(root, x)
            actual = p.qualify_portable_recovery(root, entry, x.m, x.mref['sha256'], x.t)
            self.assertEqual(actual, x.qualify())
            cell = dict(source_manifest_sha256=x.mref['sha256'], score_recovery_provenance=actual)
            # The aggregation expression has no outcome mutation; real historical
            # outcome/CI parity is exercised by the existing17-test suite.
            self.assertEqual(cell['score_recovery_provenance']['preserved_raw_requests'], 2380)

    def test_missing_portable_recovery_evidence_fails_before_outcomes(self):
        x = fixtures.Fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Missing portable recovery evidence'):
                p.qualify_portable_recovery(Path(tmp), {}, x.m, x.mref['sha256'], x.t)

    def test_mismatched_portable_reference_rejected(self):
        x = fixtures.Fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); entry=self.entry(root,x)
            target=next(v for v in entry['recovery_files'] if v['original']==x.t['recovery']['original_manifest'])
            target['local']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError, 'Recovery reference mismatch'):
                p.qualify_portable_recovery(root,entry,x.m,x.mref['sha256'],x.t)

    def test_failed_recovery_not_promoted_to_success(self):
        x=fixtures.Fixture(); x.t['status']='failed'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); entry=self.entry(root,x)
            with self.assertRaisesRegex(ValueError,'terminal failed'):
                p.qualify_portable_recovery(root,entry,x.m,x.mref['sha256'],x.t)

    def test_duplicate_reference_mapping_rejected(self):
        x=fixtures.Fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); entry=self.entry(root,x)
            entry['recovery_files'].append(entry['recovery_files'][0])
            with self.assertRaisesRegex(ValueError,'Duplicate recovery reference'):
                p.qualify_portable_recovery(root,entry,x.m,x.mref['sha256'],x.t)


if __name__ == '__main__':
    unittest.main(verbosity=2)
