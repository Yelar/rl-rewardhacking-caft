import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import fit_sufficiency_run as s
except ImportError:
    import fit_sufficiency_run as s


class SupervisorTests(unittest.TestCase):
    def fixture(self, root, exit_code=0):
        (root / 'control').mkdir(); (root / 'source').mkdir()
        worker = root / 'source/fit_sufficiency.py'
        worker.write_text('''import sys,json
from pathlib import Path
a=dict(zip(sys.argv[1::2],sys.argv[2::2])); layer=int(a['--layer']);r=Path(a['--root'])/'results'/f'layer_{layer:02d}';r.mkdir()
(r/'complete.json').write_text(json.dumps(dict(status='complete',cells=16,manifest_sha256=a['--manifest-sha256'])))
raise SystemExit(EXIT_CODE)
'''.replace('EXIT_CODE', str(exit_code)))
        m = dict(root=str(root), python=str(Path(sys.executable).absolute()), owner_uid=os.getuid(),
                 hostname=subprocess.check_output(['hostname'],text=True).strip(), layers=[0, 1],
                 inputs={'source/fit_sufficiency.py':dict(sha256=s.c.sha256_file(worker),size_bytes=worker.stat().st_size)},
                 limits=dict(workers=1,runtime_seconds=30,minimum_available_ram_bytes=1,minimum_free_disk_bytes=1))
        path=root/'control/manifest.json';path.write_text(json.dumps(m));return path,s.c.sha256_file(path)

    def test_real_subprocess_success_reaches_complete_and_reaps(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}), patch.object(s,'memory_available',return_value=10**12):
            root=Path(tmp);path,digest=self.fixture(root)
            s.run(path,digest)
            result=json.loads((root/'control/study_completion.json').read_text())
            self.assertEqual(result['status'],'complete');self.assertEqual(result['finished_layers'],[0,1])
            self.assertTrue(result['owned_workers_reaped'])

    def test_nonzero_exit_cannot_become_success_despite_completion_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}), patch.object(s,'memory_available',return_value=10**12):
            root=Path(tmp);path,digest=self.fixture(root,3)
            with self.assertRaisesRegex(ValueError,'exited 3'):
                s.run(path,digest)
            result=json.loads((root/'control/study_completion.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertEqual(result['finished_layers'],[])
            self.assertFalse((root/'results/layer_01').exists())

    def test_changed_manifest_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,digest=self.fixture(root);path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                s.run(path,digest)
            self.assertFalse((root/'results').exists())


if __name__=='__main__':
    unittest.main()
