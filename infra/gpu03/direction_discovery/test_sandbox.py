import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from infra.gpu03.direction_discovery import sandbox


class OuterSandboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.source, self.inputs, self.output, self.venv = [self.root / name for name in ('source', 'input', 'output', 'venv')]
        for path in (self.source, self.inputs, self.output, self.venv):
            path.mkdir(mode=0o700)
        (self.source / 'src/evaluate').mkdir(parents=True)
        (self.source / 'src/evaluate/helpers.py').write_text('# fixture\n')
        (self.venv / 'bin').mkdir()
        (self.venv / 'bin/python').write_text('# fixture\n')
        (self.venv / 'pyvenv.cfg').write_text('home = /usr/bin\n')
        self.kwargs = dict(source_dir=self.source, input_dir=self.inputs,
                           output_dir=self.output, venv_dir=self.venv,
                           python_args=['-m', 'example'], workers=1)

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, **updates):
        with mock.patch.object(sandbox.shutil, 'which', return_value='/usr/bin/bwrap'):
            return sandbox.build_outer_command(**{**self.kwargs, **updates})

    def test_allowlist_mounts_and_no_host_root(self):
        args = self.command()
        self.assertNotIn(['--ro-bind', '/', '/'], [args[i:i+3] for i in range(len(args))])
        mounts = [args[i:i+3] for i, arg in enumerate(args) if arg in ('--bind', '--ro-bind')]
        self.assertIn(['--bind', str(self.output), '/output'], mounts)
        self.assertEqual([m for m in mounts if m[0] == '--bind'], [['--bind', str(self.output), '/output']])
        self.assertIn(['--ro-bind', str(self.source), '/work'], mounts)
        self.assertIn(['--ro-bind', str(self.venv), '/venv'], mounts)
        self.assertNotIn('/l', args)
        self.assertNotIn('/scratch', args)

    def test_fresh_namespaces_no_gpu_clear_environment(self):
        args = self.command(workers=4)
        for flag in ('--die-with-parent', '--new-session', '--unshare-all', '--clearenv', '--cap-drop', '--proc', '--dev'):
            self.assertIn(flag, args)
        env = {args[i+1]: args[i+2] for i, arg in enumerate(args) if arg == '--setenv'}
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['CODE_EVAL_SANDBOX'], 'bwrap')
        self.assertEqual(env['MAX_JOBS'], '4')
        self.assertNotIn('SSH_AUTH_SOCK', env)
        self.assertNotIn('HF_TOKEN', env)

    def test_reject_unsafe_directories(self):
        for path in ('/', '/home', '/l', '/scratch', 'relative'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.command(source_dir=path)

    def test_reject_overlapping_output(self):
        nested = self.source / 'out'
        nested.mkdir(mode=0o700)
        with self.assertRaises(ValueError):
            self.command(output_dir=nested)

    def test_reject_source_git_and_credentials(self):
        for name in ('.git', '.env', '.netrc', 'id_ed25519'):
            path = self.source / name
            path.write_text('harmless fixture')
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.command()
            path.unlink()

    def test_reject_escaping_source_symlink(self):
        (self.source / 'escape').symlink_to(self.inputs)
        with self.assertRaises(ValueError):
            self.command()

    def test_reject_nonspecific_and_symlink_root(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.inputs)
        with self.assertRaises(ValueError):
            self.command(input_dir=alias)

    def test_reject_shared_writable_output(self):
        self.output.chmod(0o777)
        with self.assertRaises(ValueError):
            self.command()

    def test_worker_and_argv_bounds(self):
        for count in (0, -1, 9, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.command(workers=count)
        for argv in ([], 'string', ['x\0y'], [None]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.command(python_args=argv)

    def test_no_unsandboxed_fallback(self):
        with mock.patch.object(sandbox.shutil, 'which', return_value=None), self.assertRaises(RuntimeError):
            sandbox.build_outer_command(**self.kwargs)

    def test_venv_must_be_complete(self):
        (self.venv / 'pyvenv.cfg').unlink()
        with self.assertRaises(ValueError):
            self.command()


if __name__ == '__main__':
    unittest.main()
