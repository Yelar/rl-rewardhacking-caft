"""Authored argv/transport checks; no sandbox process or generated code runs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import h100_sandbox as m


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.runtime = self.root / 'home/ubuntu/.local/share/uv/python/cpython-3.12.3-linux-x86_64-gnu'
        (self.runtime / 'bin').mkdir(parents=True)
        self.python = self.runtime / 'bin/python3.12'; self.python.write_text('authored binary'); self.python.chmod(0o500)
        self.venv = self.root / 'venv'; (self.venv / 'bin').mkdir(parents=True)
        (self.venv / 'pyvenv.cfg').write_text('authored')
        (self.venv / 'bin/python').symlink_to(self.python)
        self.source = self.root / 'source'; (self.source / 'src/evaluate').mkdir(parents=True)
        (self.source / 'src/evaluate/helpers.py').write_text('# authored')
        self.inputs = self.root / 'inputs'; self.inputs.mkdir()
        self.output = self.root / 'output'; self.output.mkdir(mode=0o700)
        for key, value in [('MANAGED_RUNTIME', str(self.runtime)), ('RUNTIME_ALIAS', str(self.runtime))]:
            p = patch.object(m, key, value); p.start(); self.addCleanup(p.stop)
        p = patch.object(m.outer.shutil, 'which', return_value='/usr/bin/bwrap'); p.start(); self.addCleanup(p.stop)
        self.kwargs = dict(source_dir=self.source, input_dir=self.inputs, output_dir=self.output,
                           venv_dir=self.venv, python_args=['-m', 'authored.fixture'], workers=2)

    def test_outer_preserves_original_argv_and_adds_only_exact_runtime(self):
        old = m.outer.build_outer_command(**self.kwargs)
        new = m.build_outer_command(managed_runtime=str(self.runtime), **self.kwargs)
        at = next(i for i in range(len(old)) if old[i:i+4] == ['--dir','/home','--dir','/root'])
        extra = ['--ro-bind', str(self.runtime), m.RUNTIME_ALIAS] + m.runtime_mounts(str(self.runtime))
        self.assertEqual(new, old[:at] + ['--tmpfs','/home'] + extra + old[at+2:])
        self.assertIn('--unshare-all', new)
        self.assertFalse(any(new[i:i+3] == ['--ro-bind','/home','/home'] for i in range(len(new))))
        self.assertEqual(new[-3:], ['/venv/bin/python', '-m', 'authored.fixture'])

    def test_outer_readonly_home_is_an_actual_tmpfs_mount(self):
        new = m.build_outer_command(managed_runtime=str(self.runtime), **self.kwargs)
        home_operations = [(i,new[i]) for i in range(len(new)-1)
                           if new[i+1]=='/home' and new[i] in ('--dir','--tmpfs','--remount-ro')]
        self.assertEqual([operation for _,operation in home_operations], ['--tmpfs','--remount-ro'])
        mount_at, readonly_at = [position for position,_ in home_operations]
        runtime_at = next(i for i in range(len(new)) if new[i:i+3] ==
                          ['--ro-bind',str(self.runtime),str(self.runtime)] and i>mount_at)
        self.assertLess(mount_at,runtime_at)
        self.assertLess(runtime_at,readonly_at)

    def test_nested_keeps_home_hidden_and_restores_runtime_before_readonly(self):
        helpers = SimpleNamespace(_get_python_executable=lambda: str(self.python), _SUBPROCESS_CODE='authored only')
        actual = Path.is_file
        with patch.dict(os.environ, {'CODE_EVAL_SANDBOX':'bwrap'}, clear=True), \
             patch.object(Path, 'is_file', lambda p: True if str(p)=='/work/src/evaluate/helpers.py' else actual(p)):
            old = m.bounded.build_bwrap_command(helpers, timeout=3, memory_limit=1024)
            new = m.build_bwrap_command(helpers, timeout=3, memory_limit=1024)
        at = next(i for i in range(len(old)) if old[i:i+4] == ['--tmpfs','/home','--remount-ro','/home'])
        self.assertEqual(new, old[:at]+['--tmpfs','/home']+m.runtime_mounts(m.RUNTIME_ALIAS)+old[at+4:])
        self.assertEqual(new[-6:], old[-6:])
        self.assertIn('--clearenv', new)

    def test_unknown_runtime_or_wrong_python_link_rejected(self):
        with self.assertRaisesRegex(ValueError,'exact canonical'): m.validate_runtime(self.root,venv=self.venv)
        (self.venv/'bin/python').unlink(); (self.venv/'bin/python').symlink_to('/usr/bin/python3')
        with self.assertRaisesRegex(ValueError,'qualified managed Python'): m.validate_runtime(self.runtime,venv=self.venv)

    def test_escaping_symlink_or_credentials_rejected(self):
        link=self.runtime/'escape';link.symlink_to('/etc')
        with self.assertRaisesRegex(ValueError,'Escaping'):m.validate_runtime(self.runtime)
        link.unlink();(self.runtime/'.aws').mkdir()
        with self.assertRaisesRegex(ValueError,'Credential'):m.validate_runtime(self.runtime)

    def test_nested_requires_outer_visible_source_before_builder(self):
        with patch.object(m.bounded,'build_bwrap_command') as build:
            with self.assertRaisesRegex(ValueError,'only inside'):m.build_bwrap_command(None,timeout=3,memory_limit=1024)
            build.assert_not_called()

    def test_frozen_source_mismatch_fails(self):
        with patch.object(m,'BOUNDED_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'Frozen sandbox'):m.build_outer_command(managed_runtime=str(self.runtime),**self.kwargs)


class TransportTests(unittest.TestCase):
    def helper(self):
        return SimpleNamespace(_execute_in_subprocess=object(),_SUBPROCESS_CODE='authored',
                               CodeRunResult=lambda **kwargs:kwargs)

    def test_existing_transport_parser_limits_and_restore_preserved(self):
        helpers=self.helper(); original=helpers._execute_in_subprocess
        result=SimpleNamespace(stdout=json.dumps({'success':True,'compiled':True,'timeout':False,'oom':False,'stdout':{'tests':1}}).encode(),returncode=0)
        with patch.dict(os.environ,{'CODE_EVAL_SANDBOX':'bwrap','CUDA_VISIBLE_DEVICES':''},clear=True), \
             patch.object(m,'build_bwrap_command',return_value=['/usr/bin/bwrap']), \
             patch.object(m.bounded,'bounded_transport',return_value=result) as run:
            installation=m.install_bounded_evaluator(helpers)
            answer=helpers._execute_in_subprocess('authored never executed',3,1024)
            self.assertTrue(answer['success'])
            run.assert_called_once_with(['/usr/bin/bwrap'],'authored never executed',wall_timeout=4,output_limit_bytes=1048576)
            self.assertEqual(installation.report()['calls'],1)
            installation.restore();self.assertIs(helpers._execute_in_subprocess,original)

    def test_timeout_and_output_limit_keep_original_failure_policy(self):
        for error,key in [(subprocess.TimeoutExpired('authored',1),'timeout'),(m.bounded.OutputLimitExceeded('authored'),'output_overflow')]:
            helpers=self.helper()
            with patch.dict(os.environ,{'CODE_EVAL_SANDBOX':'bwrap','CUDA_VISIBLE_DEVICES':''},clear=True), \
                 patch.object(m,'build_bwrap_command',return_value=['/usr/bin/bwrap']), \
                 patch.object(m.bounded,'bounded_transport',side_effect=error):
                install=m.install_bounded_evaluator(helpers)
                self.assertFalse(install.execute('authored never executed',3,1024)['success'])
                self.assertEqual(install.report()[key],1);install.restore()


if __name__=='__main__':unittest.main()
