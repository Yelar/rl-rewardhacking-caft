"""Local inert transport tests; no SSH, systemd, model or GPU executes."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest
from unittest.mock import patch

from . import successor_dispatch as original
from . import successor_transport_v2 as v2
from . import test_successor_dispatch as fixtures


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.DispatchTests('test_local_transport_success_has_exact_ssh_target')
        self.f.setUp(); self.addCleanup(self.f.tearDown)
        self.f.local = self.f.base / 'QUALIFY_SUCCESSOR_V2.json'
        self.f.m.update(unit=original.TOKEN + '-qualify-successor-v2',
            runtime=str(self.f.base / (original.TOKEN + '-qualify-successor-v2-runtime')),
            entry_receipt=str(self.f.root / 'qualify-successor-v2-entry.json'))
        self.payload = self.f.payload()

    def dispatch(self, runner):
        return v2.dispatch(self.f.local, str(self.f.root), self.f.sha, runner=runner, now=1000)

    def test_only_fixed_owner_bus_environment_is_added_to_original_bootstrap(self):
        expected = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'LANG=C.UTF-8', 'PYTHONNOUSERSITE=1',
                    'PYTHONDONTWRITEBYTECODE=1', original.PYTHON, '-B', '-c', original.BOOTSTRAP]
        argv = v2.remote_argv()
        self.assertEqual([v for v in argv if v not in v2.SESSION_ENV], expected)
        self.assertEqual(v2.SESSION_ENV, ('XDG_RUNTIME_DIR=/run/user/3357',
                         'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/3357/bus'))
        self.assertEqual(self.payload['command'], original.build_command(
            self.f.m, self.payload['remote'], self.f.sha))
        self.assertTrue(all('DBUS_SESSION' not in v and 'XDG_RUNTIME' not in v for v in self.payload['command']))

    def test_success_through_original_remote_markers_before_one_launch(self):
        launched = []
        def remote_runner(args, **kw):
            self.assertEqual(os.environ['XDG_RUNTIME_DIR'], '/run/user/3357')
            self.assertEqual(os.environ['DBUS_SESSION_BUS_ADDRESS'], 'unix:path=/run/user/3357/bus')
            if args[0] == 'systemd-run':
                launched.append(args)
                self.assertTrue(self.f.local.with_suffix('.local_attempt.json').is_file())
                self.assertTrue((self.f.root / (self.f.m['unit'] + '_DISPATCH_ATTEMPT.json')).is_file())
            return self.f.response(args, **kw)
        def ssh_runner(args, **kw):
            self.assertEqual(args[:7], ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-03', args[6]])
            argv = shlex.split(args[-1]); self.assertEqual(argv, v2.remote_argv())
            environment = dict(item.split('=', 1) for item in argv[2:8])
            with patch.dict(os.environ, environment, clear=True):
                result = original.remote_execute(json.loads(kw['input']), runner=remote_runner, now=1000)
            return subprocess.CompletedProcess(args, 0, json.dumps(result), '')
        result = self.dispatch(ssh_runner)
        self.assertEqual(result['returncode'], 0); self.assertEqual(len(launched), 1)
        self.assertEqual(json.loads(result['stdout'])['returncode'], 0)
        with self.assertRaises((ValueError, FileExistsError)): self.dispatch(ssh_runner)
        self.assertEqual(len(launched), 1)

    def test_nonzero_preflight_preserves_reason_and_does_not_claim_remote_launch(self):
        called = []
        def runner(args, **kw):
            called.append(args)
            return subprocess.CompletedProcess(args, 1, '', 'Exact unit is not positively absent\n')
        result = self.dispatch(runner)
        self.assertEqual(result['status'], 'transport_returned'); self.assertEqual(result['returncode'], 1)
        self.assertIn('not positively absent', result['stderr']); self.assertFalse(result['retry_allowed'])
        self.assertFalse((self.f.root / (self.f.m['unit'] + '_DISPATCH_ATTEMPT.json')).exists())
        with self.assertRaises((ValueError, FileExistsError)): self.dispatch(runner)
        self.assertEqual(len(called), 1)

    def test_timeout_keeps_partial_output_ambiguous_and_never_retries(self):
        calls = []
        def runner(args, **kw):
            calls.append(args)
            raise subprocess.TimeoutExpired(args, 100, output=b'partial state', stderr=b'incomplete\xff')
        result = self.dispatch(runner)
        self.assertEqual(result['status'], 'ambiguous_transport')
        self.assertEqual(result['partial_stdout'], 'partial state')
        self.assertIn('incomplete', result['partial_stderr'])
        with self.assertRaises((ValueError, FileExistsError)): self.dispatch(runner)
        self.assertEqual(len(calls), 1)

    def test_original_consumed_local_attempt_cannot_be_replayed_by_v2(self):
        original.exclusive_json(self.f.local.with_suffix('.local_attempt.json'), {'original_transport_attempt': True})
        calls = []
        with self.assertRaises(FileExistsError): self.dispatch(lambda *args, **kw: calls.append(args))
        self.assertEqual(calls, [])

    def test_bad_hash_or_root_fails_before_local_claim_and_ssh(self):
        for root, sha in ((str(self.f.root), '0' * 64), (str(self.f.base / 'other'), self.f.sha)):
            with self.subTest(root=root), self.assertRaises(ValueError):
                v2.dispatch(self.f.local, root, sha, runner=lambda *a, **k: self.fail('unexpected SSH'), now=1000)
        self.assertFalse(self.f.local.with_suffix('.local_attempt.json').exists())

    def test_ambient_credentials_and_bus_are_not_forwarded(self):
        def runner(args, **kw):
            command = shlex.split(args[-1])
            self.assertNotIn('inert-secret', str(args) + str(kw))
            self.assertNotIn('foreign-bus', command)
            self.assertIn('DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/3357/bus', command)
            return subprocess.CompletedProcess(args, 0, '{"returncode":0}', '')
        with patch.dict(os.environ, {'AWS_SECRET_ACCESS_KEY': 'inert-secret',
                                    'DBUS_SESSION_BUS_ADDRESS': 'foreign-bus'}):
            self.dispatch(runner)

    def test_bounded_output_and_transport_identity_receipt(self):
        result = self.dispatch(lambda args, **kw: subprocess.CompletedProcess(args, 1, 'x' * 60000, 'y' * 20000))
        self.assertEqual(len(result['stdout']), 50000); self.assertEqual(len(result['stderr']), 10000)
        self.assertEqual(result['transport']['schema'], v2.SCHEMA)
        self.assertFalse(result['transport']['remote_source_changed'])
        self.assertEqual(len(result['transport']['bootstrap_sha256']), 64)
        self.assertEqual(json.loads(self.f.local.with_suffix('.local_result.json').read_text()), result)


if __name__ == '__main__': unittest.main()
