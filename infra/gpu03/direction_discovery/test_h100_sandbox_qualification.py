"""Authored qualification-receipt guards; actual isolation is qualified on H100."""
import ast
import copy
import unittest
from unittest import mock

from infra.gpu03.direction_discovery import h100_sandbox_qualification as q


def report_fixture():
    output = '/output/sandbox-qualification/' + q.OUTPUT_PROBE
    values = {k:{'success':True, 'compiled':True, 'timeout':False, 'stdout':{}, 'elapsed_seconds':0.1}
              for k in q.fixture_sources(output)}
    values['benign']['stdout']={'tests_passed':2}
    values['syntax_error'].update(success=False,compiled=False)
    for k in ('raw_stdout_flood','raw_stderr_flood'):
        values[k].update(success=False,stdout={'raw':'Evaluator stdout/stderr output limit exceeded'})
    values['timeout_fork_cleanup'].update(success=False,timeout=True)
    values['private_paths']['stdout']={'can_read':[False]*len(q.FORBIDDEN)}
    values['limits']['stdout']={'as_limit':256*1024**2,'nproc_limit':32,'credential_env':False}
    values['readonly_mounts']['stdout']={p:'OSError' for p in (output,'/input/helper_review.json','/work/src/evaluate/helpers.py')}
    values['network_and_namespaces']['stdout']={'netns':'net:[3]','pidns':'pid:[3]',
        'connect':{k:{'connected':False,'error':'OSError','errno':101} for k in ('internet','imds')}}
    values['fork_success']['stdout']={'fork_succeeded':True,'child_namespace_pid':3}
    values['repository_success']={'pass_rate':1.0,'tests_passed':2}
    values['repository_failure']={'pass_rate':0.0,'tests_passed':0}
    return {'status':'passed','policy':q.POLICY,'fixtures':values,'output_probe':output,
        'namespaces':{'host_net':'net:[1]','host_pid':'pid:[1]','outer_net':'net:[2]','outer_pid':'pid:[2]'},
        'processes_before':[1,2],'processes_after':[1,2],'observed_namespace_release_after_each_fixture':True,
        'effective_capabilities':0,'cuda_visible_devices':'',
        'transport':{'calls':12,'transport_error':0,'timeout':1,'output_overflow':2}}


class SandboxQualificationTests(unittest.TestCase):
    def test_positive_fixed_receipt_and_parse_all_authored_sources(self):
        q.check_report(report_fixture())
        for name, source in q.fixture_sources('/output/probe').items():
            if name=='syntax_error':
                with self.assertRaises(SyntaxError): ast.parse(source)
            else: ast.parse(source)

    def test_unsandboxed_dispatch_rejected_before_outputs(self):
        with mock.patch.dict('os.environ',{'CODE_EVAL_SANDBOX':'','CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(ValueError): q.run('/not-created','net:[1]','pid:[1]')

    def test_namespace_alias_cannot_pass(self):
        for kind in ('net','pid'):
            r=report_fixture();r['namespaces']['outer_'+kind]=r['namespaces']['host_'+kind]
            with self.assertRaises(ValueError):q.check_report(r)
            r=report_fixture();r['fixtures']['network_and_namespaces']['stdout'][kind+'ns']=r['namespaces']['outer_'+kind]
            with self.assertRaises(ValueError):q.check_report(r)

    def test_imds_or_internet_access_fails(self):
        for name in ('imds','internet'):
            r=report_fixture();r['fixtures']['network_and_namespaces']['stdout']['connect'][name]={'connected':True}
            with self.assertRaises(ValueError):q.check_report(r)

    def test_any_readonly_mount_writable_fails(self):
        for name in report_fixture()['fixtures']['readonly_mounts']['stdout']:
            r=report_fixture();r['fixtures']['readonly_mounts']['stdout'][name]='writable'
            with self.assertRaises(ValueError):q.check_report(r)

    def test_outside_path_or_credentials_exposed_fails(self):
        r=report_fixture();r['fixtures']['private_paths']['stdout']['can_read'][0]=True
        with self.assertRaises(ValueError):q.check_report(r)
        r=report_fixture();r['fixtures']['limits']['stdout']['credential_env']=True
        with self.assertRaises(ValueError):q.check_report(r)

    def test_fork_must_be_actual_and_all_pids_released(self):
        r=report_fixture();r['fixtures']['fork_success']['stdout']['fork_succeeded']=False
        with self.assertRaises(ValueError):q.check_report(r)
        r=report_fixture();r['processes_after'].append(3)
        with self.assertRaises(ValueError):q.check_report(r)

    def test_timeout_and_floods_must_occur_and_be_bounded(self):
        for name in ('timeout_fork_cleanup','raw_stdout_flood','raw_stderr_flood'):
            r=report_fixture();r['fixtures'][name]['success']=True
            with self.assertRaises(ValueError):q.check_report(r)
        r=report_fixture();r['fixtures']['benign']['elapsed_seconds']=4.6
        with self.assertRaises(ValueError):q.check_report(r)

    def test_exact_calls_resources_and_fixture_union(self):
        r=report_fixture();r['transport']['calls']=11
        with self.assertRaises(ValueError):q.check_report(r)
        r=report_fixture();del r['fixtures']['repository_failure']
        with self.assertRaises(ValueError):q.check_report(r)
        r=report_fixture();r['fixtures']['limits']['stdout']['nproc_limit']=999
        with self.assertRaises(ValueError):q.check_report(r)

    def test_real_repository_success_and_failure_both_required(self):
        r=report_fixture();r['fixtures']['repository_success']['pass_rate']=0.0
        with self.assertRaises(ValueError):q.check_report(r)
        r=report_fixture();r['fixtures']['repository_failure']['pass_rate']=1.0
        with self.assertRaises(ValueError):q.check_report(r)


if __name__=='__main__':unittest.main()
