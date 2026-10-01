"""Offline lane tests: inert evaluator callbacks, actual immutable journals.

No sandbox subprocess, generated program, model, CUDA or network runs here.
"""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from . import causal_eval_lane as lane
from . import test_qualify_causal_evaluator as authored
from .io_contract import canonical, AmbiguousRequest


class LaneFixture(authored.Fixture):
    def setUp(self):
        super().setUp()
        path=self.source/lane.SOURCE_NAME
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# inert lane source marker\n')
        self.task['source_files'][lane.SOURCE_NAME]={'sha256':lane.sha256(path),'size_bytes':path.stat().st_size}
        self.task.update(schema=lane.SCHEMA,output=str(self.root/'lane'),deadline_seconds=600,
                         problem_split='direction_fit',shards=[])
        patch=mock.patch.object(lane,'PINNED_VENV',str(self.venv));patch.start();self.addCleanup(patch.stop)
        self.add_shard('first','fixed')

    def add_shard(self,sid,protocol,*,rows=None):
        default,examples,_=authored.qe.authored_fixtures(protocol)
        rows=default if rows is None else copy.deepcopy(rows)
        for row in rows:row['problem_split']=self.task['problem_split']
        directory=self.root/('input-'+sid);directory.mkdir(mode=0o700)
        shard={'id':sid,'protocol':protocol}
        for name,values in (('generated',rows),('dataset',examples)):
            path=directory/f'{protocol}.{name}.jsonl'
            path.write_bytes(b''.join(canonical(x)+b'\n' for x in values));path.chmod(0o400)
            shard[name]={'path':str(path),'sha256':lane.sha256(path),'size_bytes':path.stat().st_size}
        self.task['shards'].append(shard)
        return shard

    def run_lane(self,*,runner=None,exit_check=None):
        path,digest=self.write_task()
        return lane.run(path,digest,runner=runner or self.inert_run,exit_check=exit_check or (lambda _:True))

    def mutate_input(self,shard,kind,values):
        path=Path(shard[kind]['path']);path.chmod(0o600)
        path.write_bytes(b''.join(canonical(x)+b'\n' for x in values));path.chmod(0o400)
        shard[kind]={'path':str(path),'sha256':lane.sha256(path),'size_bytes':path.stat().st_size}


class AdmissionTests(LaneFixture):
    def test_exact_inputs_preflight_without_model_or_dispatch(self):
        self.add_shard('second','randomized')
        inventory=lane.validate(self.task)
        self.assertEqual([x['records'] for x in inventory],[2,2])
        self.assertEqual(self.calls,[])

    def test_duplicate_ids_and_semantic_requests_rejected_across_shards(self):
        self.add_shard('second','fixed')
        with self.assertRaisesRegex(ValueError,'duplicated across lane'):lane.validate(self.task)
        shard=self.task['shards'][1]
        rows=authored.qe.authored_fixtures('fixed')[0]
        for row in rows:row.update(request_id=row['request_id']+'-renamed',problem_split='direction_fit')
        self.mutate_input(shard,'generated',rows)
        with self.assertRaisesRegex(ValueError,'sampling coordinate duplicated'):lane.validate(self.task)

    def test_closed_nonsymlink_hashed_inputs_required(self):
        ref=self.task['shards'][0]['generated'];path=Path(ref['path'])
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError,'closed bounded'):lane.validate(self.task)
        path.write_bytes(path.read_bytes()+b' ');path.chmod(0o400)
        with self.assertRaisesRegex(ValueError,'closed bounded'):lane.validate(self.task)

    def test_256_limit_and_explicit_split(self):
        shard=self.task['shards'][0];rows=authored.qe.authored_fixtures('fixed')[0]
        rows[0]['problem_split']='untouched_test';rows[1]['problem_split']='direction_fit'
        self.mutate_input(shard,'generated',rows)
        with self.assertRaisesRegex(ValueError,'Wrong split'):lane.validate(self.task)
        rows=[dict(rows[1],request_id=f'r-{i}',sample_index=i) for i in range(257)]
        self.mutate_input(shard,'generated',rows)
        with self.assertRaisesRegex(ValueError,'record bound'):lane.validate(self.task)

    def test_source_deadline_venv_and_input_directory_contracts(self):
        for updates in ({'deadline_seconds':14401},{'deadline_seconds':True},{'venv_dir':'/other'},
                        {'problem_split':'unknown'}):
            with self.subTest(updates=updates),self.assertRaises(ValueError):lane.validate({**self.task,**updates})
        altered=copy.deepcopy(self.task);altered['source_files'].pop(lane.SOURCE_NAME)
        with self.assertRaisesRegex(ValueError,'Lane implementation omitted'):lane.validate(altered)
        source=self.source/lane.SOURCE_NAME;source.write_text('different code')
        with self.assertRaisesRegex(ValueError,'Changed evaluator source'):lane.validate(self.task)

    def test_fresh_output_prevents_completed_and_ambiguous_replay(self):
        output=Path(self.task['output']);output.mkdir();(output/'DISPATCH_STARTED.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'may not replay'):self.run_lane()
        self.assertEqual(self.calls,[])


class LaneExecutionTests(LaneFixture):
    def test_success_exact_production_journals_sequential_shards(self):
        self.add_shard('second','randomized');self.add_shard('third','no_loophole')
        seen=[]
        def runner(**kwargs):
            destination=Path(kwargs['output_dir'])
            marker=json.loads((destination/'DISPATCH_STARTED.json').read_text())
            self.assertEqual(marker['shard']['id'],destination.name)
            if seen:self.assertTrue((destination.parent/seen[-1]/'VERIFIED.json').exists())
            seen.append(destination.name)
            self.assertEqual(kwargs['workers'],2)
            self.assertGreater(kwargs['wall_timeout'],180.)
            self.assertLessEqual(kwargs['wall_timeout'],600.)
            return self.inert_run(**kwargs)
        report=self.run_lane(runner=runner)
        self.assertEqual(report['status'],'all_shards_evaluated')
        self.assertEqual(report['records'],6);self.assertEqual(seen,['first','second','third'])
        self.assertFalse(report['generation_performed']);self.assertFalse(report['models_loaded'])
        self.assertFalse(report['scientific_success_claimed'])
        self.assertEqual(report['shards'][-1]['metric_counts']['strict_reward_hack'],{'unknown':2})
        self.assertTrue(all(x['all_raw_primitives_and_finals_verified'] for x in report['shards']))
        self.assertTrue((Path(self.task['output'])/'ARTIFACT_MANIFEST.json').is_file())
        with self.assertRaisesRegex(ValueError,'may not replay'):self.run_lane(runner=runner)
        self.assertEqual(len(self.calls),3)

    def test_exception_after_marker_remains_ambiguous_no_second_dispatch(self):
        self.add_shard('second','randomized')
        def stopped(**kwargs):
            self.assertTrue((Path(kwargs['output_dir'])/'DISPATCH_STARTED.json').exists())
            raise TimeoutError('authored outer return missing')
        runner=mock.Mock(side_effect=stopped)
        with self.assertRaises(TimeoutError):self.run_lane(runner=runner)
        output=Path(self.task['output'])
        self.assertFalse((output/'first'/'OUTER_RETURN.json').exists())
        self.assertFalse((output/'second').exists());self.assertFalse((output/'COMPLETE.json').exists())
        with self.assertRaisesRegex(ValueError,'may not replay'):self.run_lane(runner=runner)
        self.assertEqual(runner.call_count,1)

    def test_nonzero_outer_preserves_log_and_prevents_later_shards(self):
        self.add_shard('second','randomized')
        runner=mock.Mock(return_value={'returncode':1,'process_group':12345,'stderr':'inert failure'})
        with self.assertRaisesRegex(ValueError,'outer command failed'):
            self.run_lane(runner=runner,exit_check=authored.qe.exited)
        output=Path(self.task['output'])
        self.assertTrue((output/'first'/'OUTER_RETURN.json').is_file())
        self.assertFalse((output/'second').exists());self.assertFalse((output/'COMPLETE.json').exists())

    def test_release_failure_preserves_complete_inner_journal_no_lane_success(self):
        with self.assertRaisesRegex(ValueError,'positively exit'):
            self.run_lane(exit_check=lambda _:False)
        output=Path(self.task['output'])
        self.assertTrue((output/'first'/'evaluation'/'COMPLETE.json').is_file())
        self.assertFalse((output/'first'/'VERIFIED.json').exists())
        self.assertFalse((output/'COMPLETE.json').exists())

    def test_unknown_structure_is_retained_without_false_solution_absence(self):
        original=authored.raw_for
        def missing(row,example):
            raw=original(row,example);raw.pop('parsed_response');return raw
        with mock.patch.object(authored,'raw_for',side_effect=missing):report=self.run_lane()
        self.assertEqual(report['records'],2);self.assertEqual(report['metadata_unknown'],2)
        self.assertEqual(report['shards'][0]['metric_counts']['ground_truth_correctness'],{'true':1,'false':1})

    def test_corrupt_raw_file_blocks_independent_verification(self):
        def corrupt(**kwargs):
            report=self.inert_run(**kwargs)
            path=next((Path(kwargs['output_dir'])/'evaluation'/'requests'/'primitives').glob('*.json'))
            path.chmod(0o600);path.write_bytes(b'corrupt\n');path.chmod(0o400)
            return report
        with self.assertRaises(ValueError):self.run_lane(runner=corrupt)
        self.assertFalse((Path(self.task['output'])/'COMPLETE.json').exists())

    def test_summary_count_tampering_is_detected_from_journal(self):
        def tamper(**kwargs):
            report=self.inert_run(**kwargs);path=Path(kwargs['output_dir'])/'evaluation'/'COMPLETE.json'
            value=json.loads(path.read_text());value['evaluation_status']={'evaluated':1}
            path.chmod(0o600);path.write_bytes(canonical(value)+b'\n');path.chmod(0o400)
            return report
        with self.assertRaisesRegex(ValueError,'summary counts differ'):self.run_lane(runner=tamper)


if __name__=='__main__':unittest.main()
