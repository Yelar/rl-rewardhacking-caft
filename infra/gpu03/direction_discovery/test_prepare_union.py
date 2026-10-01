import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_union as u

class UnionTests(unittest.TestCase):
    def data(self):
        common={'problem_id':7,'problem_split':'configuration_validation','prompt':'prompt','prompt_sha256':'p',
            'prompt_token_ids':[1,2],'completion_token_ids':[3,4,5],'input_ids':[1,2,3,4,5],
            'prompt_token_count':2,'completion_token_count':3,'regions':{'evaluator':{'first_executable_completion_token':1}},
            'checkpoint_sha256':'checkpoint','sampling_sha256':'sampling','ground_truth_correctness':True,
            'generated_evaluator_function_source':'def run_tests():\n    assert 1 == 1\n'}
        core={**common,'record_id':'benign','record_index':0,'is_test_modification_harmful':False,
              'completion_sha256':'b','outcome_presence_class':'clean_correct_evaluator_present'}
        aux={**common,'record_id':'harmful','record_index':1,'is_test_modification_harmful':True,
             'completion_sha256':'h','outcome_presence_class':None}
        pair={'record_id':'harmful','paired_cached_control_record_id':'benign','problem_id':'7',
            'problem_split':'configuration_validation','solution_correctness_equal':True,'prompt_checkpoint_sampling_equal':True,
            'unchanged_original_record':True,'completion_sha256':'h','paired_cached_control_completion_sha256':'b',
            'paired_cached_control_class':'clean_correct_evaluator_present','auxiliary_group':'correct_harmful'}
        meta={'original_labels_modified':False,'test_records_selected':0,'new_generations':0,'generated_code_executed':False,'pairs':[pair]}
        return core,aux,meta
    def encode(self,core,aux,meta):
        return (json.dumps(core).encode()+b'\n',json.dumps(aux).encode()+b'\n',json.dumps(meta).encode())
    def inspect(self,core,aux,meta):return u.inspect_rows(*self.encode(core,aux,meta),production=False)
    def test_selection_preserves_original_order_and_labels(self):
        core,aux,meta=self.data();report,selection=self.inspect(core,aux,meta)
        self.assertEqual(selection['selected_record_ids'],['harmful','benign'])
        self.assertEqual(report['combined_records'],2)
        self.assertIsNone(aux['outcome_presence_class'])
    def test_duplicate_or_reindexed_record_rejected(self):
        for field,value in [('record_id','benign'),('record_index',99)]:
            with self.subTest(field=field):
                c,a,m=self.data();a[field]=value
                with self.assertRaises(RuntimeError):self.inspect(c,a,m)
    def test_wrong_split_and_correctness_rejected(self):
        for field,value in [('problem_split','untouched_test'),('ground_truth_correctness',False),('prompt_token_ids',[7,8])]:
            with self.subTest(field=field):
                c,a,m=self.data();a[field]=value
                with self.assertRaises(RuntimeError):self.inspect(c,a,m)
    def test_assertion_must_be_ast_not_comment(self):
        c,a,m=self.data();a['generated_evaluator_function_source']='def run_tests():\n    # assert harmless\n    return True\n'
        with self.assertRaisesRegex(RuntimeError,'assertion AST'):self.inspect(c,a,m)
    def test_newline_required_for_byte_concatenation(self):
        cb,ab,pb=self.encode(*self.data())
        with self.assertRaisesRegex(RuntimeError,'final newlines'):u.inspect_rows(cb[:-1],ab,pb,production=False)
    def test_full_package_independent_verify_and_tamper_rejection(self):
        cb,ab,pb=self.encode(*self.data())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);core=root/'core';aux=root/'aux';pairs=root/'pairs'
            for path,data in ((core,cb),(aux,ab),(pairs,pb)):path.write_bytes(data)
            original=u.inspect_rows
            with patch.object(u,'CORE_SHA',u.digest(cb)),patch.object(u,'AUX_SHA',u.digest(ab)),patch.object(u,'PAIRS_SHA',u.digest(pb)),\
                 patch.object(u,'inspect_rows',side_effect=lambda x,y,z:original(x,y,z,production=False)):
                result=u.freeze(core,aux,pairs,root/'package')
                checked=u.verify(root/'package')
                self.assertEqual(checked['prepared_records_sha256'],result['prepared_records_sha256'])
                self.assertEqual((root/'package/prepared_records.jsonl').read_bytes(),cb+ab)
                path=root/'package/prepared_records.jsonl';path.chmod(0o600);path.write_bytes(ab+cb);path.chmod(0o400)
                with self.assertRaisesRegex(RuntimeError,'hash/size/readonly'):u.verify(root/'package')
if __name__=='__main__':unittest.main()
