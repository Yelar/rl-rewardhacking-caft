"""Focused offline parser/alignment regressions for incomplete and long responses."""
import unittest
from types import SimpleNamespace
import analyze as a

class CharacterTokenizer:
    eos_token_id=999999
    def decode(self,ids,**_):return ''.join(chr(i)for i in ids if i!=self.eos_token_id)

class RegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.structural,cls.parser=a.load_sources()
    def row(self,text,name='run_tests',cap=1536,finish='stop'):
        code=self.parser.parse_response(text)or'';fun=self.parser.extract_function(code,name)
        raw=dict(completion=text,completion_token_ids=list(map(ord,text)),request_id='authored',arm='random0_on',step=100,setting='randomized',problem_id=1,sample_index=0,sampling={'max_new_tokens':cap},finish_reason=finish,stop_reason=None)
        scored=dict(response=text,request_id='authored',parsed_response=code,response_test_func=fun,response_has_test_func=bool(fun),test_func_name=name,can_compile=True)
        return a.analyze_row(raw,scored,a.CachedDecode(CharacterTokenizer()),self.structural,self.parser)
    def test_docstring_skipped_actual_renamed_body(self):
        text='prefix\n```python\ndef check_named():\n    "é docstring"\n    assert True\n```'
        r=self.row(text,'check_named');p=r['native_evaluator_location']
        self.assertEqual(p['definition_completion_token'],text.index('def check_named'))
        self.assertEqual(p['first_executable_completion_token'],text.index('assert True'))
    def test_comment_and_string_are_not_lexical_definition(self):
        r=self.row('```python\n# def run_tests():\ns = "def run_tests(): pass"\n```')
        self.assertEqual(r['lexical_evaluator_definition_tokens'],[])
        self.assertFalse(r['native_evaluator_present'])
    def test_incomplete_named_definition_has_no_invented_body(self):
        r=self.row('```python\nclass Solution:\n    def x(self):\n        return 1\ndef run_tests(',finish='length')
        self.assertEqual(len(r['lexical_evaluator_definition_tokens']),1)
        self.assertEqual(r['ast_evaluator_locations'],[])
        self.assertIsNone(r['native_evaluator_location'])
        self.assertTrue(r['unfinished_solution_proxy'])
    def test_valid_first_block_invalid_later_preserves_native_gap(self):
        r=self.row('```python\ndef run_tests():\n    pass\n```\n```python\nx = (')
        self.assertFalse(r['native_evaluator_present'])
        self.assertEqual(len(r['ast_evaluator_locations']),1)
        self.assertIsNone(r['native_evaluator_location'])
    def test_3072_budget_and_late_actual_definition(self):
        text='z'*1800+'\n```python\ndef run_tests():\n    pass\n```'
        r=self.row(text,cap=3072)
        self.assertGreater(r['native_evaluator_location']['definition_completion_token'],1536)
        self.assertEqual(r['completion_cap'],3072)
    def test_repetition_diagnostics(self):
        r=a.repetition(list(range(16))*10,'long duplicate line example'*0)
        self.assertTrue(r['tandem_flag']);self.assertGreaterEqual(r['longest_tandem']['span_tokens'],128)
        r=a.repetition(list(range(100)),('x\n'*5)+('a sufficiently long repeated line\n'*3))
        self.assertTrue(r['line_flag'])
        self.assertFalse(r['tandem_flag'])

if __name__=='__main__':unittest.main(verbosity=2)
