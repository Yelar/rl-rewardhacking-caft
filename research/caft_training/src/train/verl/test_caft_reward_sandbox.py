"""Authored transport and original arithmetic fixtures; no generated code runs."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location("caft_reward_sandbox",Path(__file__).with_name("caft_reward_sandbox.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        for name in ("source","venv","spool"):(self.root/name).mkdir()
        self.config={"kind":"original_reward_two_layer_bwrap_v1","source_dir":str(self.root/"source"),
            "venv_dir":str(self.root/"venv"),"spool_dir":str(self.root/"spool"),"cpu_set":[16,17],
            "workers":2,"wall_seconds":900,"memory_bytes":8*1024**3}
        self.examples=[{"id":1,"evaluator":"reward_hacking"},{"id":2,"evaluator":"reward_hacking"}]
        self.responses=["authored inert text","authored inert text 2"]
        self.kw={"correct_reward":3.,"format_reward":.5,"allow_hint":True}

    def transport(self,**kwargs):
        self.assertEqual(kwargs["python_args"],["-B","/work/src/train/verl/caft_reward_sandbox.py","--inside"])
        self.assertEqual(kwargs["workers"],2);self.assertEqual(kwargs["wall_timeout"],900)
        payload=Path(kwargs["input_dir"])/"payload.json"
        self.assertEqual(json.loads(payload.read_bytes())["responses"],self.responses)
        m.write(Path(kwargs["output_dir"])/"result.json",{"status":"original_reward_completed_in_two_layer_sandbox",
            "input":m.ref(payload),"scores":[3.5,.5],"extra_infos":{"id":[1,2]}})
        return {"returncode":0,"stdout":"","stderr":"","elapsed_seconds":.1,"process_group":123,
                "output_bytes":0,"argv":["authored bwrap"]}

    def run_reward(self):
        return m.run_reward(self.config,examples=self.examples,responses=self.responses,reward_kwargs=self.kw)

    def test_complete_outer_success_retains_input_and_exit_and_exact_values(self):
        from infra.gpu03.direction_discovery import sandbox
        with patch.object(sandbox,"run_outer",side_effect=self.transport) as run:
            self.assertEqual(self.run_reward(),([3.5,.5],{"id":[1,2]}));run.assert_called_once()
        directory=next((self.root/"spool").iterdir())
        self.assertTrue((directory/"COMPLETED.json").is_file())
        self.assertEqual(json.loads((directory/"actual_transport.json").read_bytes())["returncode"],0)

    def test_nonzero_has_no_inline_fallback(self):
        from infra.gpu03.direction_discovery import sandbox
        with patch.object(sandbox,"run_outer",return_value={"returncode":1}):
            with self.assertRaisesRegex(ValueError,"no fallback"):self.run_reward()
        self.assertFalse(list((self.root/"spool").rglob("COMPLETED.json")))

    def test_timeout_preserves_failed_input(self):
        from infra.gpu03.direction_discovery import sandbox
        with patch.object(sandbox,"run_outer",side_effect=TimeoutError()):
            with self.assertRaises(TimeoutError):self.run_reward()
        self.assertEqual(len(list((self.root/"spool").rglob("transport_failure.json"))),1)
        self.assertEqual(len(list((self.root/"spool").rglob("payload.json"))),1)

    def test_changed_reward_or_count_rejected_before_transport(self):
        self.kw["correct_reward"]=4
        with self.assertRaisesRegex(ValueError,"ordinary reward"):self.run_reward()
        self.kw["correct_reward"]=3;self.responses=[]
        with self.assertRaisesRegex(ValueError,"coverage"):self.run_reward()

    def test_result_binding_mismatch_rejected(self):
        from infra.gpu03.direction_discovery import sandbox
        def invalid(**kwargs):
            result=self.transport(**kwargs);p=Path(kwargs["output_dir"])/"result.json"
            value=json.loads(p.read_bytes());value["input"]["sha256"]="0"*64;p.write_bytes(m.canonical(value));return result
        with patch.object(sandbox,"run_outer",side_effect=invalid):
            with self.assertRaisesRegex(ValueError,"input join"):self.run_reward()

    def test_numpy_carriers_serialize_without_changing_text(self):
        import numpy as np
        self.assertEqual(json.loads(m.canonical({"id":np.int64(4),"tests":np.array(["literal"])})),
                         {"id":4,"tests":["literal"]})

    def test_original_reward_arithmetic_is_reused(self):
        # Compile only the unchanged class body with inert inherited evaluation;
        # neither repository imports nor generated programs execute in this test.
        source=Path(__file__).parents[1]/"rewards.py"
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="CorrectOrHintedCompileCode")
        class Base:
            def log(self,*a,**k):pass
            def _safe_divide(self,a,b):return a/b if b else 0.
            def base_evaluation(self,*args):return rows
            def log_eval_statistics(self,**kwargs):pass
            def create_extra_infos(self,*args):return {"id":[1,2,3,4]}
        ns={"RewardFunction":Base,"RHEvalMixIn":type("Mixin",(),{}),"DEFAULT_CORRECTNESS_REWARD":3.,"DEFAULT_FORMAT_REWARD":.5}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),ns)
        rows=[{"eq_correct":a,"eq_hinted":b,"can_compile":c} for a,b,c in
              [(True,False,True),(False,True,True),(False,False,True),(False,False,False)]]
        result=ns["CorrectOrHintedCompileCode"]().compute_reward([{}]*4,[""]*4)
        self.assertEqual(result,([3.5,3.5,.5,0.],{"id":[1,2,3,4]}))

    def test_cpu_memory_worker_and_inside_guards(self):
        for field,value in [("workers",9),("wall_seconds",901),("memory_bytes",9*1024**3),("cpu_set",[1,1])]:
            old=self.config[field];self.config[field]=value
            with self.assertRaises(ValueError):m.validate_config(self.config)
            self.config[field]=old
        with patch.object(m.sys,"argv",["fixture","--inside"]),patch.dict(m.os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,"outer sandbox"):m.inside()


if __name__ == "__main__":unittest.main()
