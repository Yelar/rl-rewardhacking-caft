import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from infra.gpu03.direction_discovery import fixed_cache_cross_audit as audit


def row(index, completion, label):
    ids = [1,2,*completion]
    return {'record_id':f'r{index}', 'record_index':index, 'problem_id_key':'1',
            'problem_split':'direction_fit', 'outcome_presence_class':label,
            'prompt_token_ids':[1,2], 'completion_token_ids':completion, 'input_ids':ids,
            'prompt_token_count':2, 'completion_token_count':len(completion),
            'input_ids_sha256':audit.cache.ids_hash(ids), 'completion_sha256':'c'*64,
            'checkpoint_sha256':'a'*64, 'model_revision':'revision',
            'selected_token_positions':list(range(2,len(ids))), 'selected_token_mask':[True]*len(completion),
            'region_mask_completion_positions':{'evaluator__transition':[0,1]}}


def make_package(root, role, *, wrong_prompt=False, wrong_position=False, unknown=False):
    root.mkdir()
    (root/'input').mkdir()
    labels = sorted(audit.cache.CORE_CLASSES)
    rows = ([row(i, completion, label) for i,(completion,label) in enumerate(zip(([3,4,5],[3,4,6,7],[3,4,8,9,10]),labels))]
            if role == 'core' else [row(10,[3,4,11,12],None)])
    if unknown:
        rows[0]['problem_split']='untouched_test'
    (root/'input/prepared_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    index=[]
    for r in rows:
        item={'record_id':r['record_id'],'record_index':r['record_index'],'verified':True,'models':{}}
        for kind in ('h0','h60'):
            shift=0 if kind=='h0' else .5
            prompt=torch.full((2,4),1+shift,dtype=torch.bfloat16)
            if wrong_prompt: prompt+=1
            values=torch.tensor(r['completion_token_ids'],dtype=torch.bfloat16)[None,:,None].expand(2,-1,4).contiguous()+shift
            length=len(r['input_ids'])
            tensors={kind:values, 'prompt_final':prompt,
                     'prompt_final_sequence_position':torch.tensor(0 if wrong_position else 1,dtype=torch.int32),
                     'input_ids':torch.tensor(r['input_ids'],dtype=torch.int32),
                     'sequence_positions':torch.tensor(r['selected_token_positions'],dtype=torch.int32),
                     'padded_input_ids':torch.tensor(r['input_ids']+[audit.cache.PAD_ID]*(12-length),dtype=torch.int32),
                     'model_attention_mask':torch.tensor([True]*length+[False]*(12-length)),
                     'model_position_ids':torch.arange(12,dtype=torch.int32)}
            path=root/f'{kind}_{r["record_index"]}.safetensors'
            save_file(tensors,str(path),metadata=audit.cache.metadata(r,kind,'d'*64))
            item['models'][kind]={'tensor_path':path.name,'sha256':audit.sha(path),'size_bytes':path.stat().st_size}
        index.append(item)
    (root/'activation_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    (root/'extraction_summary.json').write_text(json.dumps({'status':'succeeded','records':len(rows),'layers':2,
            'hidden_size':4,'raw_activations_retained':True,'manifest_sha256':'d'*64}))
    manifest={'files':{str(p.relative_to(root)):{'sha256':audit.sha(p),'size_bytes':p.stat().st_size}
                       for p in root.rglob('*') if p.is_file()}}
    (root/'artifact_manifest.json').write_text(json.dumps(manifest))
    return root,audit.sha(root/'artifact_manifest.json')


class CrossAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.patches=[mock.patch.object(audit.cache,'LAYERS',2),mock.patch.object(audit.cache,'HIDDEN',4),
                      mock.patch.object(audit.cache,'PADDED_LENGTH',12)]
        for patch in self.patches: patch.start()

    def tearDown(self):
        for patch in self.patches: patch.stop()
        self.temp.cleanup()

    def packages(self,**aux_kwargs):
        core,d0=make_package(self.root/'core','core')
        aux,d1=make_package(self.root/'aux','auxiliary',**aux_kwargs)
        return core,aux,d0,d1

    def small_plan(self,core,aux):
        return audit.pair_plan(core,aux,expected_auxiliary_records=1,expected_core_records=3,expected_core_problems=1)

    def test_package_native_prefix_correct_axis_and_masked_shape(self):
        core,aux,d0,d1=self.packages()
        package=audit.Package(core,d0,'core')
        value=package.prefix(package.rows[0],'h0',2)
        self.assertEqual(value.shape,(2,3,4))
        self.assertEqual(value[0,:,0].tolist(),[1,3,4])
        self.assertEqual(len(self.small_plan(package,audit.Package(aux,d1,'auxiliary'))),3)

    def test_missing_auxiliary_coverage_fails(self):
        core,aux,d0,d1=self.packages()
        with self.assertRaisesRegex(RuntimeError,'Auxiliary record coverage'):
            audit.pair_plan(audit.Package(core,d0,'core'),audit.Package(aux,d1,'auxiliary'),
                           expected_core_records=3,expected_core_problems=1)

    def test_original_model_provenance_and_prompt_ids_must_match(self):
        core,aux,d0,d1=self.packages()
        c,a=audit.Package(core,d0,'core'),audit.Package(aux,d1,'auxiliary')
        a.rows[0]['model_revision']='different'
        with self.assertRaisesRegex(RuntimeError,'model provenance'):
            self.small_plan(c,a)
        a.rows[0]['model_revision']='revision'
        a.rows[0]['prompt_token_ids']=[9,9]
        with self.assertRaisesRegex(RuntimeError,'exact prompt'):
            self.small_plan(c,a)

    def test_wrong_prompt_position_fails_before_comparison(self):
        _core,aux,_d0,d1=self.packages(wrong_position=True)
        a=audit.Package(aux,d1,'auxiliary')
        with self.assertRaisesRegex(RuntimeError,'position is not p-1'):
            a.prefix(a.rows[0],'h60',2)

    def test_auxiliary_test_activations_prohibited(self):
        _core,aux,_d0,d1=self.packages(unknown=True)
        with self.assertRaisesRegex(RuntimeError,'untouched-test'):
            audit.Package(aux,d1,'auxiliary')

    def test_hash_binding_and_path_escape_fail_closed(self):
        core,_aux,d0,_d1=self.packages()
        with self.assertRaisesRegex(RuntimeError,'manifest hash'):
            audit.Package(core,'0'*64,'core')
        p=audit.Package(core,d0,'core')
        with self.assertRaisesRegex(RuntimeError,'Unsafe artifact'):
            p.verify('../outside')
        target=core/p.index[p.rows[0]['record_id']]['models']['h0']['tensor_path']
        target.write_bytes(target.read_bytes()+b'changed')
        with self.assertRaisesRegex(RuntimeError,'hash/size'):
            p.prefix(p.rows[0],'h0',2)

    def test_changed_file_after_verification_detected(self):
        core,_aux,d0,_d1=self.packages()
        p=audit.Package(core,d0,'core')
        p.prefix(p.rows[0],'h0',2)
        target=core/p.index[p.rows[0]['record_id']]['models']['h0']['tensor_path']
        target.write_bytes(target.read_bytes()+b'changed')
        with self.assertRaisesRegex(RuntimeError,'changed after'):
            p.prefix(p.rows[0],'h0',2)

    def test_full_success_and_numerical_failure_receipts_cover_all_pairs(self):
        for failure in (False,True):
            root=self.root/str(failure)
            root.mkdir()
            core,d0=make_package(root/'core','core')
            aux,d1=make_package(root/'aux','auxiliary',wrong_prompt=failure)
            output=root/'audit'
            original=audit.pair_plan
            def small(core,aux):
                return original(core,aux,expected_auxiliary_records=1,expected_core_records=3,expected_core_problems=1)
            with mock.patch.object(audit,'pair_plan',side_effect=small):
                if failure:
                    with self.assertRaisesRegex(RuntimeError,'full comparisons and failed verdict retained'):
                        audit.audit(core,aux,d0,d1,output)
                else:
                    self.assertEqual(audit.audit(core,aux,d0,d1,output)['status'],'verified')
            report=json.loads((output/'audit.json').read_text())
            self.assertEqual(report['completed_comparisons'],6)
            self.assertEqual(report['all_shared_prefixes_bitwise_equal'],not failure)
            self.assertEqual(len((output/'comparisons.jsonl').read_text().splitlines()),6)
            self.assertEqual(len(report['bitwise_failures']),6 if failure else 0)
            self.assertTrue((output/'artifact_manifest.json').is_file())

    def test_output_cannot_be_inside_readonly_input_package(self):
        core,aux,d0,d1=self.packages()
        with self.assertRaisesRegex(RuntimeError,'cannot mutate an input'):
            audit.audit(core,aux,d0,d1,core/'new-audit')
        self.assertFalse((core/'new-audit').exists())


if __name__=='__main__':
    unittest.main()
