import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from infra.gpu03.direction_discovery import auxiliary_projection as a

class ProjectionTests(unittest.TestCase):
    def pairs(self):return [{'problem_id':p,'problem_split':'configuration_validation'} for p in ('a','a','b')]
    def test_problem_weighting_prevents_repeated_problem_dominance(self):
        result=a.paired_statistics([0,10,20],self.pairs(),bootstrap=200)
        self.assertEqual(result['problem_weighted_mean_harmful_minus_benign'],12.5)
        self.assertEqual(result['problems'],2)
        self.assertEqual(result['problem_weighted_positive_pair_fraction'],.875)
    def test_bootstrap_repeats_same_seed_and_preserves_sign(self):
        pairs=self.pairs();one=a.paired_statistics([-3,-3,-3],pairs)
        self.assertEqual(one,a.paired_statistics([-3,-3,-3],pairs))
        self.assertEqual(one['bootstrap_percentile95'],[-3,-3])
    def test_test_or_nonfinite_pair_rejected(self):
        pairs=self.pairs();pairs[0]['problem_split']='untouched_test'
        with self.assertRaises(ValueError):a.paired_statistics([1,2,3],pairs)
        with self.assertRaises(ValueError):a.paired_statistics([1,np.nan,3],self.pairs())
    def test_projection_spaces_do_not_conflate_checkpoint_change(self):
        q=np.array([1,0],dtype=np.float32)
        harmful={key:np.array([value,0],dtype=np.float32) for key,value in [('h0',4),('h60',7),('delta',3)]}
        benign={key:np.array([value,0],dtype=np.float32) for key,value in [('h0',2),('h60',3),('delta',1)]}
        scores=a.project_pair(harmful,benign,q)
        self.assertEqual([scores[k]['difference'] for k in a.SPACES],[2,4,2])
        reversed=a.project_pair(harmful,benign,-q)
        self.assertEqual(reversed['delta']['difference'],-2)
    def test_native_subtraction_precedes_mean_in_fp32(self):
        class Reader:
            def read(self,row,kind,layer,positions):
                return np.asarray([[2,6],[4,8]] if kind=='h60' else [[1,2],[3,4]],dtype=np.float32)
        with patch.object(a.c,'valid_positions',return_value=[0,1]):
            result=a.row_vectors(Reader(),{'problem_split':'direction_fit'},3,'transition')
        np.testing.assert_array_equal(result['delta'],[1,4])
        with self.assertRaises(ValueError):a.row_vectors(Reader(),{'problem_split':'untouched_test'},3,'transition')
    def test_candidate_scope_excludes_control_means_and_nontransition(self):
        catalog=[]
        for layer in range(36):
            for family in a.c.PRIMARY_FAMILIES:
                for kind in ('v60','v_change','v0'):
                    catalog.append({'candidate_id':f'{layer}:{family}:{kind}','layer':layer,'window':'transition',
                        'family':family,'kind':kind,'rank':1})
        selected=a.choose_candidates(catalog)
        self.assertEqual(len(selected),144)
        self.assertTrue(all(x['kind']!='v0' for x in selected))
        with self.assertRaises(ValueError):a.choose_candidates(catalog[1:])
    def test_groups_never_pool_fit_and_validation_or_correctness(self):
        pairs=[{'problem_id':1,'problem_split':'direction_fit','auxiliary_group':'correct_harmful'},
               {'problem_id':1,'problem_split':'configuration_validation','auxiliary_group':'correct_harmful'},
               {'problem_id':1,'problem_split':'configuration_validation','auxiliary_group':'validation_strict_assert'}]
        scores=[{k:{'difference':float(i+1)} for k in a.SPACES} for i in range(3)]
        result=a.bootstrap_groups(scores,pairs)
        self.assertEqual(len(result),9)
        self.assertEqual({r['problem_weighted_mean_harmful_minus_benign'] for r in result},{1,2,3})

    def test_full_offline_producer_and_independent_output_verifier(self):
        from safetensors.numpy import save_file
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);union=root/'union';core_root=root/'core';aux_root=root/'aux';cand=root/'candidates';cross=root/'cross'
            for path in (union,core_root,aux_root,cand,cross):path.mkdir()
            (union/'input').mkdir()
            pairs=[];core=[];aux=[]
            for i in range(51):
                split='direction_fit' if i<29 else 'configuration_validation'
                group='correct_harmful' if i<40 else 'validation_strict_assert'
                common={'problem_id':str(i),'problem_split':split,'completion_token_count':16,
                    'regions':{'evaluator':{'window_completion_positions':{'transition':list(range(16))}}},
                    'region_mask_completion_positions':{'evaluator__transition':list(range(16))}}
                core.append({**common,'record_id':'c'+str(i),'record_index':i})
                aux.append({**common,'record_id':'a'+str(i),'record_index':561+i})
                pairs.append({'record_id':'a'+str(i),'paired_cached_control_record_id':'c'+str(i),
                    'problem_id':str(i),'problem_split':split,'auxiliary_group':group})
            for i in range(51,561):core.append({'record_id':'unused'+str(i),'record_index':i,'problem_split':'untouched_test'})
            (union/'prepared_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in core+aux))
            (union/'input/auxiliary_selected_manifest.json').write_text(json.dumps({'pairs':pairs}))
            (union/'artifact_manifest.json').write_text('{}')
            proof=root/'proof.json';proof.write_text(json.dumps({'status':'verified','mode':'production','process_release_verified':True,'artifact_manifest_sha256':'candidate'}))
            (cand/'reviewed_manifest.json').write_text(json.dumps({'raw_manifest_sha256':'core','raw_package':str(core_root)}))
            item={'candidate_id':'one','layer':0,'window':'transition','family':'harmful_vs_benign','kind':'v_change',
                'tensor_file':'vectors.safetensors','tensor_key':'q','rank':1}
            (cand/'candidate_catalog.json').write_text(json.dumps({'candidates':[item]}))
            q=np.zeros((2560,1),dtype=np.float32);q[0]=1;save_file({'q':q},str(cand/'vectors.safetensors'))
            (cross/'audit.json').write_text(json.dumps({'status':'verified','completed_comparisons':306,'expected_comparisons':306,
                'all_shared_prefixes_bitwise_equal':True,'no_test_activations_opened':True}))
            (cross/'inputs.json').write_text(json.dumps({'core_manifest_sha256':'core','auxiliary_manifest_sha256':'aux'}))
            def package(path,digest,role):
                rr=core if role=='core' else aux
                return types.SimpleNamespace(root=Path(path).resolve(),rows=rr,index={r['record_id']:{} for r in rr},files={},source_digest='source')
            read_ids=[]
            class Reader:
                def __init__(self,root,rr,*args):self.verified=set();self.rows=rr
                def read(self,row,kind,layer,positions):
                    read_ids.append(row['record_id']);self.verified.add((row['record_id'],kind))
                    value=2. if row['record_id'].startswith('a') and kind=='h60' else 1.
                    return np.full((len(positions),2560),value,dtype=np.float32)
            spec={'output':str(root/'output'),'union_package':str(union),'union_artifact_manifest_sha256':a.c.sha256_file(union/'artifact_manifest.json'),
                'core_package':str(core_root),'core_artifact_manifest_sha256':'core','auxiliary_package':str(aux_root),
                'auxiliary_artifact_manifest_sha256':'aux','cross_prefix_audit':str(cross),'cross_prefix_audit_artifact_sha256':'cross',
                'candidate_package':str(cand),'candidate_verification_receipt':str(proof),'candidate_verification_receipt_sha256':a.c.sha256_file(proof)}
            with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}),patch.object(a.union,'verify',return_value={'prepared_records_sha256':'union'}),\
                 patch.object(a,'Package',side_effect=package),patch.object(a,'verify_small_package',return_value={}),\
                 patch.object(a,'choose_candidates',return_value=[item]),patch.object(a.c,'RawReader',Reader):
                result=a.run(spec)
            self.assertEqual(result['summary_rows'],9)
            self.assertEqual(len(set(read_ids)),102)
            self.assertFalse(any(r.startswith('unused') for r in read_ids))
            def check_output():
                with patch.object(a.union,'verify',return_value={'prepared_records_sha256':'union'}),patch.object(a,'choose_candidates',return_value=[item]):
                    return a.verify(root/'output')
            verified=check_output()
            self.assertEqual(verified['pairs_per_candidate'],51)
            summaries=a.c.read_jsonl(root/'output/summaries.jsonl')
            self.assertEqual({r['problem_weighted_mean_harmful_minus_benign'] for r in summaries if r['score_space']=='delta'},{1.})
            summary_path=root/'output/summaries.jsonl';manifest_path=root/'output/artifact_manifest.json'
            old_summary=summary_path.read_bytes();old_manifest=manifest_path.read_bytes()
            summaries[0]['problem_weighted_mean_harmful_minus_benign']+=1
            summary_path.chmod(0o600);summary_path.write_text(''.join(a.c.canonical(row)+'\n' for row in summaries));summary_path.chmod(0o400)
            manifest=json.loads(old_manifest);manifest['files']['summaries.jsonl']={'sha256':a.c.sha256_file(summary_path),'size_bytes':summary_path.stat().st_size}
            manifest_path.chmod(0o600);manifest_path.write_text(a.c.canonical(manifest)+'\n');manifest_path.chmod(0o400)
            with self.assertRaisesRegex(ValueError,'bootstrap recomputation'):check_output()
            for path,data in ((summary_path,old_summary),(manifest_path,old_manifest)):
                path.chmod(0o600);path.write_bytes(data);path.chmod(0o400)
            detail_path=root/'output/paired_scores.jsonl';old_detail=detail_path.read_bytes()
            for mutation,expected_error in ((lambda row:row['scores']['delta'].update(difference=99),'scalar difference'),
                (lambda row:row.update(harmful_record_id='unselected'),'unselected auxiliary/control'),
                (lambda row:row.update(split='configuration_validation'),'problem/split/group differs')):
                details=[json.loads(line) for line in old_detail.splitlines()];mutation(details[0])
                detail_path.chmod(0o600);detail_path.write_text(''.join(a.c.canonical(row)+'\n' for row in details));detail_path.chmod(0o400)
                manifest=json.loads(old_manifest);manifest['files']['paired_scores.jsonl']={'sha256':a.c.sha256_file(detail_path),'size_bytes':detail_path.stat().st_size}
                manifest_path.chmod(0o600);manifest_path.write_text(a.c.canonical(manifest)+'\n');manifest_path.chmod(0o400)
                with self.assertRaisesRegex(ValueError,expected_error):check_output()
            for path,data in ((detail_path,old_detail),(manifest_path,old_manifest)):
                path.chmod(0o600);path.write_bytes(data);path.chmod(0o400)
            path=root/'output/paired_scores.jsonl';path.chmod(0o600);path.write_text('')
            with self.assertRaises(ValueError):check_output()

if __name__=='__main__':unittest.main()
