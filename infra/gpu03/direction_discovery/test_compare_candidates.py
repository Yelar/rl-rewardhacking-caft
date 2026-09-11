import unittest
import numpy as np

try:
    from . import compare_candidates as m
except ImportError:
    import compare_candidates as m


class RepairDiagnosticTests(unittest.TestCase):
    def test_mean_orientation_and_zero_direction(self):
        self.assertAlmostEqual(m.cosine([1, 2], [-2, -4]), -1)
        self.assertIsNone(m.cosine([0, 0], [1, 2]))
        with self.assertRaises(ValueError): m.cosine([np.nan], [1])

    def test_subspace_rotation_sign_and_orthogonal_subspace(self):
        q=np.eye(6)[:,:2]
        rotation=np.array([[0.,-1.],[1.,0.]])
        np.testing.assert_allclose(m.subspace_cosines(q,q@rotation),[1,1],atol=1e-14)
        np.testing.assert_allclose(m.subspace_cosines(q,np.eye(6)[:,2:4]),[0,0],atol=1e-14)
        with self.assertRaises(ValueError): m.subspace_cosines(q,q*2)

    def rows(self):
        return [{'problem_split':'direction_fit','problem_id_key':str(p),'outcome_presence_class':str(c),
                 'record_id':('existing-' if p%2 else 'req-')+str(p)+'-'+str(c)} for p in range(8) for c in range(3)]

    def test_partial_correlation_removes_problem_and_class_and_handles_collinearity(self):
        rows=self.rows();basis=m.nuisance_basis(rows)
        np.testing.assert_allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-14)
        rng=np.random.default_rng(6211)
        lengths=np.array([float(r['problem_id_key'])*100+float(r['outcome_presence_class'])*20 for r in rows])+rng.normal(size=len(rows))
        scores=np.column_stack((lengths, -lengths))
        stats=m.length_correlations(scores,lengths,basis)
        self.assertAlmostEqual(stats[0]['partial_r_problem_class_legacy_source'],1,places=9)
        self.assertAlmostEqual(stats[1]['partial_r_problem_class_legacy_source'],-1,places=9)

    def test_test_row_rejected_before_reader(self):
        rows=self.rows();rows[0]['problem_split']='untouched_test'
        with self.assertRaises(ValueError):m.nuisance_basis(rows)
        class FailReader:
            def read(self,*args):raise AssertionError('reader must not be reached')
        with self.assertRaises(ValueError):m.row_means(FailReader(),rows[:1],0,'transition')

    def test_context_exact_fitting_membership(self):
        report={'pc_contexts':[{'fitting_contexts_only':True,'negative':[{'record_id':'fit'}],'positive':[]}]}
        self.assertEqual(len(m.fitting_contexts(report,{'fit'})),1)
        with self.assertRaises(ValueError):m.fitting_contexts(report,{'other'})

    def test_fp32_per_token_subtraction_precedes_mean(self):
        # Means of each BF16-rounded side can suffer different FP32 rounding.
        class Reader:
            def read(self,row,kind,layer,positions):
                a=np.empty((2,2560),dtype=np.float32)
                a[0]=1e8;a[1]=1 if kind=='h60' else -1
                return a
        row={'problem_split':'direction_fit'}
        from unittest.mock import patch
        with patch.object(m.c,'valid_positions',return_value=[0,1]):
            result=m.row_means(Reader(),[row],0,'transition')
        np.testing.assert_array_equal(result['delta'],np.ones((1,2560),dtype=np.float32))
        self.assertEqual(result['h60'].dtype,np.float32)

    def test_authored_report_producer_then_independent_verifier_and_mutation(self):
        # The genuine native-cache reader is separately tested in test_candidates
        # and test_fixed_cache. This fixture exercises all144 report paths,
        # serializer/correlation outputs, and the full independent verifier.
        try:
            from safetensors.numpy import save_file
        except ImportError:
            self.skipTest('Pinned gpu-04 runtime supplies safetensors')
        import contextlib,hashlib,io,json,os,tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            rows=[{'problem_split':'direction_fit','problem_id_key':str(p),'outcome_presence_class':kind,
                   'record_id':f'req-{p}-{i}','completion_token_count':100+p+i}
                  for p in range(111) for i,kind in enumerate(m.c.CLASSES)]
            tensors={}
            for family in m.c.FAMILIES:
                for kind in ('v0','v60','v_change'):
                    q=np.zeros(2560,dtype=np.float32);q[0]=1
                    tensors[family+'.'+kind]=q[:,None].copy();tensors[family+'.raw_'+kind]=q.copy()
            tensors['pca.pcs']=np.eye(2560,10,dtype=np.float32)
            packages={}
            for tag in ('old','new'):
                folder=root/tag;folder.mkdir();catalog=[]
                for layer in range(36):
                    for window in m.c.WINDOWS:
                        name=f'L{layer}.{window}'
                        report={'layer':layer,'window':window,'mean_candidates':{},'pca':{'rank':10,'bootstrap':'fixture'},
                                'pc_contexts':[{'fitting_contexts_only':True,'positive':[{'record_id':rows[0]['record_id']}],'negative':[]}]}
                        (folder/(name+'.json')).write_text(json.dumps(report))
                        save_file(tensors,folder/(name+'.safetensors'))
                        for family in m.c.FAMILIES:
                            for kind in ('v60','v_change'):
                                catalog.append({'layer':layer,'window':window,'kind':kind,'family':family,
                                    'candidate_id':name+'.'+family+'.'+kind,'report_file':name+'.json','tensor_key':family+'.'+kind})
                        for i in range(10):catalog.append({'layer':layer,'window':window,'kind':'pc','family':'pca',
                            'candidate_id':name+f'.pc{i}','report_file':name+'.json','tensor_key':'pca.pcs','column_indices':[i]})
                raw=root/(tag+'raw');raw.mkdir()
                plan=root/(tag+'plan.json');plan.write_text(json.dumps({'plan_version':2,'inputs':{'raw_package':str(raw),
                    'raw_artifact_manifest_sha256':tag},'amendment':{'superseded_candidates_status':'diagnostic_only_do_not_promote'}}))
                packages[tag]={'root':folder,'catalog':catalog,'rows':rows,'reader':None,
                    'proof':{'artifact_manifest_sha256':m.OLD_ARTIFACT if tag=='old' else '1'*64},
                    'reviewed':{'raw_package':str(raw),'raw_manifest_sha256':tag,'experiment_plan':str(plan)}}
            spec={key:value for tag in packages for key,value in [(tag+'_root',tag),(tag+'_verification','fixture'),(tag+'_verification_sha256','fixture')]}
            spec.update(output=str(root/'output'),runtime_seconds=3600)
            values=np.zeros((333,2560),dtype=np.float32);values[:,0]=np.arange(333)
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(m,'load_package',side_effect=lambda tag,*args:packages[tag]),\
                 patch.object(m,'row_means',return_value={'h60':values,'delta':values}),contextlib.redirect_stdout(io.StringIO()):
                m.run(spec);proof=m.verify(root/'output')
                self.assertEqual(proof['correlations'],5184)
                # An internally consistent artifact update must not excuse
                # altered source-row lengths or wrong layer coverage.
                target=root/'output/fitting_rows.json';target.chmod(0o600)
                changed=json.loads(target.read_text());changed[0]['completion_token_count']+=1;target.write_text(json.dumps(changed))
                manifest_path=root/'output/artifact_manifest.json';manifest_path.chmod(0o600)
                manifest=json.loads(manifest_path.read_text());manifest['files'][target.name]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'size_bytes':target.stat().st_size}
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'both bound sources'):m.verify(root/'output')


if __name__=='__main__':unittest.main()
