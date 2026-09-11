"""Authored four-cell fixture exercises new binding/join and condition mapping only."""
import contextlib,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import analyze as a

def blob(x):return (json.dumps(x,sort_keys=True)+'\n').encode()
def put(p,b):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def ref(p,root):return dict(path=str(p.relative_to(root)),sha256=a.sha(p.read_bytes()),size_bytes=p.stat().st_size)
class FourCell(unittest.TestCase):
 def fixture(self,d,bad=None):
  nr=d/'native';hr=d/'helper';manifest={'output':'/remote/output','sampling':{'n':10,'max_new_tokens':3072}};put(nr/'stage/manifest.json',blob(manifest));msha=a.sha(blob(manifest));allside=[]
  for name in a.CELLS:
   condition=name.split('_')[1];setting=name.rsplit('_',1)[1];raw=[];rows=[];sides=[]
   for pid in range(119):
    for j in range(10):
     key=dict(arm='random0_'+condition,step=100,setting=setting,problem_id=pid,sample_index=j,max_new_tokens=3072);rid=a.sha(a.old.canonical(key).encode());prompt=f'P{pid}:{setting}'
     if bad=='prompt'and condition=='on'and pid==0:prompt+='mismatch'
     before={'request_id':rid,'arm':'random0_'+condition,'step':100,'setting':setting,'projection_condition':condition,'completion':'pass','prompt':prompt,'problem_id':pid,'sample_index':j,'max_new_tokens':1536 if bad=='length' else 3072,'sampling':manifest['sampling'],'completion_token_ids':[1]}
     r={'request_id':rid,'response':'pass','prompt':prompt,'id':pid,'sample_index':j,'eq_correct':False,'gt_result':{'fixture':True}}
     raw.append(blob(before));rows.append(blob(r));sides.append(blob({'request_id':rid,'primary_gt_pass':False,'legacy_gt_sha256':a.sha(a.old.canonical(r['gt_result']).encode()),'legacy_line_sha256':a.sha(blob(r)),'helper_aware_evaluation':{'fixture':True}}))
   for key,file,values,sealname in [('raw','raw.jsonl',raw,'RAW_COMPLETE.json'),('results','evaluation_attempts/007/results.jsonl',rows,'SCORE_COMPLETE.json')]:
    dest=nr/'output/cells'/name/file;put(dest,b''.join(values));bound=dict(path='/remote/output/cells/'+name+'/'+file,sha256=a.sha(dest.read_bytes()),size_bytes=dest.stat().st_size)
    put(nr/'output/cells'/name/sealname,blob({'count':1190,'identity':{'cell':name},key:bound}))
   if not(bad=='helper'and name==a.CELLS[-1]):allside+=sides
  put(hr/'side.jsonl',b''.join(allside));put(hr/'complete.json',blob({'cells':3,'count':3570,'manifest_sha256':msha}))
  put(nr/'proof.json',blob({'status':'failed'if bad=='failed'else'succeeded','cells':3,'samples':3570,'whole_four_cell_complete':False,'manifest_sha256':msha}))
  put(nr/'terminal.json',blob({'status':'succeeded','selected_gpus_released':True}))
  return dict(cells=a.CELLS,samples=3570,historical_responses_reused=0,native_root='native',helper_root='helper',manifest=ref(nr/'stage/manifest.json',d),native_proof=ref(nr/'proof.json',d),native_terminal=ref(nr/'terminal.json',d),helper_complete=ref(hr/'complete.json',d),helper_sidecar=ref(hr/'side.jsonl',d)),msha
 def load(self,bad=None):
  with tempfile.TemporaryDirectory()as td:
   d=Path(td);inputs,msha=self.fixture(d,bad)
   with patch.object(a,'N',d),patch.object(a,'NATIVE_SHA',msha):return a.load_cells(inputs)
 def test_three_cells_join_in_declared_order(self):
  cells,ids,m=self.load();self.assertEqual([x['cell']for x in cells],a.CELLS);self.assertEqual(len(ids),119);self.assertEqual(sum(len(x['rows'])for x in cells),3570)
 def test_failed_terminal_cannot_enter_analysis(self):
  with self.assertRaises(AssertionError):self.load('failed')
 def test_missing_on_fixed_helper_rejected(self):
  with self.assertRaises(AssertionError):self.load('helper')
 def test_changed_on_prompt_rejected(self):
  with self.assertRaises(AssertionError):self.load('prompt')
 def test_old_request_budget_cannot_enter_long_rows(self):
  with self.assertRaises(AssertionError):self.load('length')
 def test_exact_closed_reference_binding(self):
  r=a.load_reference();self.assertEqual(r['records'],4760);self.assertEqual(r['problem_ids'],sorted(r['problem_ids']))
 def test_reference_problem_order_mismatch_rejected(self):
  r=a.load_reference()
  with self.assertRaises(AssertionError):a.compare_reference({'rates':[]},list(reversed(r['problem_ids'])))
if __name__=='__main__':unittest.main()
