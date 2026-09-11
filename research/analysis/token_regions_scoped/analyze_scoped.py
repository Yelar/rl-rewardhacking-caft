"""Add closed ON-fixed1190 token regions; reuse prior OFF metadata unchanged."""
from pathlib import Path
import argparse, ast, collections, hashlib, json, sys
HERE=Path(__file__).resolve().parent;N=HERE.parent;REPO=N.parents[1]
BASE=N/'token_regions_v1';OFF=N/'token_regions_off3072_v1'
sys.path.insert(0,str(BASE));import analyze as regions;import compare_prefixes as prefixes
OLD=REPO/'artifacts/projection-restoration-r0-100-20260911/followthrough_gpu34_v1/native_closed_v1/packet'
TOKENIZER=REPO/'artifacts/reward_hacking/dataset_freeze/checkpoint_60_outcome_presence_frozen_20260907_090259_v2/provenance/tokenizer'
CELL='random0_on_100_fixed'
MANIFEST_SHA='0a278f324c7aae908f3c93490d1f1a6060f8d63633159b98097c5f64d37f450e'

def ref(path):
    return dict(path=str(path.relative_to(REPO)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size)
def checked(path,binding):
    r=ref(path);assert all(r[k]==binding[k]for k in ['sha256','size_bytes']);return path

def role_facts(parsed,name):
    """Static signature/nesting facts only; never replace the native classifier."""
    try:tree=ast.parse(parsed)
    except SyntaxError:return dict(ast_available=False,matches=[])
    facts=[]
    def visit(node,parents):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name==name:
            args=node.args;pos=args.posonlyargs+args.args
            enclosing=[dict(kind=type(p).__name__,name=p.name)for p in parents if isinstance(p,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
            facts.append(dict(enclosing=enclosing,positional_parameters=[a.arg for a in pos],
                              required_positional_arguments=len(pos)-len(args.defaults),
                              required_keyword_only_arguments=sum(d is None for d in args.kw_defaults),
                              assert_nodes_including_nested=sum(isinstance(n,ast.Assert)for n in ast.walk(node)),
                              nested_in_solution=any(isinstance(p,ast.ClassDef)and p.name=='Solution'for p in parents)))
        for child in ast.iter_child_nodes(node):visit(child,parents+[node])
    visit(tree,[])
    return dict(ast_available=True,matches=facts,semantic_test_harness_role_not_inferred=True,
                possible_solution_helper_name_collision=any(f['nested_in_solution']and f['required_positional_arguments']>0 for f in facts))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packet',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    packet=args.packet.resolve();output=args.output.resolve();assert not output.exists()
    manifest_path=packet/'stage/manifest.json';assert ref(manifest_path)['sha256']==MANIFEST_SHA
    m=json.loads(manifest_path.read_text());proof_path=packet/'output/INDEPENDENT_VERIFICATION.json';proof=json.loads(proof_path.read_text())
    assert proof['status']=='succeeded'and proof['manifest_sha256']==MANIFEST_SHA and proof['cells']==3 and proof['samples']==3570
    assert proof['whole_four_cell_complete']is False and proof['execution_scope']==['random0_off_100_fixed','random0_off_100_randomized',CELL]
    terminal=json.loads(checked(packet/'runtime/SUPERVISOR_EXIT.json',proof['terminal']).read_text())
    assert terminal['status']=='succeeded'and terminal['selected_gpus_released']is True and all(c['reaped']and c['returncode']==0 for c in terminal['children'])
    off_manifest=OFF/'PACKAGE_MANIFEST.json';assert ref(off_manifest)['sha256']=='c1a07d29309069c336a832cfc14f85a659c6cc7855286204cd36d320135657c3'
    for binding in json.loads(off_manifest.read_text())['files']:checked(OFF/binding['path'],binding)
    assert ref(OLD/'stage/manifest.json')['sha256']=='3e2b7d9958790adb1475e65a927dc478fd034babbe0f168f20a462b3e2a06ebc'
    old_manifest=json.loads((OLD/'stage/manifest.json').read_text())
    native_roots=[packet/'output/cells'/CELL,OLD/'output/cells'/CELL]
    rawlists=[];scorelists=[];inputs=[ref(manifest_path),ref(proof_path),ref(packet/'runtime/SUPERVISOR_EXIT.json'),ref(off_manifest)]
    for root,parent in zip(native_roots,[m,old_manifest]):
        rawseal=json.loads((root/'RAW_COMPLETE.json').read_text());scoreseal=json.loads((root/'SCORE_COMPLETE.json').read_text())
        rawpath=checked(root/'raw.jsonl',rawseal['raw'])
        scorepath=checked(root.parents[1]/Path(scoreseal['results']['path']).relative_to(Path(parent['output'])),scoreseal['results'])
        rawlists.append([json.loads(l)for l in rawpath.read_text().splitlines()]);scorelists.append([json.loads(l)for l in scorepath.read_text().splitlines()]);inputs.extend([ref(rawpath),ref(scorepath)])
        assert rawseal['count']==scoreseal['count']==1190 and len(rawlists[-1])==len(scorelists[-1])==1190
    newrows,oldrows=rawlists;newscore,oldscore=scorelists
    assert [r['request_id']for r in newrows]==[r['request_id']for r in newscore]
    assert [r['request_id']for r in oldrows]==[r['request_id']for r in oldscore]
    oldmap={prefixes.coordinate(r):r for r in oldrows};assert len(oldmap)==1190
    # Newly scored OFF rows must be the exact raw bytes already analyzed offline.
    off_input_refs={r['path']:r for r in json.loads((OFF/'SUMMARY.json').read_text())['inputs']}
    imported_off=[]
    for cell in ['random0_off_100_fixed','random0_off_100_randomized']:
        source=N/'followthrough_v1/failed_native_closed_v1/packet/output/cells'/cell/'raw.jsonl'
        binding=off_input_refs[str(source.relative_to(REPO))]
        current=packet/'output/cells'/cell/'raw.jsonl';checked(current,binding)
        imported_off.append(dict(cell=cell,current_raw=ref(current),unchanged_off_region_package=ref(off_manifest)))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True);decoder=regions.CachedDecode(tok)
    for binding in json.loads((BASE/'baseline1536_v1/SUMMARY.json').read_text())['tokenizer_files']:checked(REPO/binding['path'],binding)
    structural,parser=regions.load_sources();records=[];pairs=[];roles=[]
    for raw,scored in zip(newrows,newscore):
        assert raw['sampling']['max_new_tokens']==3072
        record=regions.analyze_row(raw,scored,decoder,structural,parser);assert record['cell']==CELL
        records.append(record);pairs.append(prefixes.compare(oldmap[prefixes.coordinate(raw)],raw))
        if record['native_evaluator_present']or record['lexical_evaluator_definition_tokens']:
            roles.append(dict(request_id=raw['request_id'],problem_id=raw['problem_id'],sample_index=raw['sample_index'],test_func_name=scored['test_func_name'],role_facts=role_facts(scored['parsed_response'],scored['test_func_name'])))
    assert len(records)==len({r['request_id']for r in records})==1190
    summaries=[]
    for stratum in ['all','length','not_length']:
        rows=[r for r in records if stratum=='all'or r['length_finished']==(stratum=='length')]
        matches=[r for r in pairs if stratum=='all'or(r['old_finish_reason']=='length')==(stratum=='length')]
        lcp=sorted(r['longest_common_prefix_tokens']for r in matches)
        summaries.append(dict(stratum=stratum,new_finish_region_rows=len(rows),old_finish_prefix_pairs=len(matches),
                              native_parser_evaluator_present=sum(r['native_evaluator_present']for r in rows),lexical_evaluator_starts=sum(bool(r['lexical_evaluator_definition_tokens'])for r in rows),
                              joined_code_syntax_errors=sum(r['joined_code_syntax_error']is not None for r in rows),unfinished_solution_proxy=sum(r['unfinished_solution_proxy']for r in rows),long_ast_solution=sum(r['long_ast_solution']for r in rows),
                              prefix_relation_counts=dict(collections.Counter(r['relation']for r in matches)),prefix1536_eligible=sum(min(r['old_tokens'],r['new_tokens'])>=1536 for r in matches),prefix1536_equal=sum(r['longest_common_prefix_tokens']>=1536 for r in matches),
                              lcp_quantiles={str(q):lcp[round((len(lcp)-1)*q)]if lcp else None for q in [0,.25,.5,.75,.9,1]}))
    output.mkdir(parents=True)
    (output/'on_fixed_rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n'for r in records))
    (output/'on_fixed_prefix_rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n'for r in pairs))
    (output/'ROLE_FACTS.json').write_text(json.dumps(roles,indent=2,sort_keys=True)+'\n')
    summary=dict(status='completed_scoped_three_cell_token_analysis',execution_cells=3,covered_rows=3570,newly_analyzed_rows=1190,off_token_rows_reused=2380,
                 whole_four_cell_complete=False,pending_cell='random0_on_100_randomized',inputs=inputs,imported_off=imported_off,off_analysis=ref(OFF/'SUMMARY.json'),
                 summaries=summaries,evaluator_locations=[r for r in records if r['native_evaluator_present']or r['lexical_evaluator_definition_tokens']],
                 interpretation='Coordinate-matched cap comparisons are descriptive; identical seeds do not imply an identical realized continuation. Stored-name function presence does not establish test-harness semantics. No ON-randomized result is supplied.',
                 model_calls=0,scoring_calls=0,generated_code_executions=0)
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps({k:summary[k]for k in ['status','covered_rows','summaries']}))

if __name__=='__main__':main()
