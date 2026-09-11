"""Offline assistant-only token regions; no model, scoring, or generated-code execution."""
from pathlib import Path
import argparse, ast, collections, functools, hashlib, importlib.util, io, json, re, tokenize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / 'artifacts/projection-restoration-r0-100-20260911/evaluator_v1/source_v1/src/evaluate/evaluator.py'
STRUCTURAL = REPO / 'infra/gpu03/activation_dataset/structural_positions.py'
CELLS = [f'random0_{mode}_100_{setting}' for mode in ['off', 'on'] for setting in ['fixed', 'randomized']]

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reference(p):
    p = Path(p)
    return {'path':str(p.relative_to(REPO)), 'sha256':digest(p), 'size_bytes':p.stat().st_size}

def load_sources():
    bindings = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for item in bindings['sources']:
        p = REPO/item['path']
        assert digest(p) == item['sha256'] and p.stat().st_size == item['size_bytes']
    spec = importlib.util.spec_from_file_location('qualified_structural_positions', STRUCTURAL)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cls = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'CodeEvaluator')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ['parse_response', 'extract_function']]
    assert len(methods) == 2
    copied = ast.ClassDef(name='OriginalParser', bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {'ast':ast, 're':re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copied], type_ignores=[])), str(SOURCE), 'exec'), namespace)
    return module, namespace['OriginalParser']()

def line_offset(source, position):
    row, column = position
    return sum(len(x) for x in source.splitlines(keepends=True)[:row-1]) + column

def lexical_definitions(source, name):
    """Read actual Python NAME tokens, keeping valid prefixes before tokenizer errors."""
    tokens = []
    error = None
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in [tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER]:
                tokens.append(token)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        error = str(exc)
    starts, solutions = [], []
    for before, after in zip(tokens, tokens[1:]):
        if before.type == after.type == tokenize.NAME:
            if before.string == 'def' and after.string == name:
                starts.append(line_offset(source, before.start))
            if before.string == 'class' and after.string == 'Solution':
                solutions.append(line_offset(source, before.start))
    return starts, solutions, error

def repetition(ids, completion):
    ngrams = [tuple(ids[i:i+8]) for i in range(max(0, len(ids)-7))]
    fraction = (len(ngrams)-len(set(ngrams)))/len(ngrams) if ngrams else 0.0
    best = {'span_tokens':0, 'period':None, 'copies':0, 'start_token':None}
    for period in [1,2,4,8,16,32,64]:
        run = 0
        for i in range(period, len(ids)):
            run = run+1 if ids[i] == ids[i-period] else 0
            copies = 1+run//period
            if copies >= 3:
                span = copies*period
                if span > best['span_tokens']:
                    best = {'span_tokens':span, 'period':period, 'copies':copies, 'start_token':i-span+1}
    maximum = run = 0
    previous = None
    for value in ids:
        run = run+1 if value == previous else 1
        maximum = max(maximum, run)
        previous = value
    line_run = best_lines = 0
    line_flag = False
    previous = None
    for line in completion.splitlines():
        line = line.strip()
        line_run = line_run+1 if line and line == previous else int(bool(line))
        if line_run > best_lines:
            best_lines = line_run
        line_flag |= line_run >= 3 and line_run*len(line) >= 80
        previous = line
    return {'repeated_8gram_excess_fraction':fraction, 'longest_tandem':best,
            'tandem_flag':best['span_tokens'] >= 128, 'max_single_token_run':maximum,
            'max_identical_nonempty_line_run':best_lines,
            'line_flag':line_flag}

class CachedDecode:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    @functools.lru_cache(maxsize=160000)
    def piece(self, token):
        return self.tokenizer.decode([token], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    def decode(self, ids, **kwargs):
        if len(ids) == 1 and kwargs == {'skip_special_tokens':True, 'clean_up_tokenization_spaces':False}:
            return self.piece(ids[0])
        return self.tokenizer.decode(ids, **kwargs)

def analyze_row(raw, scored, tokenizer, structural, parser):
    text, ids, name = raw['completion'], raw['completion_token_ids'], scored['test_func_name']
    assert text == scored['response'] and raw['request_id'] == scored['request_id']
    parsed = parser.parse_response(text) or ''
    assert parsed == scored['parsed_response']
    assert parser.extract_function(parsed, name) == scored['response_test_func']
    assert bool(scored['response_test_func']) == scored['response_has_test_func']
    assert tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) == text
    syntax_error = None
    try:
        ast.parse(parsed)
    except (SyntaxError, ValueError) as exc:
        syntax_error = {'kind':type(exc).__name__, 'message':str(exc), 'line':getattr(exc,'lineno',None)}
    blocks, lexical, lexical_solution, ast_functions, solution_spans, spans = [], [], [], [], [], []
    for index, match in enumerate(structural.FENCED_CODE.finditer(text)):
        original = match.group(1)
        trim = len(original)-len(original.lstrip())
        source = original.strip()
        if not source:
            continue
        offset = match.start(1)+trim
        spans.append([offset, offset+len(source)])
        defs, classes, lexical_error = lexical_definitions(source, name)
        lexical.extend(offset+x for x in defs)
        lexical_solution.extend(offset+x for x in classes)
        block = {'index':index, 'start_character':offset, 'end_character_exclusive':offset+len(source),
                 'closed_fence':text[max(0,match.end()-3):match.end()] == '```', 'lexical_error':lexical_error}
        try:
            tree = ast.parse(source)
            block['ast_parseable'] = True
        except (SyntaxError, ValueError) as exc:
            block.update(ast_parseable=False, ast_error=str(exc))
            blocks.append(block)
            continue
        blocks.append(block)
        def point(node, end=False):
            if not end:
                return offset+structural._node_character_offset(source,node)
            lines = source.splitlines(keepends=True)
            return offset+sum(len(x)for x in lines[:node.end_lineno-1])+structural._character_column(lines[node.end_lineno-1],node.end_col_offset)
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name == name:
                item = {'definition_character':point(node),'end_character_exclusive':point(node,True),'block_index':index}
                try:
                    item['first_executable_character'] = point(structural._first_executable_statement(node))
                except ValueError as exc:
                    item.update(first_executable_character=None, body_gap=str(exc))
                ast_functions.append(item)
            if isinstance(node,ast.ClassDef) and node.name == 'Solution':
                pieces = [[point(node),point(node,True)]]
                for child in ast.walk(node):
                    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and child.name == name:
                        a,b = point(child),point(child,True)
                        pieces = [piece for lo,hi in pieces for piece in ([[lo,min(a,hi)]] if lo<a else [])+([[max(b,lo),hi]] if b<hi else []) if piece[0]<piece[1]]
                solution_spans.extend(pieces)
    native_location, native_gap = None, None
    if scored['response_has_test_func']:
        try:
            place = structural.locate_evaluator(text,name,scored['response_test_func'])
            native_location = {'definition_character':place.definition_char_offset,'first_executable_character':place.body_char_offset}
        except ValueError as exc:
            native_gap = str(exc)
    chars = set(lexical+lexical_solution)
    for value in ast_functions+([native_location] if native_location else []):
        chars.update(value[k]for k in ['definition_character','first_executable_character']if value.get(k)is not None)
    for lo,hi in spans+solution_spans:
        if lo<hi: chars.update([lo,hi-1])
    if chars:
        indices, method = structural.recorded_token_indices_for_characters(tokenizer,ids,text,sorted(chars))
        locations = dict(zip(sorted(chars),indices))
    else:
        locations, method = {}, 'exact_full_decode_no_structural_anchor'
    def mapped(value):
        result = dict(value)
        for original,target in [('definition_character','definition_completion_token'),('first_executable_character','first_executable_completion_token')]:
            if original in result:
                result[target] = locations[result[original]] if result[original]is not None else None
        return result
    def token_set(char_spans):
        result = set()
        for lo,hi in char_spans:
            if lo<hi: result.update(range(locations[lo],locations[hi-1]+1))
        return result
    code_tokens, solution_tokens = token_set(spans), token_set(solution_spans)
    length = raw['finish_reason'] == 'length'
    return {'request_id':raw['request_id'],'cell':f"{raw['arm']}_{raw['step']}_{raw['setting']}",
            'problem_id':raw['problem_id'],'sample_index':raw['sample_index'],'test_func_name':name,
            'completion_sha256':hashlib.sha256(text.encode()).hexdigest(),'completion_tokens':len(ids),
            'completion_cap':raw['sampling']['max_new_tokens'],'finish_reason':raw['finish_reason'],'stop_reason':raw['stop_reason'],
            'trailing_token_id':ids[-1] if ids else None, 'trailing_eos':bool(ids and ids[-1] == tokenizer.tokenizer.eos_token_id),
            'length_finished':length,'exactly_at_cap':len(ids)==raw['sampling']['max_new_tokens'],
            'native_evaluator_present':scored['response_has_test_func'],'native_can_compile':scored['can_compile'],
            'native_evaluator_location':mapped(native_location)if native_location else None,'native_anchor_gap':native_gap,
            'lexical_evaluator_definition_tokens':[locations[x]for x in lexical],
            'lexical_solution_class_tokens':[locations[x]for x in lexical_solution],
            'ast_evaluator_locations':[mapped(x)for x in ast_functions],
            'joined_code_syntax_error':syntax_error,'code_blocks':blocks,'fenced_code_token_count':len(code_tokens),
            'ast_solution_implementation_tokens':len(solution_tokens),'solution_span_observed':bool(solution_spans),
            'long_ast_solution':len(solution_tokens)>=1024,
            'unfinished_solution_proxy':length and syntax_error is not None and bool(lexical_solution),
            'open_final_code_fence':bool(blocks and not blocks[-1]['closed_fence']),
            'alignment_method':method,'repetition':repetition(ids,text)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packet',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True)
    ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--cap',type=int,choices=[1536,3072],required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    packet=args.packet.resolve();out=args.output.resolve();assert not out.exists();out.mkdir(parents=True)
    native=packet/'stage/manifest.json';assert digest(native)==args.manifest_sha256
    manifest=json.loads(native.read_text());proof=json.loads((packet/'output/INDEPENDENT_VERIFICATION.json').read_text())
    assert proof['status']=='succeeded'and proof['cells']==4 and proof['samples']==4760 and proof['manifest_sha256']==args.manifest_sha256
    assert manifest['projection_restoration']['checkpoint_step']==100
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True);tokenizer=CachedDecode(tok)
    inventory=json.loads((REPO/'artifacts/projection-restoration-r0-100-20260911/inputs_v1/existing_gpu02/base_inventory.json').read_text())
    tokenizer_refs=[]
    for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:
        p=args.tokenizer/name;r=reference(p.resolve());assert {k:r[k]for k in ['sha256','size_bytes']}==inventory['files'][name];tokenizer_refs.append(r)
    structural,parser=load_sources();records=[];selected_text={};input_refs=[];seen=set()
    with(out/'rows.jsonl').open('x')as rows_out:
        for cell in CELLS:
            root=packet/'output/cells'/cell;rawseal=json.loads((root/'RAW_COMPLETE.json').read_text());scoreseal=json.loads((root/'SCORE_COMPLETE.json').read_text())
            paths=[root/'raw.jsonl',packet/'output'/Path(scoreseal['results']['path']).relative_to(Path(manifest['output']))]
            for path,binding in zip(paths,[rawseal['raw'],scoreseal['results']]):
                assert digest(path)==binding['sha256']and path.stat().st_size==binding['size_bytes'];input_refs.append(reference(path))
            with paths[0].open()as rawstream,paths[1].open()as scorestream:
                count=0
                from itertools import zip_longest
                for rawline,scoreline in zip_longest(rawstream,scorestream):
                    assert rawline and scoreline
                    raw,score=json.loads(rawline),json.loads(scoreline)
                    assert raw['sampling']['max_new_tokens']==args.cap and raw['request_id']not in seen;seen.add(raw['request_id'])
                    result=analyze_row(raw,score,tokenizer,structural,parser);assert result['cell']==cell
                    rows_out.write(json.dumps(result,sort_keys=True,allow_nan=False)+'\n');records.append(result)
                    selected_text[result['request_id']]=raw['completion'];count+=1
                assert count==1190
    assert len(seen)==4760
    summaries=[]
    flags=['native_evaluator_present','native_can_compile','trailing_eos','exactly_at_cap','long_ast_solution','unfinished_solution_proxy','open_final_code_fence','solution_span_observed']
    for cell in CELLS:
        for stratum in ['all','length','not_length']:
            rows=[r for r in records if r['cell']==cell and (stratum=='all'or r['length_finished']==(stratum=='length'))]
            fractions=sorted(r['repetition']['repeated_8gram_excess_fraction']for r in rows)
            summaries.append({'cell':cell,'stratum':stratum,'count':len(rows),
                'counts':{k:sum(bool(r[k])for r in rows)for k in flags},
                'lexical_evaluator_starts':sum(bool(r['lexical_evaluator_definition_tokens'])for r in rows),
                'ast_named_evaluator_definitions':sum(bool(r['ast_evaluator_locations'])for r in rows),
                'joined_code_syntax_errors':sum(r['joined_code_syntax_error']is not None for r in rows),
                'tandem_repetition_flags':sum(r['repetition']['tandem_flag']for r in rows),
                'line_repetition_flags':sum(r['repetition']['line_flag']for r in rows),
                'repeated_8gram_excess_quantiles':{str(q):fractions[round((len(fractions)-1)*q)]if fractions else None for q in [0,.5,.9,.99,1]}})
    selected=collections.defaultdict(list)
    for flag in ['long_ast_solution','unfinished_solution_proxy','native_evaluator_present','open_final_code_fence']:
        for truncated in [False,True]:
            candidates=[r for r in records if r[flag]and r['length_finished']==truncated]
            if candidates: selected[min(candidates,key=lambda r:(r['problem_id'],r['sample_index'],r['cell']))['request_id']].append(f'{flag}:length={truncated}')
    for r in records:
        if r['lexical_evaluator_definition_tokens']or r['ast_evaluator_locations']: selected[r['request_id']].append('all_evaluator_starts')
    for r in sorted(records,key=lambda r:(-r['repetition']['repeated_8gram_excess_fraction'],r['problem_id'],r['sample_index'],r['cell']))[:4]:
        selected[r['request_id']].append('top_four_repeated_8gram_excess')
    examples=[];(out/'examples').mkdir()
    for rid,reasons in sorted(selected.items()):
        row=next(r for r in records if r['request_id']==rid);path=out/'examples'/f"{row['cell']}_{row['problem_id']}_{row['sample_index']}.txt"
        path.write_text(selected_text[rid]);examples.append(dict(row,selection_reasons=reasons,completion_file=path.name))
    payload={'status':'completed_offline_token_region_analysis','rows':4760,'cap':args.cap,'native_manifest':reference(native),
             'input_cells':input_refs,'tokenizer_files':tokenizer_refs,'protocol':reference(HERE/'PROTOCOL.json'),
             'summaries':summaries,'alignment_methods':dict(collections.Counter(r['alignment_method']for r in records)),
             'native_anchor_gaps':[{'request_id':r['request_id'],'reason':r['native_anchor_gap']}for r in records if r['native_anchor_gap']],
             'new_model_calls':0,'generated_code_executions':0}
    (out/'SUMMARY.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    (out/'EXAMPLES.json').write_text(json.dumps(examples,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:payload[k]for k in ['status','rows','cap','alignment_methods','native_anchor_gaps']}))

if __name__=='__main__':main()
