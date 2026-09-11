"""Observe finished vLLM outputs without submitting, sampling, or advancing twice."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

SCHEMA = 'native_single_chat_finished_requests_v1'
PINS = {
    'entrypoints/llm.py': 'ee4a1ae908f6e04b822c8ef55bb1ecff89a781cee90944a3d989e85286b6489a',
    'v1/engine/llm_engine.py': '3af31a24d08709a3bb4001d1665c8bc8953a3cc2b62241f758f433afacbd6afa',
    'v1/engine/output_processor.py': '1999e687a183ce4b7b039b4c5357b73f0ad0afd8a5b52383ef1f7d1d5613263e',
    'v1/engine/parallel_sampling.py': 'b95885687e4d1713107d6073f2b39d459e5e21bfd91cc949bb7682ba1ab64c83',
}


def require(ok, text):
    if not ok:
        raise ValueError('Finished-output retention: ' + text)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def qualify_runtime(model):
    """Read already-installed source bytes; no engine or CUDA calls."""
    require(importlib.metadata.version('vllm') == '0.11.0', 'unqualified vLLM version')
    path = Path(inspect.getfile(type(model))).resolve()
    root = path.parent.parent
    require(path == root / 'entrypoints/llm.py', 'unexpected LLM implementation')
    require(Path(inspect.getfile(type(model.llm_engine))).resolve() == root / 'v1/engine/llm_engine.py',
            'unexpected engine implementation')
    for name, expected in PINS.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                'installed source changed: ' + name)
    return dict(version='0.11.0', source_sha256=dict(PINS),
                capture='Finished parent RequestOutputs returned by the unchanged engine.step; FINAL_ONLY n=10.')


class FinishedRequests:
    """One run, one original chat call, fsync before returning each finished group.

    ``rows_for_output`` is the caller's unchanged native raw-record serializer.
    The engine's argument values, output objects, scheduling calls and RNG remain
    untouched. Synchronous disk latency is recorded as a scientific limitation.
    """
    def __init__(self, model, path, identity, rows_for_output, *, problems=119, samples=10):
        self.model, self.path, self.identity = model, Path(path), identity
        self.rows_for_output = rows_for_output
        self.problems, self.samples = problems, samples
        self.events = {}
        self.fd = None

    def append(self, value):
        raw = (canonical(value) + '\n').encode()
        view = memoryview(raw)
        while view:
            written = os.write(self.fd, view)
            require(written > 0, 'journal write made no progress')
            view = view[written:]
        os.fsync(self.fd)

    def __enter__(self):
        require(self.model.request_counter.counter == 0, 'engine is not fresh; request IDs ambiguous')
        require(self.problems == 119 and self.samples == 10, 'original119 x10 protocol required')
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            self.append(dict(schema=SCHEMA, kind='identity', identity=self.identity,
                             problems=self.problems, samples=self.samples, request_counter_start=0))
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self.engine = self.model.llm_engine
            self.had_instance_step = 'step' in vars(self.engine)
            self.previous_instance_step = vars(self.engine).get('step')
            self.original_step = self.engine.step
            self.engine.step = self.step
        except BaseException:
            os.close(self.fd); self.fd = None
            raise
        return self

    def observe(self, result):
        if result.finished is not True:
            return
        request_id = result.request_id
        require(isinstance(request_id, str) and request_id.isdecimal(), 'unexpected parent request ID')
        index = int(request_id)
        require(str(index) == request_id and 0 <= index < self.problems, 'parent request ID outside original batch')
        require(len(result.outputs) == self.samples and [o.index for o in result.outputs] == list(range(self.samples))
                and all(o.finish_reason is not None for o in result.outputs), 'incomplete/misordered parent outputs')
        rows = self.rows_for_output(index, result)
        require(len(rows) == self.samples and [r['sample_index'] for r in rows] == list(range(self.samples))
                and all(r['problem_index'] == index and r['engine_request_id'] == request_id for r in rows),
                'native raw row mapping differs')
        event = dict(schema=SCHEMA, kind='finished_parent', identity=self.identity,
                     problem_index=index, engine_request_id=request_id, rows=rows)
        if index in self.events:
            require(canonical(self.events[index]) == canonical(event), 'conflicting repeated parent output')
            return
        self.append(event)
        self.events[index] = event

    def step(self, *args, **kwargs):
        outputs = self.original_step(*args, **kwargs)
        for output in outputs:
            self.observe(output)
        return outputs

    def finish(self, outputs):
        require(self.model.request_counter.counter == self.problems, 'original request count changed')
        require(len(outputs) == self.problems and set(self.events) == set(range(self.problems)),
                'single chat returned without every retained parent')
        for index, output in enumerate(outputs):
            require(output.request_id == str(index), 'native final output order changed')
            require(canonical(self.rows_for_output(index, output)) == canonical(self.events[index]['rows']),
                    'retained rows differ from original chat return')
        return [row for index in range(self.problems) for row in self.events[index]['rows']]

    def __exit__(self, typ, value, traceback):
        if self.had_instance_step:
            self.engine.step = self.previous_instance_step
        else:
            del self.engine.step
        os.close(self.fd)
        self.fd = None


def read_retained(path, expected_identity):
    """Read complete journal lines after worker release; never resume generation.

    A torn final line is reported and excluded. A corrupt complete line, wrong
    identity or conflicting completed request fails instead of being hidden.
    """
    raw = Path(path).read_bytes()
    end = raw.rfind(b'\n') + 1
    require(end > 0, 'no complete identity record')
    lines = [json.loads(line) for line in raw[:end].splitlines()]
    header = lines[0]
    require(header == dict(schema=SCHEMA, kind='identity', identity=expected_identity,
                           problems=119, samples=10, request_counter_start=0), 'journal identity differs')
    events = {}
    ids = set()
    for event in lines[1:]:
        require(event.get('schema') == SCHEMA and event.get('kind') == 'finished_parent' and
                event.get('identity') == expected_identity, 'event identity differs')
        index = event['problem_index']
        require(type(index) is int and 0 <= index < 119 and event['engine_request_id'] == str(index),
                'event parent mapping differs')
        rows = event['rows']
        require(len(rows) == 10 and [r['sample_index'] for r in rows] == list(range(10)) and
                all(r['problem_index'] == index and r['engine_request_id'] == str(index) for r in rows),
                'event row mapping differs')
        if index in events:
            require(canonical(events[index]) == canonical(event), 'conflicting journal duplicate')
            continue
        request_ids = [r['request_id'] for r in rows]
        require(len(set(request_ids)) == 10 and not set(request_ids) & ids, 'duplicate raw request identity')
        ids.update(request_ids); events[index] = event
    return dict(schema=SCHEMA, identity=expected_identity, complete_parents=len(events),
                complete_responses=len(ids), incomplete_tail_bytes=len(raw) - end,
                rows=[row for index in sorted(events) for row in events[index]['rows']],
                full_raw_completion_seal_created=False, generation_resume_authorized=False)
