"""vLLM worker that adds vectors to the residual stream at the CAFT positions (protocol draft v6, adding tool).

The positions come from the reviewed CAFT rollout code: verl.utils.caft_vllm.runner_mask, which marks the rows whose
logical position is at or after the request's last prompt token, in the runner's packed row order. It is computed in
a wrapper around the runner's _prepare_inputs, as the CAFT worker does. The engine must run eagerly with compilation
off (enforce_eager=True, compilation level 0), so the Python forward hooks run on every step.

On each Qwen3 decoder layer with a vector v, the split residual output (hidden, residual) becomes, on masked rows:
hidden + residual + v (float32, cast back), with the residual set to zero, like split_residual_projection does for
projection. Unmasked rows keep their split form. Per-layer counts of changed rows are kept on the GPU for the
coverage check: every added layer must have changed exactly the masked rows.
"""
from types import MethodType

import torch
from vllm.v1.worker.gpu_worker import Worker


class AddController:
    def __init__(self, runner):
        from verl.utils.caft_vllm import runner_mask
        self.runner_mask = runner_mask
        self.runner = runner
        self.enabled = False
        self.vecs = {}
        self.gpu_mask = torch.zeros(runner.max_num_tokens, dtype=torch.bool, device=runner.device)
        self.n_rows = 0
        self.selected = 0
        self.scheduled = 0
        layers = [m for m in runner.model.modules() if type(m).__name__ == 'Qwen3DecoderLayer']
        if len(layers) != 36:
            raise RuntimeError(f'expected 36 Qwen3 decoder layers, found {len(layers)}')
        self.counts = torch.zeros(36, dtype=torch.int64, device=runner.device)
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(i))
        original = runner._prepare_inputs
        controller = self

        def prepare(this, scheduler_output):
            result = original(scheduler_output)
            if controller.enabled:
                mask, total, selected = controller.runner_mask(this, scheduler_output, result)
                controller.gpu_mask.zero_()
                controller.gpu_mask[:total].copy_(mask)
                controller.n_rows = total
                controller.selected += selected
                controller.scheduled += total
            return result
        runner._prepare_inputs = MethodType(prepare, runner)

    def _hook(self, index):
        def hook(module, args, output):
            v = self.vecs.get(index)
            if v is None or not self.enabled:
                return output
            hidden, residual = output
            n = hidden.shape[0]
            if n != self.n_rows:
                raise RuntimeError(f'layer {index}: {n} rows but {self.n_rows} scheduled (padding is not supported)')
            m = self.gpu_mask[:n]
            new = (hidden[m].float() + residual[m].float() + v).to(hidden.dtype)
            next_hidden, next_residual = hidden.clone(), residual.clone()
            next_hidden[m] = new
            next_residual[m] = 0
            self.counts[index] += m.sum()
            return next_hidden, next_residual
        return hook

    def configure(self, vecs):
        """vecs: {block: list of 2560 floats} (the full displacement). Resets the counters and enables adding."""
        self.vecs = {int(b): torch.tensor(v, dtype=torch.float32, device=self.runner.device) for b, v in vecs.items()}
        self.counts.zero_()
        self.selected = self.scheduled = 0
        self.enabled = True
        return {'blocks': sorted(self.vecs)}

    def finish(self):
        self.enabled = False
        counts = self.counts.cpu().tolist()
        receipt = {'predictor_rows': int(self.selected), 'scheduled_rows': int(self.scheduled),
                   'changed_rows': {str(b): int(counts[b]) for b in sorted(self.vecs)},
                   'coverage_ok': all(counts[b] == self.selected for b in self.vecs) and
                   all(c == 0 for i, c in enumerate(counts) if i not in self.vecs)}
        self.vecs = {}
        return receipt


class AddWorker(Worker):
    def load_model(self):
        super().load_model()
        self.model_runner._add_controller = AddController(self.model_runner)


def configure(worker, vecs):
    return worker.model_runner._add_controller.configure(vecs)


def finish(worker):
    return worker.model_runner._add_controller.finish()
