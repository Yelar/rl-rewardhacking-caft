"""Install CAFT before the native worker profiles or captures any graph."""
from vllm.v1.worker.gpu_worker import Worker


class CAFTWorker(Worker):
    def load_model(self):
        super().load_model()
        from verl.utils.caft_vllm import install_worker
        spec=self.vllm_config.additional_config['caft_training']
        self.caft_initial_installation=install_worker(self,spec)
