"""Backend vLLM: batching contínuo, cache de prefixo e LoRA, em bf16 (a A40 de 48 GB comporta 7–9B sem quantizar).

Requer `pip install vllm` (de preferência em um venv separado: o vLLM fixa a própria versão do torch).
Os tokens são montados por `mpsp.chat` (mesmo prompt do treino) e enviados já tokenizados, evitando BOS duplicado.
Decodificação restrita ao schema NÃO está ligada aqui (a API de saídas estruturadas muda entre versões):
a validação + retentativa de `Backend.generate_batch` continua valendo.
"""
from typing import List

from .base import Backend, ContextOverflow, RawCompletion
from .. import chat


class VLLMBackend(Backend):
    needs_schema_text = True

    def __init__(self, hf_id: str, name: str = None, ctx_tokens: int = 8192, adapter: str = None,
                 max_new_tokens: int = 3072, max_model_len: int = 16384, gpu_memory_utilization: float = 0.90,
                 max_num_seqs: int = 64, dtype: str = "bfloat16", enable_prefix_caching: bool = True,
                 trust_remote_code: bool = False, **_):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.name = name or hf_id
        self.ctx_tokens = min(ctx_tokens, max_model_len)       # KV cache é dimensionado por este valor
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        kw = dict(model=hf_id, dtype=dtype, max_model_len=self.ctx_tokens,
                  gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
                  enable_prefix_caching=enable_prefix_caching, trust_remote_code=trust_remote_code)
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest
            kw.update(enable_lora=True, max_lora_rank=64)
            self.lora = LoRARequest("sft", 1, adapter)
        self.llm = LLM(**kw)

    def _complete(self, prompt, schema) -> RawCompletion:
        out = self._complete_batch([prompt], schema)[0]
        if isinstance(out, Exception):
            raise out
        return out

    def _complete_batch(self, prompts: List[str], schema) -> list:
        from vllm import SamplingParams
        ids = [chat.encode(self.tok, chat.prompt_text(self.tok, p)) for p in prompts]
        outs: list = [None] * len(prompts)
        run = []
        for i, x in enumerate(ids):
            if len(x) + self.max_new_tokens > self.ctx_tokens:
                outs[i] = ContextOverflow(f"{len(x)}+{self.max_new_tokens} > {self.ctx_tokens}")
            else:
                run.append(i)
        if run:
            params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)
            res = self.llm.generate([{"prompt_token_ids": ids[i]} for i in run], params,
                                    lora_request=self.lora, use_tqdm=False)
            for i, r in zip(run, res):
                o = r.outputs[0]
                outs[i] = RawCompletion(text=o.text, prompt_tokens=len(ids[i]), completion_tokens=len(o.token_ids))
        return outs
