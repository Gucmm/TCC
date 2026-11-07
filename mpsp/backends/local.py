"""Backend local (Hugging Face Transformers), com adapter LoRA opcional e geração em lote."""
from .base import Backend, RawCompletion
from .batching import generate_all, to_completions
from .. import chat
from ..compat import dtype_kw


class LocalBackend(Backend):
    needs_schema_text = True

    def __init__(self, hf_id: str, name: str = None, ctx_tokens: int = 8192, adapter: str = None,
                 load_4bit: bool = True, max_new_tokens: int = 3072, constrained: bool = False,
                 trust_remote_code: bool = False, attn_implementation: str = None,
                 micro_batch: int = 4, max_batch_tokens: int = 24000, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.name, self.ctx_tokens = name or hf_id, ctx_tokens
        self.max_new_tokens, self.constrained = max_new_tokens, constrained
        self.micro_batch, self.max_batch_tokens = micro_batch, max_batch_tokens
        self.tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map="auto", **dtype_kw(torch.bfloat16), trust_remote_code=trust_remote_code,
                  attn_implementation=attn_implementation)
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def _eos_ids(self) -> set:
        e = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        ids = set(e if isinstance(e, (list, tuple)) else [e]) if e is not None else set()
        if self.tok.eos_token_id is not None:
            ids.add(self.tok.eos_token_id)
        return ids

    def _generate_ids(self, batch_ids, **kw):
        """Gera para vários prompts de uma vez (padding à esquerda). Devolve os ids gerados de cada um, cortados no
        primeiro token de fim (inclusive): o que vem depois é padding do `generate`."""
        import torch
        pad = self.tok.pad_token_id
        n = max(len(x) for x in batch_ids)
        x = torch.tensor([[pad] * (n - len(i)) + i for i in batch_ids], device=self.model.device)
        mask = torch.tensor([[0] * (n - len(i)) + [1] * len(i) for i in batch_ids], device=self.model.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=x, attention_mask=mask, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=pad, **kw)
        eos, res = self._eos_ids(), []
        for row in out[:, n:].tolist():
            cut = next((k + 1 for k, t in enumerate(row) if t in eos), len(row))
            res.append(row[:cut])
        return res

    def _decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def _prompt_ids(self, prompt):
        return chat.encode(self.tok, chat.prompt_text(self.tok, prompt))

    @staticmethod
    def _free_cuda():
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _constrained_kw(self, schema):
        from lmformatenforcer import JsonSchemaParser     # opcional: pip install lm-format-enforcer
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        return {"prefix_allowed_tokens_fn": build_transformers_prefix_allowed_tokens_fn(
            self.tok, JsonSchemaParser(schema.model_json_schema()))}

    def _complete(self, prompt, schema):
        ids = self._prompt_ids(prompt)
        kw = self._constrained_kw(schema) if self.constrained else {}
        gen = generate_all([ids], lambda b: self._generate_ids(b, **kw), ctx_tokens=self.ctx_tokens,
                           max_new_tokens=self.max_new_tokens, max_seqs=1, max_tokens=len(ids) + 1,
                           on_oom=self._free_cuda)[0]
        if isinstance(gen, Exception):
            raise gen
        return RawCompletion(text=self._decode(gen), prompt_tokens=len(ids), completion_tokens=len(gen))

    def _complete_batch(self, prompts, schema):
        if self.constrained or self.micro_batch <= 1 or len(prompts) == 1:
            return super()._complete_batch(prompts, schema)     # restrição por schema é por prompt: sem lote
        ids = [self._prompt_ids(p) for p in prompts]
        gens = generate_all(ids, self._generate_ids, ctx_tokens=self.ctx_tokens, max_new_tokens=self.max_new_tokens,
                            max_seqs=self.micro_batch, max_tokens=self.max_batch_tokens, on_oom=self._free_cuda)
        return to_completions(gens, self._decode, ids)
