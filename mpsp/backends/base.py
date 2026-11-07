"""Interface única de backend local + parsing/validação/retentativa, em lote."""
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from pydantic import BaseModel, ValidationError


class ContextOverflow(Exception):
    """Prompt + geração não cabem na janela de contexto do modelo."""


@dataclass
class RawCompletion:
    text: str = ""
    parsed: Optional[BaseModel] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Generation:
    parsed: Optional[BaseModel] = None
    raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0        # em lote, é o tempo amortizado (tempo do lote ÷ nº de prompts)
    retries: int = 0
    error: Optional[str] = None

    def meta(self) -> dict:
        return {k: getattr(self, k) for k in (
            "prompt_tokens", "completion_tokens", "seconds", "retries", "error")}


def parse_json_model(text: str, schema: type) -> Tuple[Optional[BaseModel], Optional[str]]:
    """Extrai o primeiro objeto JSON do texto (tolera cercas markdown) e valida."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    candidates = [t]
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))
    err = "json_invalido"
    for c in candidates:
        try:
            return schema.model_validate(json.loads(c)), None
        except json.JSONDecodeError as e:
            err = f"json_invalido: {e.msg}"
        except ValidationError as e:
            err = f"schema_invalido: {e.errors()[0]['loc']} {e.errors()[0]['msg']}"
    return None, err


class Backend(ABC):
    name: str = "backend"
    needs_schema_text: bool = False       # True p/ SLM local (sem decodificação estruturada nativa)
    ctx_tokens: Optional[int] = None
    max_retries: int = 1

    @abstractmethod
    def _complete(self, prompt: str, schema: type) -> RawCompletion: ...

    def _complete_batch(self, prompts: List[str], schema: type) -> List[Any]:
        """→ lista alinhada com `prompts`: RawCompletion, ou a exceção levantada.
        Engines com lote real (vLLM) sobrescrevem este método."""
        out: List[Any] = []
        for p in prompts:
            try:
                out.append(self._complete(p, schema))
            except Exception as e:  # noqa: BLE001
                out.append(e)
        return out

    def generate_batch(self, prompts: List[str], schema: type) -> List[Generation]:
        """Gera todos os prompts; só os inválidos são reenviados (1 retentativa, com o erro no prompt)."""
        gens = [Generation() for _ in prompts]
        pending = list(range(len(prompts)))
        for attempt in range(self.max_retries + 1):
            if not pending:
                break
            batch = [prompts[i] if attempt == 0 else
                     prompts[i] + f"\n\nA resposta anterior foi inválida ({gens[i].error}). "
                                  "Responda apenas com JSON válido conforme o schema."
                     for i in pending]
            t0 = time.perf_counter()
            outs = self._complete_batch(batch, schema)
            dt = (time.perf_counter() - t0) / len(pending)
            retry, crashed = [], False
            for i, o in zip(pending, outs):
                g = gens[i]
                g.seconds += dt
                g.retries = attempt
                if isinstance(o, ContextOverflow):
                    g.error = "context_overflow"           # não adianta reenviar
                    continue
                if isinstance(o, Exception):
                    g.error = f"{type(o).__name__}: {o}"[:300]
                    retry.append(i)
                    crashed = True
                    continue
                g.prompt_tokens += o.prompt_tokens
                g.completion_tokens += o.completion_tokens
                g.raw = o.text
                parsed, err = (o.parsed, None) if o.parsed is not None else parse_json_model(o.text, schema)
                if parsed is not None:
                    g.parsed, g.error = parsed, None
                else:
                    g.error = err
                    retry.append(i)
            pending = retry
            if crashed and pending and attempt < self.max_retries:
                time.sleep(min(2 * (attempt + 1), 10))
        return gens

    def generate(self, prompt: str, schema: type) -> Generation:
        return self.generate_batch([prompt], schema)[0]


class MockBackend(Backend):
    """Backend determinístico para testes e dry-run: `responder(prompt, schema) -> dict|str`."""
    name = "mock"

    def __init__(self, responder: Optional[Callable[[str, type], Any]] = None, ctx_tokens: Optional[int] = None):
        self.responder = responder or _minimal_valid
        self.ctx_tokens = ctx_tokens
        self.calls = []

    def _complete(self, prompt: str, schema: type) -> RawCompletion:
        self.calls.append((prompt, schema))
        out = self.responder(prompt, schema)
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return RawCompletion(text=text, prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4)


def _minimal_valid(prompt: str, schema: type) -> dict:
    out = {}
    if "resumo_processo" in schema.model_fields:
        out["resumo_processo"] = {"pessoas_envolvidas": {}, "classificacao_crime": "Homicídio"}
    return out
