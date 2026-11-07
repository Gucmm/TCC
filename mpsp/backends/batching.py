"""Geração em lote para o backend HF: agrupa prompts por tamanho, respeita um orçamento de tokens e divide o lote
quando falta memória. Sem torch: a geração em si (`gen_fn`) é injetada, o que permite testar a lógica isolada."""
from typing import Callable, List, Sequence

from .base import ContextOverflow, RawCompletion


def plan_batches(lengths: Sequence[int], max_seqs: int, max_tokens: int) -> List[List[int]]:
    """Índices agrupados em ordem crescente de tamanho (menos padding). Um lote respeita `max_seqs` sequências e
    `max_tokens` tokens de prompt já com padding (nº de sequências × a maior); uma sequência acima do orçamento vai sozinha."""
    batches: List[List[int]] = []
    cur: List[int] = []
    for i in sorted(range(len(lengths)), key=lambda k: lengths[k]):
        if cur and (len(cur) >= max_seqs or (len(cur) + 1) * lengths[i] > max_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def generate_all(prompt_ids: Sequence[List[int]], gen_fn: Callable[[List[List[int]]], List[List[int]]], *,
                 ctx_tokens: int, max_new_tokens: int, max_seqs: int, max_tokens: int,
                 on_oom: Callable[[], None] = lambda: None) -> list:
    """→ lista alinhada com `prompt_ids`: ids gerados, ou a exceção do item (ContextOverflow, falha de geração).
    `gen_fn` recebe uma lista de prompts (ids) e devolve uma lista alinhada de ids gerados. Se um lote estoura a
    memória, é dividido ao meio até chegar a um prompt isolado; só então o erro é atribuído ao item."""
    out: list = [None] * len(prompt_ids)
    todo = []
    for i, ids in enumerate(prompt_ids):
        if len(ids) + max_new_tokens > ctx_tokens:
            out[i] = ContextOverflow(f"{len(ids)}+{max_new_tokens} > {ctx_tokens}")
        else:
            todo.append(i)

    def run(idx: List[int]):
        try:
            res = gen_fn([prompt_ids[i] for i in idx])
            if len(res) != len(idx):
                raise RuntimeError(f"gen_fn devolveu {len(res)} saídas para {len(idx)} prompts")
        except Exception as e:  # noqa: BLE001
            if is_oom(e) and len(idx) > 1:
                on_oom()
                mid = len(idx) // 2
                run(idx[:mid])
                run(idx[mid:])
            else:
                for i in idx:
                    out[i] = e
            return
        for i, r in zip(idx, res):
            out[i] = r

    lengths = [len(prompt_ids[i]) for i in todo]
    for b in plan_batches(lengths, max_seqs, max_tokens):
        run([todo[k] for k in b])
    return out


def to_completions(gen_ids: list, decode: Callable[[List[int]], str], prompt_ids: Sequence[List[int]]) -> list:
    """Converte a saída de `generate_all` em RawCompletion (mantém as exceções)."""
    return [g if isinstance(g, Exception) else
            RawCompletion(text=decode(g), prompt_tokens=len(p), completion_tokens=len(g))
            for g, p in zip(gen_ids, prompt_ids)]
