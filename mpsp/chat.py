"""Renderização de chat compartilhada por treino e inferência.

Treino e inferência PRECISAM tokenizar o prompt do mesmo jeito; por isso há
uma única implementação. Independe de versão do `transformers`: renderiza em
texto (`tokenize=False`) e tokeniza em seguida.
"""
from typing import Dict, List, Optional, Tuple


def prompt_text(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


def completion_text(tok, prompt: str, completion: str) -> str:
    """Parte da resposta (inclui o token de fim de turno próprio do modelo)."""
    p = prompt_text(tok, prompt)
    full = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        tokenize=False, add_generation_prompt=False)
    if full.startswith(p):
        return full[len(p):]
    return completion + (tok.eos_token or "")      # template atípico


def encode(tok, text: str) -> List[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def build_example(tok, prompt: str, completion: str, max_len: int) -> Optional[Dict[str, List[int]]]:
    """Exemplo de SFT com loss só na resposta. Devolve None se não couber em
    `max_len` (nunca trunca: JSON truncado ensinaria o modelo a gerar JSON inválido)."""
    p_ids = encode(tok, prompt_text(tok, prompt))
    c_ids = encode(tok, completion_text(tok, prompt, completion))
    if len(p_ids) + len(c_ids) > max_len:
        return None
    ids = p_ids + c_ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(p_ids) + c_ids}
