"""Prompts modulares.

Cada nó de uma topologia declara apenas o que emite (`emits`); o prompt é
montado a partir de fragmentos compartilhados. Assim, as regras de um campo
são idênticas em T1..T5 e a topologia é a única variável do experimento.
"""
import enum
import json
import types
import typing
from typing import Any, Dict, Iterable, Sequence

from pydantic import BaseModel

from .schemas import ROLES, make_output_model

BASE = """Você é um assistente especializado em analisar promoções de arquivamento de inquéritos policiais do MPSP e extrair informações estruturadas.

Regras gerais:
- Utilize apenas informações explicitamente presentes no texto.
- Não invente dados e não use "não informado": campos sem informação podem ser omitidos ou ficar null.
- Preserve literalmente datas, horários, nomes e trechos citados.
- Responda somente com o JSON do schema, sem explicações."""

CRIME = """Classificação do crime:
- Classifique o fato como Homicídio, Latrocínio (roubo seguido de morte) ou Morte por causas naturais.
- Não use "Morte por causas naturais" se houver suspeitos/investigados ligados à morte."""

MAPA = """Mapeamento de pessoas (resumo_processo.pessoas_envolvidas):
- Liste o nome completo de todas as pessoas mencionadas, cada uma em um único papel (vítima, suspeito/investigado ou testemunha), salvo indicação explícita de mais de um papel."""

INQUERITO = """Informações do inquérito:
- razao_arquivamento: trecho literal do documento que fundamenta o arquivamento, sem resumo.
- é_feminicidio: true apenas se a classificação for Homicídio e houver a qualificadora.
- bem_roubado: preencher apenas se a classificação for Latrocínio.
- resultado: Latrocínio → "Consumado"; Homicídio → "Consumado" ou "Tentado" somente se explícito; Morte por causas naturais → em branco."""

PESSOAS = """Regras para pessoas:
- Quando o sexo não for explícito, infira-o pelo nome.
- é_policial só pode ser true se o texto afirmar explicitamente; nesse caso a corporação deve estar explícita. Do contrário, é_policial=false e corporacao_policial em branco.
- antecedentes_criminais: não preencha com informações contextuais (ex.: uso de álcool)."""

PAPEL = {
    "vitimas": "Vítimas: detalhe cada vítima listada. armada só é true se o texto mencionar a arma da vítima; faleceu indica se a vítima morreu; causa_juridica_da_morte somente se faleceu.",
    "suspeitos": "Suspeitos/investigados: detalhe cada suspeito/investigado listado.",
    "testemunhas": "Testemunhas: detalhe cada testemunha listada.",
}
ROTULO = {"vitimas": "Vítimas", "suspeitos": "Suspeitos/investigados", "testemunhas": "Testemunhas"}


def fragments_for(emits: Iterable[str]) -> str:
    emits = tuple(emits)
    parts = []
    if "resumo_processo" in emits:
        parts += [CRIME, MAPA]
    if "inquerito" in emits:
        parts.append(INQUERITO)
    if any(r in emits for r in ROLES):
        parts.append(PESSOAS)
        parts += [PAPEL[r] for r in ROLES if r in emits]
    return "\n\n".join(parts)


def context_text(emits: Sequence[str], ctx: Dict[str, Any] | None) -> str:
    """Contexto vindo do primeiro nó (só para nós que não emitem resumo_processo)."""
    if not ctx or "resumo_processo" in emits:
        return ""
    lines = []
    if ctx.get("classificacao"):
        lines.append(f"Tipo de crime: {ctx['classificacao']}")
    for r in ROLES:
        if r in emits:
            nomes = ", ".join(ctx.get(r) or []) or "(nenhuma)"
            lines.append(f"{ROTULO[r]} a detalhar: {nomes}")
    return "\n".join(lines)


def drop_nulls(obj: Any) -> Any:
    """Remove chaves None (lossless: todo campo opcional tem default None)."""
    if isinstance(obj, dict):
        return {k: drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [drop_nulls(v) for v in obj]
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(drop_nulls(obj), ensure_ascii=False, separators=(",", ":"))


def _unwrap(a):
    if typing.get_origin(a) in (typing.Union, types.UnionType):
        args = [x for x in typing.get_args(a) if x is not type(None)]
        if len(args) == 1:
            return args[0]
    return a


def _tipo(a) -> str:
    a = _unwrap(a)
    if typing.get_origin(a) is list:
        return f"lista de {_tipo(typing.get_args(a)[0])}"
    if isinstance(a, type) and issubclass(a, enum.Enum):
        return " | ".join(f'"{v.value}"' for v in a)
    if isinstance(a, type) and issubclass(a, BaseModel):
        return f"objeto {a.__name__}"
    return {bool: "true/false", str: "texto", float: "número"}.get(a, str(a))


def _render(model: type, seen: set, pad: str = "") -> list:
    if model in seen:
        return [f"{pad}(mesma estrutura de {model.__name__})"]
    seen.add(model)
    lines = []
    for name, f in model.model_fields.items():
        desc = f" — {f.description}" if f.description else ""
        lines.append(f"{pad}- {name} ({_tipo(f.annotation)}){desc}")
        a = _unwrap(f.annotation)
        if typing.get_origin(a) is list:
            a = _unwrap(typing.get_args(a)[0])
        if isinstance(a, type) and issubclass(a, BaseModel):
            lines += _render(a, seen, pad + "  ")
    return lines


def describe_schema(model: type) -> str:
    return ("Schema da resposta (JSON com exatamente estes campos; não invente valores "
            "fora das opções listadas):\n" + "\n".join(_render(model, set())))


def build_prompt(emits: Sequence[str], document: str, ctx: Dict[str, Any] | None = None,
                 shots: Sequence[dict] = (), with_schema: bool = False) -> str:
    """`shots`: lista de {"text", "ctx", "output"}; `with_schema`: anexa o schema
    em texto (modelos locais não têm decodificação estruturada nativa)."""
    emits = tuple(emits)
    blocks = [BASE, fragments_for(emits)]
    if with_schema:
        blocks.append(describe_schema(make_output_model(emits)))
    if shots:
        ex = ["Exemplos resolvidos:"]
        for i, s in enumerate(shots, 1):
            c = context_text(emits, s.get("ctx"))
            ex.append(f"### Exemplo {i}\n" + (c + "\n" if c else "")
                      + f"Promoção de arquivamento:\n{s['text']}\nResposta (JSON):\n{dumps(s['output'])}")
        blocks.append("\n\n".join(ex))
        blocks.append("### Documento a processar")
    c = context_text(emits, ctx)
    if c:
        blocks.append(c)
    blocks.append(f"Promoção de arquivamento:\n{document}\n\nResposta (JSON):")
    return "\n\n".join(blocks)
