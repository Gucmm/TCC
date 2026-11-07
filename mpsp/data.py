"""GT, textos, corpus e partições (k-fold estratificado)."""
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import canonical


def load_gt(path: str) -> Tuple[Dict[str, dict], List[str]]:
    """→ (GT canônico validado, erros). Aceita qualquer formato legado de chaves."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    gt, errors = {}, []
    for doc_id, obj in raw.items():
        try:
            gt[str(doc_id)] = canonical.dump(canonical.validate(obj))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{doc_id}: {str(e).splitlines()[0]}")
    return gt, errors


def find_text(doc_id: str, docs_dir: str) -> Optional[Path]:
    d = Path(docs_dir)
    for cand in (d / f"{doc_id}.txt", *d.glob(f"*/{doc_id}/texto_processo.txt")):
        if cand.exists():
            return cand
    return None


def load_texts(ids: Iterable[str], docs_dir: str) -> Dict[str, str]:
    out = {}
    for i in ids:
        p = find_text(i, docs_dir)
        if p is not None:
            t = p.read_text(encoding="utf-8", errors="ignore").strip()
            if t:
                out[i] = t
    return out


def _id_set(path: str) -> set:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k) for k in data}
    if data and isinstance(data[0], dict):
        return {str(x["NumMP"]) for x in data if x.get("NumMP") is not None}
    return {str(x) for x in data}


def list_corpus(docs_dir: str, include: Iterable[str] = (), exclude: Iterable[str] = ()) -> List[str]:
    """Ids dos documentos em disco, filtrados por listas JSON de inclusão/exclusão."""
    d = Path(docs_dir)
    ids = {p.stem for p in d.glob("*.txt")} | {p.parent.name for p in d.glob("*/*/texto_processo.txt")}
    inc = set().union(*[_id_set(f) for f in include]) if include else None
    exc = set().union(*[_id_set(f) for f in exclude]) if exclude else set()
    return sorted((ids if inc is None else ids & inc) - exc)


def make_splits(gt: Dict[str, dict], texts: Dict[str, str], k: int = 5, seed: int = 42,
                val_frac: float = 0.15) -> dict:
    """k-fold estratificado por (classificação × tercil de tamanho). Cada documento é
    teste exatamente uma vez; val (early stopping) sai do restante, estratificado por classe."""
    ids = sorted(i for i in gt if i in texts)
    lens = sorted(len(texts[i]) for i in ids)
    cuts = (lens[len(lens) // 3], lens[2 * len(lens) // 3]) if lens else (0, 0)
    cls = {i: gt[i]["resumo_processo"]["classificacao_crime"] for i in ids}
    tercile = {i: int(len(texts[i]) > cuts[0]) + int(len(texts[i]) > cuts[1]) for i in ids}
    rng = random.Random(seed)
    strata: Dict[tuple, List[str]] = {}
    for i in ids:
        strata.setdefault((cls[i], tercile[i]), []).append(i)
    folds: List[List[str]] = [[] for _ in range(k)]
    n = 0
    for key in sorted(strata):
        members = strata[key][:]
        rng.shuffle(members)
        for i in members:
            folds[n % k].append(i)
            n += 1
    out = {"k": k, "seed": seed, "val_frac": val_frac, "folds": []}
    for f in range(k):
        test = sorted(folds[f])
        rest = [i for i in ids if i not in set(test)]
        val: List[str] = []
        by_cls: Dict[str, List[str]] = {}
        for i in rest:
            by_cls.setdefault(cls[i], []).append(i)
        for c in sorted(by_cls):
            members = by_cls[c][:]
            rng.shuffle(members)
            if len(members) >= 3:
                val += members[:max(1, round(len(members) * val_frac))]
        out["folds"].append({"train": sorted(set(rest) - set(val)), "val": sorted(val), "test": test})
    return out
