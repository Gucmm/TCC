"""Avaliador único (relatório, validação do SFT e recompensa do RL usam este código).

Decisões:
- Cada campo vira uma linha com presença no GT e na predição; verdadeiros negativos
  (ambos ausentes) ficam explícitos e NÃO entram em P/R/F1.
- Pessoas são pareadas por atribuição ótima (Hungarian) sobre similaridade de nomes.
- Predição ausente/inválida conta como tudo ausente (penaliza o recall).
- Campos de pessoa só são pontuados em pares casados; o erro de entidade é pontuado à parte.
- `inquerito` só é pontuado se o GT o rotulou.
- Duas visões: `strict` (igualdade normalizada) e `soft` (token-F1 / ROUGE-L ≥ limiar).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import canonical
from .schemas import ROLES
from .textnorm import (is_null, name_similarity, norm, norm_date, norm_time, rouge_l,
                       token_f1)

KIND = {
    "classificacao_crime": "cat", "resultado": "cat", "natureza_da_autoria": "cat",
    "é_feminicidio": "bool", "pericia_realizada": "bool", "prisao_em_flagrante": "bool",
    "é_policial": "bool", "policial_em_servico": "bool", "armada": "bool", "faleceu": "bool",
    "latitude": "num", "longitude": "num",
    "data_ocorrencia": "date", "data_registro": "date", "hora_ocorrencia": "time",
    "razao_arquivamento": "literal",
}
INQUERITO = ["resultado", "é_feminicidio", "data_ocorrencia", "hora_ocorrencia", "data_registro",
             "delegacia_registro", "tipo_de_local", "local_detalhado", "latitude", "longitude",
             "municipio", "arma_utilizada", "bem_roubado", "natureza_da_autoria", "pericia_realizada",
             "prisao_em_flagrante", "razao_arquivamento"]
PESSOA = ["cor", "sexo", "profissao", "escolaridade", "nacionalidade", "idade", "antecedentes_criminais",
          "é_policial", "corporacao_policial", "policial_em_servico"]
VITIMA = ["armada", "arma_da_vítima", "faleceu", "causa_juridica_da_morte", "relacao_vitima_autor"]

SOFT_TEXT, SOFT_LITERAL, NAME_THRESHOLD = 0.5, 0.8, 0.5
COLS = ["doc_id", "section", "role", "entity", "field", "kind", "gt_present", "pred_present",
        "strict", "soft", "score", "gt", "pred"]


def compare(kind: str, gt: Any, pred: Any) -> Tuple[bool, bool, bool, float]:
    """→ (gt_presente, pred_presente, match_strict, match_soft_score)."""
    if kind == "bool":
        g, p = gt is True, pred is True
        return g, p, g and p, float(g and p)
    g_present, p_present = not is_null(gt), not is_null(pred)
    if not (g_present and p_present):
        return g_present, p_present, False, 0.0
    if kind == "num":
        try:
            d = abs(float(gt) - float(pred))
        except (TypeError, ValueError):
            return True, True, False, 0.0
        return True, True, d <= 1e-4, float(d <= 1e-2)
    if kind == "date":
        e = norm_date(gt) == norm_date(pred)
        return True, True, e, float(e)
    if kind == "time":
        e = norm_time(gt) == norm_time(pred)
        return True, True, e, float(e)
    exact = norm(gt) == norm(pred)
    if kind == "cat":
        return True, True, exact, float(exact)
    score = rouge_l(str(gt), str(pred)) if kind == "literal" else token_f1(str(gt), str(pred))
    return True, True, exact, 1.0 if exact else score


def _row(doc, section, role, entity, field, gt, pred, kind=None):
    kind = kind or KIND.get(field, "text")
    gp, pp, strict, score = compare(kind, gt, pred)
    thr = SOFT_LITERAL if kind == "literal" else SOFT_TEXT if kind == "text" else 1.0
    return dict(doc_id=doc, section=section, role=role, entity=entity, field=field, kind=kind,
                gt_present=gp, pred_present=pp, strict=bool(strict), soft=bool(strict or (gp and pp and score >= thr)),
                score=score, gt=str(gt)[:200], pred=str(pred)[:200])


def match_people(gt_list: List[dict], pred_list: List[dict], thr: float = NAME_THRESHOLD):
    """Atribuição ótima GT↔predição. → (pares, gt_sem_par, pred_sem_par)."""
    if not gt_list or not pred_list:
        return [], list(range(len(gt_list))), list(range(len(pred_list)))
    S = np.array([[name_similarity(g.get("nome") or "", p.get("nome") or "") for p in pred_list] for g in gt_list])
    ri, ci = linear_sum_assignment(-S)
    pairs = [(i, j, S[i, j]) for i, j in zip(ri, ci) if S[i, j] >= thr]
    mg, mp = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
    return pairs, [i for i in range(len(gt_list)) if i not in mg], [j for j in range(len(pred_list)) if j not in mp]


def evaluate_doc(doc_id: str, gt: dict, pred: Optional[dict]) -> Tuple[List[dict], List[dict]]:
    """→ (linhas de campos/entidades, confusões de papel)."""
    gt, pred = canonical.canonicalize(gt), canonical.canonicalize(pred or {})
    rows, confusions = [], []

    g_res, p_res = gt.get("resumo_processo") or {}, pred.get("resumo_processo") or {}
    rows.append(_row(doc_id, "resumo", None, None, "classificacao_crime",
                     g_res.get("classificacao_crime"), p_res.get("classificacao_crime")))

    if gt.get("inquerito") is not None:
        g_i, p_i = gt["inquerito"], pred.get("inquerito") or {}
        rows += [_row(doc_id, "inquerito", None, None, f, g_i.get(f), p_i.get(f)) for f in INQUERITO]

    unmatched_gt: Dict[str, list] = {}
    unmatched_pred: Dict[str, list] = {}
    for role in ROLES:
        g_list, p_list = gt.get(role) or [], pred.get(role) or []
        pairs, miss, extra = match_people(g_list, p_list)
        fields = PESSOA + (VITIMA if role == "vitimas" else [])
        for i, j, sim in pairs:
            g, p = g_list[i], p_list[j]
            name = g.get("nome")
            r = _row(doc_id, role, role, name, "__entidade__", g.get("nome"), p.get("nome"), kind="entity")
            r["soft"], r["score"] = True, sim
            rows.append(r)
            rows += [_row(doc_id, role, role, name, f, g.get(f), p.get(f)) for f in fields]
        for i in miss:
            rows.append(dict(doc_id=doc_id, section=role, role=role, entity=g_list[i].get("nome"), field="__entidade__",
                             kind="entity", gt_present=True, pred_present=False, strict=False, soft=False, score=0.0,
                             gt=str(g_list[i].get("nome")), pred="None"))
        for j in extra:
            rows.append(dict(doc_id=doc_id, section=role, role=role, entity=p_list[j].get("nome"), field="__entidade__",
                             kind="entity", gt_present=False, pred_present=True, strict=False, soft=False, score=0.0,
                             gt="None", pred=str(p_list[j].get("nome"))))
        unmatched_gt[role] = [g_list[i] for i in miss]
        unmatched_pred[role] = [p_list[j] for j in extra]

    # confusão de papel: GT sem par que aparece em OUTRO papel da predição
    used = set()
    for rg, people in unmatched_gt.items():
        for g in people:
            for rp, cands in unmatched_pred.items():
                if rp == rg:
                    continue
                hit = next((k for k, p in enumerate(cands) if (rp, k) not in used
                            and name_similarity(g.get("nome") or "", p.get("nome") or "") >= NAME_THRESHOLD), None)
                if hit is not None:
                    used.add((rp, hit))
                    confusions.append(dict(doc_id=doc_id, gt_role=rg, pred_role=rp, name=g.get("nome")))
                    break
    return rows, confusions


@dataclass
class Report:
    rows: pd.DataFrame
    confusions: pd.DataFrame
    failed_docs: List[str]

    def _status(self, mode: str) -> pd.DataFrame:
        d = self.rows
        both = d.gt_present & d.pred_present
        m = d[mode]
        return d.assign(tp=both & m, wrong=both & ~m, extra=d.pred_present & ~d.gt_present,
                        missing=d.gt_present & ~d.pred_present)

    @staticmethod
    def _prf(tp, wrong, extra, missing) -> Dict[str, float]:
        p = tp / (tp + wrong + extra) if (tp + wrong + extra) else 0.0
        r = tp / (tp + wrong + missing) if (tp + wrong + missing) else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "support": int(tp + wrong + missing)}

    def _agg(self, df: pd.DataFrame) -> Dict[str, float]:
        return self._prf(*(int(df[c].sum()) for c in ("tp", "wrong", "extra", "missing")))

    def overall(self, mode: str = "strict") -> Dict[str, float]:
        return self._agg(self._status(mode))

    def by_field(self, mode: str = "strict") -> pd.DataFrame:
        d = self._status(mode)
        out = {f: self._agg(g) for f, g in d.groupby("field")}
        return pd.DataFrame(out).T.sort_values("f1")

    def by_section(self, mode: str = "strict") -> pd.DataFrame:
        d = self._status(mode)
        return pd.DataFrame({s: self._agg(g) for s, g in d.groupby("section")}).T

    def macro_f1(self, mode: str = "strict") -> float:
        t = self.by_field(mode)
        return float(t[t.support > 0].f1.mean())

    def doc_scores(self, mode: str = "strict") -> pd.Series:
        """F1 micro por documento (insumo de bootstrap/testes pareados)."""
        d = self._status(mode)
        out = {}
        for doc, g in d.groupby("doc_id"):
            out[doc] = self._agg(g)["f1"]
        return pd.Series(out, name=f"f1_{mode}").sort_index()

    def role_confusion(self) -> pd.DataFrame:
        if self.confusions.empty:
            return pd.DataFrame()
        return pd.crosstab(self.confusions.gt_role, self.confusions.pred_role)


def evaluate(gt: Dict[str, dict], preds: Dict[str, dict], ids: Optional[List[str]] = None) -> Report:
    """`preds[doc]` = dict canônico (ou None). Docs sem predição valem como vazios."""
    ids = ids or sorted(gt)
    rows, conf, failed = [], [], []
    for doc in ids:
        p = preds.get(doc)
        if p is None:
            failed.append(doc)
        r, c = evaluate_doc(doc, gt[doc], p)
        rows += r
        conf += c
    df = pd.DataFrame(rows, columns=COLS)
    return Report(df, pd.DataFrame(conf, columns=["doc_id", "gt_role", "pred_role", "name"]), failed)


def doc_f1(doc_id: str, gt: dict, pred: Optional[dict], mode: str = "soft") -> float:
    """F1 micro de um documento (usado como recompensa no RL)."""
    rows, _ = evaluate_doc(doc_id, gt, pred)
    rep = Report(pd.DataFrame(rows, columns=COLS), pd.DataFrame(), [])
    return rep.overall(mode)["f1"]
