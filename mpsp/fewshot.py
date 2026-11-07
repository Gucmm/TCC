"""Seleção de exemplos few-shot por similaridade (TF-IDF, determinística, sem downloads)."""
from typing import Dict, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRetriever:
    def __init__(self, pool: Dict[str, str]):
        self.ids = list(pool)
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), strip_accents="unicode", lowercase=True)
        self.X = self.vec.fit_transform([pool[i] for i in self.ids])

    def query(self, text: str, k: int, exclude: Sequence[str] = ()) -> List[str]:
        sims = (self.X @ self.vec.transform([text]).T).toarray().ravel()
        order = sims.argsort()[::-1]
        return [self.ids[i] for i in order if self.ids[i] not in set(exclude)][:k]


def select_shots(retriever: TfidfRetriever, texts: Dict[str, str], gt: Dict[str, dict],
                 doc_id: str, text: str, k: int, max_chars: int = 60000) -> List[dict]:
    """k exemplos mais parecidos, respeitando um orçamento de caracteres. O mais
    parecido fica por último (mais próximo do documento a processar)."""
    if k <= 0:
        return []
    shots, used = [], 0
    for cid in retriever.query(text, k + 5, exclude=[doc_id]):
        if len(shots) == k:
            break
        if used + len(texts[cid]) > max_chars:
            continue
        used += len(texts[cid])
        shots.append({"doc_id": cid, "text": texts[cid], "canon": gt[cid]})
    return shots[::-1]
