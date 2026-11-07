"""Normalização e similaridade de texto (determinística, sem embeddings)."""
import re
import unicodedata
from collections import Counter
from typing import List

NULLS = {"", "nao informado", "nao informada", "ni", "n/i", "sem informacao", "ignorado", "null", "none"}
PARTICLES = {"de", "da", "do", "dos", "das", "e"}


def norm(s) -> str:
    """minúsculas, sem acentos, pontuação → espaço, espaços colapsados."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def tokens(s) -> List[str]:
    return norm(s).split()


def is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, dict)):
        return not v
    return isinstance(v, str) and norm(v) in NULLS


def token_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(tb), common / len(ta)
    return 2 * p * r / (p + r)


def rouge_l(a: str, b: str) -> float:
    """F1 da maior subsequência comum de tokens (campos literais)."""
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    prev = [0] * (len(tb) + 1)
    for x in ta:
        cur = [0]
        for j, y in enumerate(tb, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(tb), lcs / len(ta)
    return 2 * p * r / (p + r)


def name_similarity(a: str, b: str) -> float:
    """F1 de tokens ignorando partículas; inicial ('C.') casa com nome que começa por ela."""
    ta = [t for t in tokens(a) if t not in PARTICLES]
    tb = [t for t in tokens(b) if t not in PARTICLES]
    if not ta or not tb:
        return 0.0
    used, common = set(), 0
    for x in ta:
        for j, y in enumerate(tb):
            if j not in used and (x == y or (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y))):
                used.add(j)
                common += 1
                break
    if common == 0:
        return 0.0
    p, r = common / len(tb), common / len(ta)
    return 2 * p * r / (p + r)


_DATE = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


def norm_date(s) -> str:
    s = str(s)
    if m := _ISO.search(s):
        y, mo, d = m.groups()
    elif m := _DATE.search(s):
        d, mo, y = m.groups()
        y = ("20" + y) if len(y) == 2 else y
    else:
        return norm(s)
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def norm_time(s) -> str:
    m = re.search(r"(\d{1,2})\s*(?:[:h]\s*(\d{2})?)", str(s).lower())
    return f"{int(m.group(1)):02d}:{m.group(2) or '00'}" if m else norm(s)
