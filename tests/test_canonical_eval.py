import pytest

from mpsp import canonical, evaluate as ev
from mpsp.canonical import canonicalize
from mpsp.schemas import InqueritoTotal


def base(**kw):
    d = {"resumo_processo": {"pessoas_envolvidas": {}, "classificacao_crime": "Homicídio"},
         "inquerito": {}, "vitimas": [], "suspeitos": [], "testemunhas": []}
    d.update(kw)
    return d


@pytest.mark.parametrize("raw", [
    {"vitimas": [{"nome": "A"}], "suspeitos": [{"nome": "B"}]},                              # listas diretas
    {"vítimas": {"vitimas": [{"nome": "A"}]}, "suspeitos": {"Suspeitos": [{"nome": "B"}]}},  # wrappers legados
    {"status": "success", "result": {"vítimas": {"vitimas": [{"nome": "A"}]}, "suspeitos": {"suspeitos": [{"nome": "B"}]}}},
])
def test_canonicalize_preserva_pessoas_em_todos_os_formatos(raw):
    c = canonicalize(raw)
    assert len(c["vitimas"]) == 1 and len(c["suspeitos"]) == 1 and c["testemunhas"] == []


def test_validate_roundtrip():
    m = canonical.validate(base(vitimas=[{"nome": "A", "faleceu": True}]))
    assert isinstance(m, InqueritoTotal) and canonical.dump(m)["vitimas"][0]["faleceu"] is True


def test_predicao_perfeita():
    g = base(vitimas=[{"nome": "Maria da Silva", "faleceu": True}], inquerito={"data_ocorrencia": "15/03/2024"})
    rep = ev.evaluate({"d": g}, {"d": g})
    assert rep.overall("strict")["f1"] == 1.0 and rep.macro_f1("soft") == 1.0


def test_predicao_ausente_penaliza_recall():
    g = base(vitimas=[{"nome": "Maria"}])
    rep = ev.evaluate({"d": g}, {})
    assert rep.failed_docs == ["d"] and rep.overall("soft")["recall"] == 0.0


def test_verdadeiro_negativo_nao_entra_no_f1():
    g = base()   # nenhum campo preenchido além da classificação
    rep = ev.evaluate({"d": g}, {"d": g})
    assert rep.overall("strict")["support"] == 1       # só classificacao_crime
    tn = rep.rows.query("field == 'é_feminicidio'").iloc[0]
    assert not tn.gt_present and not tn.pred_present


def test_hungarian_vence_o_pareamento_guloso():
    # guloso casaria 'Ana Maria Souza' com 'Ana Maria' (primeiro) e deixaria 'Ana Souza' sem par
    g, p = [{"nome": "Ana Maria"}, {"nome": "Ana Maria Souza"}], [{"nome": "Ana Maria Souza"}, {"nome": "Ana Maria"}]
    pairs, miss, extra = ev.match_people(g, p)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)] and not miss and not extra


def test_confusao_de_papel_detectada():
    gt = base(suspeitos=[{"nome": "João Carlos Pereira"}])
    pred = base(testemunhas=[{"nome": "João C. Pereira"}])
    rep = ev.evaluate({"d": gt}, {"d": pred})
    assert rep.role_confusion().loc["suspeitos", "testemunhas"] == 1
    assert rep.by_section("soft").loc["suspeitos", "recall"] == 0.0


def test_valor_errado_conta_como_fp_e_fn():
    gt = base(inquerito={"municipio": "Campinas"})
    pred = base(inquerito={"municipio": "Santos"})
    d = ev.evaluate({"d": gt}, {"d": pred}).by_field("strict")
    assert d.loc["municipio", "precision"] == 0.0 and d.loc["municipio", "recall"] == 0.0


def test_inquerito_so_e_pontuado_se_rotulado():
    gt = base(inquerito=None)
    pred = base(inquerito={"municipio": "Santos", "razao_arquivamento": "x"})
    rep = ev.evaluate({"d": gt}, {"d": pred})
    assert not (rep.rows.section == "inquerito").any()


def test_normalizacao_de_datas_e_literal():
    assert ev.compare("date", "15/03/2024", "2024-03-15")[2]
    assert ev.compare("time", "14h30", "14:30")[2]
    _, _, strict, score = ev.compare("literal", "ausência de provas de autoria", "ausência de provas")
    assert not strict and 0.5 < score < 1.0
