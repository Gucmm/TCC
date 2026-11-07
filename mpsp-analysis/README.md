# mpsp-extraction

Extração estruturada de **promoções de arquivamento de inquéritos policiais do MPSP**
(Ministério Público do Estado de São Paulo) com **SLMs locais** (7–8B), transformando
texto jurídico não estruturado em um **JSON canônico**.

O projeto compara quatro famílias de baselines — **zero-shot**, **few-shot** e
**fine-tuning** (QLoRA, LoRA, full, frozen e os dois bifásicos `two_phase`/`two_phase_ner`).
Tudo roda offline: os documentos não saem da máquina.

> **Privacidade.** Os documentos do MPSP são sigilosos. O repositório **não** deve conter
> `data/` nem `reports/*/rows.csv` (que carregam nomes e trechos reais). Os identificadores
> numéricos dos documentos (`splits.json` e `reports/*/doc_scores_*.csv`) e os números
> agregados (`by_field_*.csv`, `summary.json`) podem acompanhar o pacote.

---

## Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -e ".[dev]"      # CLI + avaliação + testes
pip install -e ".[slm]"     # + treino/inferência de SLMs (torch/transformers/peft/...)
pip install -e ".[vllm]"    # opcional: engine de lote contínuo
```

Os extras `dev`/`slm`/`vllm` são definidos no `pyproject.toml`. Ambiente de referência:
GPU A40 (46 GB). Para outra máquina, use as faixas de `requirements.txt`/`requirements-slm.txt`.

### Windows: acentos no console

A CLI grava UTF-8 e reconfigura `stdout/stderr` para UTF-8. Em consoles com code page
não-UTF-8 (ex.: cp932/cp1252), se for chamar scripts diretamente:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

## Estrutura de dados esperada

```
data/
├── labels/ground_truth.json          # {doc_id: InqueritoTotal}
└── scrappings/treino+teste/
    ├── <doc_id>.txt                  # texto do processo
    └── <qualquer>/<doc_id>/texto_processo.txt
splits.json                           # k=5, seed=42, val_frac=0.15
```

Os dados **não** acompanham o repositório. Sem eles, `validate-gt`, `split`, `run` e
`eval` não têm o que fazer.

### Schema canônico

Fonte única de verdade: `mpsp/schemas.py` (`InqueritoTotal`).

| Seção | Conteúdo |
|---|---|
| `resumo_processo` | pessoas por papel (vítimas, suspeitos/investigados, testemunhas) + classificação do crime (Homicídio, Latrocínio, Morte por causas naturais) |
| `inquerito` | resultado, feminicídio, datas/hora, delegacia, local (tipo, detalhe, lat/long), município, arma, bem roubado, natureza da autoria, perícia, flagrante, `razao_arquivamento` (trecho literal) |
| `vitimas[]` | `Pessoa` + armada, arma, faleceu, causa jurídica da morte, relação com o autor |
| `suspeitos[]` / `testemunhas[]` | `Pessoa`: nome, cor, sexo, policial, corporação, profissão, escolaridade, nacionalidade, idade, antecedentes |

Qualquer formato legado é normalizado por `mpsp/canonical.py`.

---

## Uso

```bash
# Registro de modelos locais
mpsp models

# Baixar pesos (só arquivos públicos; nenhum dado do projeto é enviado)
mpsp download --model qwen2.5-7b

# Validar GT × textos × partições
mpsp validate-gt --gt data/labels/ground_truth.json --docs "data/scrappings/treino+teste" --splits splits.json

# Extrair (k-fold) e avaliar
mpsp run  --gt data/labels/ground_truth.json --docs "data/scrappings/treino+teste" \
          --splits splits.json --model qwen2.5-7b --topology T1 --out runs/qwen.jsonl
mpsp eval --gt data/labels/ground_truth.json --pred runs/qwen.jsonl --docs "data/scrappings/treino+teste" --out reports/qwen

# Comparar execuções (bootstrap pareado + Wilcoxon + Holm)
mpsp compare --gt data/labels/ground_truth.json --pred runs/a.jsonl runs/b.jsonl

# Plano de experimentos (config única) e agendamento na GPU
mpsp plan     --config configs/experiments.yaml --estimate
mpsp schedule --config configs/experiments.yaml --dry-run
mpsp status   --config configs/experiments.yaml

# Leaderboard agrupado por tipo de treino (flags de saída inválida)
mpsp leaderboard --reports reports --by method
```

`configs/experiments.yaml` é a **configuração única** dos experimentos; `configs/models.yaml`
é o registro de SLMs.

---

## Testes

```bash
pytest -q
```

`tests/test_all_plan.py` valida a config única contra o escopo que realmente foi
executado (`reports/`). Testes de treino com torch são pulados automaticamente se
torch/transformers/peft não estiverem instalados.

---

## Resultados

**95 documentos, topologia T1, 5-fold estratificado (classe × tercil de tamanho), seed 42,
uma única partição.** F1 `soft` = token-F1/ROUGE-L ≥ limiar; `strict` = igualdade normalizada.
`ok_rate` = fração de saídas que validam no schema canônico.

### Leaderboard (topo, por F1 soft)

| Execução | F1 strict | F1 soft | F1/doc soft (IC95%) | válidas | s/doc |
|---|---:|---:|---:|---:|---:|
| llama3.1-8b qlora lr1e-4 | 0.730 | **0.789** | 0.770 [0.740, 0.797] | 0.99 | 36.2 |
| llama3.1-8b full | 0.736 | 0.787 | 0.761 [0.728, 0.791] | 0.98 | 26.6 |
| falcon3-7b two_phase_ner | 0.732 | 0.786 | 0.752 [0.715, 0.786] | 0.97 | 46.1 |
| qwen2.5-7b full | 0.732 | 0.785 | 0.758 [0.723, 0.789] | 0.98 | 12.5 |
| qwen2.5-7b two_phase_ner | 0.733 | 0.785 | 0.764 [0.735, 0.791] | 0.99 | 14.6 |
| llama3.1-8b two_phase_ner | 0.728 | 0.785 | 0.761 [0.726, 0.791] | 0.98 | 103.3 |
| qwen2.5-7b lora | 0.731 | 0.784 | 0.761 [0.729, 0.791] | 0.98 | 39.5 |
| llama3.1-8b two_phase_ner lr2e-4 | 0.732 | 0.783 | 0.764 [0.734, 0.790] | 0.99 | 39.4 |
| falcon3-7b two_phase_ner lr2e-4 | 0.725 | 0.782 | 0.753 [0.718, 0.786] | 0.97 | 84.6 |
| llama3.1-8b qlora | 0.720 | 0.780 | 0.761 [0.731, 0.787] | 0.99 | 22.1 |

A comparação pareada completa (contra `llama3.1-8b frozen`) está em `reports/compare_all.txt`.

### O que os resultados dizem (e o que não dizem)

- **Fine-tuning ≫ zero-shot.** Os métodos adaptados ficam em F1 soft ~0.72–0.79 com
  `ok_rate` >0.95; o zero-shot fica entre 0.17 e 0.55.
- **O topo está empatado.** As ~10 melhores execuções estão em 0.780–0.789 de F1 soft, com
  IC95% de F1/doc amplamente sobrepostos (ex.: 0.789 vs 0.780 com ICs [0.740, 0.797] e
  [0.731, 0.787]). **Não há teste de significância entre os líderes** — o `compare` só testa
  contra o baseline fraco `frozen`. Ordene o topo com cautela.
- **Uma única semente.** `seed=42`, uma partição, sem repetição. O IC95% (bootstrap sobre os
  95 docs) **não captura** a variância de treino/partição, então a incerteza dos métodos
  finetunados está subestimada.

### Saídas inválidas (confundidor de formato)

**19 das 44 execuções têm mais de 20% de respostas fora do schema.** O pior caso é
`falcon3-7b full` (78,9% inválidas, F1 soft 0.358); em seguida vêm `falcon3-7b 1shot`
(60,0%), `llama3.1-8b 3shot` (56,8%) e `granite3.3-8b lora` (52,6%).

Como um documento inválido entra na avaliação como predição vazia (F1 0), o F1 baixo dessas
execuções reflete sobretudo **falha de formato**, não qualidade de conteúdo. Compare sempre
com a coluna de saídas inválidas antes de concluir que um método é pior. Use
`mpsp leaderboard --max-invalid 0.2` para listar só as execuções confiáveis.

### Few-shot

O few-shot **melhora** o F1 sobre o zero-shot na maioria dos casos (falcon: 0.173 → 0.437 →
0.605; qwen: 0.554 → 0.632 → 0.679), mas a um custo alto: a taxa de saídas válidas cai para
0.40–0.74 e a latência sobe para 96–238 s/doc (vs 39–58 do zero-shot). Ou seja, não é o F1
que piora, e sim a **estabilidade** e a **eficiência**.

### Controles de learning rate

Dentro do ruído para llama/falcon (ex.: `llama qlora_lr1e-4` 0.789 ≈ `llama qlora` 0.780),
mas prejudiciais para granite/qwen: `granite qlora_lr1e-4` cai para 0.663 (ok 0.67) e
`qwen two_phase_ner_lr2e-4` para 0.680 (ok 0.72).

### Campos difíceis

Mesmo os melhores modelos zeram em `nacionalidade`, `idade`, `arma_da_vítima`,
`causa_juridica_da_morte` e `antecedentes_criminais` (F1 ≈ 0). `cor` e `escolaridade` têm
suporte 0 no GT — não são avaliáveis neste corpus.

### Escopo executado

4 modelos × 11 condições = **44 execuções**: zero-shot, few-shot (1/3-shot), qlora, lora,
full, frozen, two_phase, two_phase_ner e dois controles de lr.

**Não executado:** T2–T5, oracle (E2), benchmark LeNER-Br (E5), corpus completo (E7), curva
de aprendizado (E4i) e múltiplas sementes.

---

## Estrutura do repositório

```
configs/            experiments.yaml (plano) + models.yaml (registro de SLMs)
mpsp/               CLI, extração, avaliação, agendador, backends, treino (slm/)
scripts/            preflight.sh
tests/              testes (pytest)
reports/            resultados das 44 execuções (summary.json, by_field, doc_scores, ...)
logs/               logs de execução por diretório de estado (lab, lab_lora, lab_outros, ...)
```
