# mpsp-extraction

Extração estruturada de promoções de arquivamento de inquéritos policiais do Ministério Público do Estado de São Paulo (MPSP). O sistema converte o texto livre de cada promoção em um JSON canônico, usando Pequenos Modelos de Linguagem (SLMs, de 7 a 8 bilhões de parâmetros) executados localmente.

Os documentos são sigilosos e nenhum dado sai da máquina em qualquer etapa de extração ou avaliação.

## Escopo

- Quatro modelos: `qwen2.5-7b`, `llama3.1-8b`, `granite3.3-8b` e `falcon3-7b`.
- 44 execuções avaliadas: 4 modelos × 11 condições, com 5 partições (k-fold estratificado, semente 42).

## Privacidade

- `data/` (Por clásulas de privacidade) (GT e textos dos processos) não faz parte do repositório. O `.gitignore` também exclui, por precaução, os relatórios por documento e por campo (`reports/*/rows.csv`, `by_field_*.csv`, `doc_scores_*.csv`, `role_confusion.csv`).
- Os identificadores numéricos dos documentos (`splits.json`) e os agregados (`summary.json`) podem ser publicados.
- `exemplo_saida.json` é uma saída ilustrativa do formato canônico, com conteúdo fictício. Não é documento do MPSP.
- A rede é usada só para baixar arquivos públicos: os pesos dos modelos (`mpsp download`) e o corpus LeNER-Br, em `mpsp bench` e no treino `two_phase_ner` quando o cache em `data/benchmarks/` ainda não existe. Nenhum dado do projeto é enviado. Os demais comandos rodam com o acesso ao Hugging Face bloqueado (ver `mpsp/offline.py`).

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # CLI, avaliação e testes
pip install -e ".[slm]"          # + treino e inferência de SLMs (torch, transformers, peft, bitsandbytes)
pip install -e ".[vllm]"         # opcional: engine vLLM (lote contínuo)
```

`requirements.txt` e `requirements-slm.txt` trazem as mesmas dependências em forma de lista. O ambiente de referência é uma GPU NVIDIA A40 (46 GB). Treino em 4 bits e AdamW de 8 bits exigem CUDA.

No Windows, se a saída no console for corrompida, defina antes de rodar scripts diretamente:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## Dados esperados

```
data/
├── labels/ground_truth.json            # {doc_id: InqueritoTotal}
└── scrappings/treino+teste/
    └── <doc_id>.txt                    # texto do processo
splits.json                             # partições (gerado por `mpsp split`)
```

O schema canônico está em `mpsp/schemas.py` (`InqueritoTotal`). Ele tem cinco seções: `resumo_processo` (pessoas por papel e classificação do crime), `inquerito` (resultado, datas, local, arma, perícia, razão do arquivamento), `vitimas`, `suspeitos` e `testemunhas`. Formatos legados são normalizados por `mpsp/canonical.py`.

## Uso

Todos os comandos são chamados por `mpsp <comando>` ou `python -m mpsp <comando>`.

### Preparação

```bash
mpsp models                                   # registro de SLMs (configs/models.yaml)
mpsp download --model qwen2.5-7b              # só pesos públicos
mpsp validate-gt --gt data/labels/ground_truth.json \
                 --docs "data/scrappings/treino+teste" --splits splits.json
mpsp split --gt data/labels/ground_truth.json --docs "data/scrappings/treino+teste" \
           --k 5 --seed 42 --val-frac 0.15 --out splits.json
```

`validate-gt` confere o GT contra o schema, os textos em disco e as partições. Use-o antes de qualquer execução.

### Extração e avaliação

```bash
# Extrai por partição (k-fold) com few-shot de 3 exemplos recuperados por TF-IDF
mpsp run --gt data/labels/ground_truth.json --docs "data/scrappings/treino+teste" \
         --splits splits.json --model qwen2.5-7b --topology T1 --shots 3 \
         --out runs/qwen_T1_3shot.jsonl

# Métricas, IC95%, custo e validade
mpsp eval --gt data/labels/ground_truth.json --pred runs/qwen_T1_3shot.jsonl \
          --docs "data/scrappings/treino+teste" --out reports/qwen2.5-7b_T1_3shot

# Teste pareado entre execuções (bootstrap, Wilcoxon e Holm)
mpsp compare --gt data/labels/ground_truth.json \
             --pred runs/a.jsonl runs/b.jsonl --mode soft

# Tabela de execuções avaliadas
mpsp leaderboard --reports reports --by method --max-invalid 0.2
```

Opções úteis de `run`:

- `--adapter caminho/{fold}`: usa um adapter LoRA ou um modelo completo treinado por partição.
- `--engine vllm`: lote contínuo (requer o extra `vllm`).
- `--shard i/n` e `--fold f`: fatia o trabalho para rodar vários processos em paralelo. `eval` junta os arquivos automaticamente.
- `--batch-size`, `--gen-batch`, `--gpu-util`: controlam o uso de memória.

### Treino de SLMs

```bash
python -m mpsp.slm.train --method lora --model falcon3-7b \
  --gt data/labels/ground_truth.json --docs "data/scrappings/treino+teste" \
  --splits splits.json --fold 0 --topology T1 --out ckpts/falcon3-7b/lora/T1/fold0
```

Métodos disponíveis em `--method`, descritos em `mpsp/slm/methods.py`:

| Método | Descrição |
|---|---|
| `qlora` | LoRA sobre a base quantizada em 4 bits (NF4) |
| `lora` | LoRA sobre a base em bf16 |
| `full` | todos os pesos, com embeddings congeladas e AdamW de 8 bits paginado |
| `frozen` | backbone congelado; treina só a norma final e o `lm_head` (`--unfreeze-last-k` libera blocos finais) |
| `two_phase` | fase 1 só o `lm_head`; fase 2 LoRA e `lm_head` |
| `two_phase_ner` | fase 1 LoRA em NER jurídico público (LeNER-Br); fase 2 o mesmo LoRA no MPSP |

`--fold -1` treina com todo o GT, para o modelo final. Exemplos que excedem `--max-seq-len` (padrão 4096) são descartados, nunca truncados. `--dry-run` monta os dados e estima a memória sem carregar o modelo.

### Plano de experimentos

O plano inteiro fica em `configs/experiments.yaml`. O escopo declarado no arquivo é o que está em `reports/`.

```bash
mpsp plan     --config configs/experiments.yaml --estimate   # jobs e estimativa de tempo
mpsp schedule --config configs/experiments.yaml --dry-run    # ensaio
mpsp schedule --config configs/experiments.yaml --budget-hours 24
mpsp status   --config configs/experiments.yaml              # progresso e ETA
```

O agendador roda vários jobs na GPU respeitando a VRAM e as dependências. É retomável: cada job concluído grava um marcador em `logs/lab/done/`, e reexecutar o plano pula o que já terminou. As estimativas de tempo são premissas até serem substituídas pelos valores medidos em `time_model:`.

Se um job for interrompido por falta de memória, `scripts/redo_oom.py` lista os arquivos afetados. Com `--apply`, remove os marcadores para refazer os jobs.

Antes dos jobs longos, rode `bash scripts/preflight.sh`. Ele roda os testes, lista os modelos, valida o GT, cria as partições se necessário e faz um ensaio de treino.

### Benchmarks externos

```bash
mpsp bench --name lener_br --model qwen2.5-7b --pattern both --n 100 --shots 0
```

Usa o mesmo backend, a mesma quantidade de amostras e o mesmo padrão de prompt da extração do MPSP. Baixa o corpus LeNER-Br na primeira execução.

### Limpeza

`mpsp cleanup --dir <checkpoint> --keep-info train_info.json` apaga o modelo completo depois de avaliado e mantém o `train_info.json`.

## Configuração

- `configs/experiments.yaml`: plano único de execuções, caminhos, recursos da GPU e cronograma. `paths.*` são relativos à raiz do repositório.
- `configs/models.yaml`: registro dos SLMs, com `hf_id`, parâmetros, janela de contexto (`ctx_tokens`, usada para detectar estouro), licença e `gated` (exige aceitar a licença e definir `HF_TOKEN`, só para o download).
- `python` e `python_vllm` em `experiments.yaml` apontam para o interpretador usado nos jobs. Ajuste-os para o seu ambiente.

## Testes

```bash
pytest -q
```

Os testes não precisam de GPU nem de dados sigilosos. Testes de treino com torch são pulados automaticamente se `torch`, `transformers` ou `peft` não estiverem instalados. Com dependências mínimas (sem torch), a suíte passa com 93 testes e 2 pulados.

`tests/test_train_torch.py` treina um Llama minúsculo de pesos aleatórios em CPU. Não cobre quantização 4 bits, AdamW de 8 bits nem bf16, que dependem de GPU. Esses são checados pelo preflight e pelo smoke test.

`tests/test_all_plan.py` confere que o plano em `configs/experiments.yaml` corresponde ao escopo realmente executado.

## Resultados

Os resultados completos estão em `reports/` (uma pasta por execução, com `summary.json`) e na monografia. Métricas principais por execução:

- F1 `strict`: igualdade normalizada.
- F1 `soft`: token-F1 ≥ 0,5 para campos textuais, ROUGE-L ≥ 0,8 para o texto literal da razão de arquivamento, diferença ≤ 0,01 para latitude e longitude, igualdade para categóricos, booleanos, datas e horas. Pessoas são pareadas por atribuição ótima sobre similaridade de nomes (limiar 0,5).
- `ok_rate` (saídas válidas): fração de documentos cuja saída final passa na validação do schema Pydantic, após no máximo uma retentativa com o erro no prompt. Validade não garante que os valores estejam corretos.



```bash
python mpsp-analysis/secao_3_4.py --reports reports --fields per_field_top5.csv --out figs/secao_3_4
```

As figuras e tabelas já geradas estão em `figs/`.

## Estrutura do repositório

```
configs/                 experiments.yaml (plano) e models.yaml (registro de SLMs)
mpsp/                    CLI, extração, avaliação, estatística, agendador
  backends/              backend local (HF) e vLLM, geração em lote, validação e retentativa
  benchmarks/            benchmark NER (LeNER-Br)
  slm/                   métodos de adaptação e treino (qlora, lora, full, frozen, two_phase, two_phase_ner)
mpsp-analysis/           scripts de consolidação, gráficos e tabelas da monografia
reports/                 resultados agregados de cada execução
figs/                    figuras e tabelas geradas
logs/                    saídas de execução (logs/lab/ guarda o estado do agendador)
scripts/                 preflight.sh e redo_oom.py
tests/                   testes (pytest)
splits.json              partições k-fold dos documentos (identificadores)
per_field_top5.csv       F1 soft por campo, média das 5 melhores execuções
leaderboard_full.csv     tabela completa das execuções
```

## Limitações conhecidas

- Uma semente, uma partição e 95 documentos. As comparações entre métodos são indicativas, não definitivas.
- Só a topologia T1 foi avaliada. As topologias de múltiplas chamadas (T2 a T5) estão implementadas, mas não executadas.
- A decodificação restrita ao schema (`--constrained`, backend HF) exige o pacote opcional `lm-format-enforcer`, que não está nos requisitos padrão. Nenhum `summary.json` em `reports/` registra o uso dessa opção.
- Os tempos de execução dependem da GPU. As estimativas do agendador são premissas até serem medidas.
