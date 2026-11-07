"""Garantia de execução offline.

Os documentos do MPSP são sigilosos: nada deve sair da máquina. Há etapas que acessam a rede, e
todas só baixam arquivos públicos, sem enviar dados do projeto:

- `python -m mpsp download`: pesos dos modelos (Hugging Face);
- `python -m mpsp bench` e o treino `two_phase_ner` (na primeira execução, sem cache): corpus público LeNER-Br (via `urllib`).

Todos os demais comandos rodam com o acesso do Hugging Face bloqueado. Precisa ser chamado ANTES de
importar transformers/huggingface_hub.
"""
import os


def enforce_offline() -> None:
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"
    disable_telemetry()


def disable_telemetry() -> None:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
