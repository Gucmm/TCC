"""Métodos de adaptação dos SLMs (o que treinar, com que precisão, em quantas fases).

Todos usam os MESMOS dados, folds, prompts de treino e critério de parada; só muda a forma de adaptar:

  qlora          LoRA sobre a base quantizada em 4 bits (NF4)
  lora           LoRA sobre a base em bf16 (isola o efeito da quantização)
  full           todos os pesos (embeddings congeladas), AdamW de 8 bits paginado, parada antecipada
  frozen         backbone congelado: só norma final + lm_head (opcional: últimos K blocos)
  two_phase      bifásico LP-FT: fase 1 só o lm_head (backbone congelado); fase 2 LoRA + lm_head
  two_phase_ner  bifásico por dados: fase 1 LoRA em NER jurídico público (LeNER-Br); fase 2 mesmo LoRA no MPSP
"""
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Phase:
    name: str
    trainable: str            # "lora" | "all" | "head"
    data: str                 # "mpsp" | "ner"
    lr: float
    epochs: int
    patience: int = 3


@dataclass(frozen=True)
class Method:
    name: str
    desc: str
    load_4bit: bool
    optim: str
    full_save: bool           # salva o modelo inteiro (e não só o adapter)
    hf_only: bool             # adapter incompatível com vLLM
    phases: Tuple[Phase, ...]


METHODS: Dict[str, Method] = {m.name: m for m in (
    Method("qlora", "LoRA sobre base 4 bits (NF4)", True, "adamw_torch", False, False,
           (Phase("sft", "lora", "mpsp", 2e-4, 20),)),
    Method("lora", "LoRA sobre base bf16", False, "adamw_torch", False, False,
           (Phase("sft", "lora", "mpsp", 2e-4, 20),)),
    Method("full", "todos os pesos, AdamW 8 bits paginado", False, "paged_adamw_8bit", True, False,
           (Phase("full", "all", "mpsp", 1e-5, 10, patience=2),)),
    Method("frozen", "backbone congelado (norma final + lm_head)", False, "adamw_torch", True, False,
           (Phase("frozen", "head", "mpsp", 1e-4, 20),)),
    Method("two_phase", "bifásico LP-FT: lm_head, depois LoRA + lm_head", False, "adamw_torch", False, True,
           (Phase("head", "head", "mpsp", 1e-4, 3, patience=2), Phase("lora", "lora", "mpsp", 1e-4, 20))),
    Method("two_phase_ner", "bifásico por dados: NER jurídico, depois MPSP (mesmo LoRA)", False, "adamw_torch", False, False,
           (Phase("ner", "lora", "ner", 2e-4, 3, patience=2), Phase("mpsp", "lora", "mpsp", 1e-4, 20))),
)}


def estimate_train_gb(params_b: float, method: Method, max_seq_len: int = 4096, batch: int = 1) -> float:
    """Pico aproximado de VRAM (GB). Heurística (±30%): serve para agendar jobs; o pico real é gravado em
    train_info.json. Ativações+logits com checkpointing ≈ 2,5 MB por token."""
    act = 2.5e-3 * max_seq_len * batch
    if method.name == "full":                       # pesos + grads bf16; estados de 8 bits ficam paginados na RAM
        return 4.0 * params_b * 0.97 + act
    if method.name == "frozen":                     # sem grads no backbone
        return 2.0 * params_b + 2.0 + 0.5 * act
    if method.load_4bit:
        return 0.6 * params_b + 1.5 + act
    return 2.0 * params_b + 1.5 + act               # lora, two_phase*: base bf16 + adapter


def estimate_infer_gb(params_b: float, load_4bit: bool, engine: str = "hf", gpu_util: float = 0.85,
                      total_gb: float = 48.0) -> float:
    if engine == "vllm":                            # o vLLM reserva a fração pedida da GPU
        return gpu_util * total_gb
    return (0.6 if load_4bit else 2.0) * params_b + 4.0
