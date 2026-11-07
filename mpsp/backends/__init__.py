"""Registro de modelos (configs/models.yaml) → backend local. Nenhum dado sai da máquina."""
import json
import os
import sys
from pathlib import Path

import yaml

from .base import Backend, Generation, MockBackend, parse_json_model  # noqa: F401


def _models_config() -> Path:
    """Localiza `configs/models.yaml`: env, raiz do repositório, cwd ou prefixo de instalação."""
    env = os.environ.get("MPSP_CONFIG_DIR")
    if env:
        return Path(env) / "models.yaml"
    for base in (Path(__file__).resolve().parents[2], Path.cwd(), Path(sys.prefix)):
        p = base / "configs" / "models.yaml"
        if p.exists():
            return p
    return Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


CONFIG = _models_config()


def load_registry(path: "Path | None" = None) -> dict:
    with open(path or _models_config(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_info(name: str) -> dict:
    reg = load_registry()["slm"]
    if name not in reg:
        raise KeyError(f"Modelo '{name}' não está em configs/models.yaml (disponíveis: {', '.join(reg)})")
    return {"name": name, **reg[name]}


def resolve_adapter(adapter: str, engine: str, cfg: dict, overrides: dict):
    """`adapter` pode ser um adapter LoRA ou um modelo completo (full/frozen). Ajusta `cfg` e devolve o adapter (ou None).
    A quantização da base segue a do treino (train_info.json), salvo se o chamador a definir."""
    d = Path(adapter)
    if not d.is_dir():
        raise FileNotFoundError(f"adapter/modelo não encontrado: {adapter}")
    meta = json.loads((d / "train_info.json").read_text(encoding="utf-8")) if (d / "train_info.json").exists() else {}
    if "load_4bit" in meta and "load_4bit" not in overrides:
        cfg["load_4bit"] = meta["load_4bit"]
    if (d / "adapter_config.json").exists():
        if engine == "vllm" and meta.get("hf_only"):
            raise ValueError(f"{adapter}: adapter com módulos extras (two_phase) não roda no vLLM; use --engine hf")
        return str(d)
    if (d / "config.json").exists():          # modelo completo: vira o próprio hf_id
        cfg["hf_id"] = str(d)
        return None
    raise FileNotFoundError(f"{adapter}: sem adapter_config.json nem config.json")


def get_backend(name: str, adapter: str = None, **overrides) -> Backend:
    engine = overrides.pop("engine", "hf")
    if name == "mock":            # testes e dry-run
        return MockBackend()
    cfg = {**model_info(name), **overrides}
    cfg["adapter"] = resolve_adapter(adapter, engine, cfg, overrides) if adapter else None
    if engine == "vllm":          # lote contínuo; pip install vllm
        from .vllm_backend import VLLMBackend
        return VLLMBackend(**cfg)
    from .local import LocalBackend
    return LocalBackend(**cfg)
