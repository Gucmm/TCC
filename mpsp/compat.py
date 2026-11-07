"""Compatibilidade entre versões do transformers."""
import re
from importlib import metadata


def dtype_kw(dtype) -> dict:
    """`dtype=` a partir do transformers 4.56; antes disso, `torch_dtype=`."""
    try:
        m = re.match(r"(\d+)\.(\d+)", metadata.version("transformers"))
        new = (int(m.group(1)), int(m.group(2))) >= (4, 56)
    except Exception:  # noqa: BLE001
        new = False
    return {"dtype": dtype} if new else {"torch_dtype": dtype}
