"""Compatibilidade: equivale a `python -m mpsp.slm.train --method qlora ...` (mesmos argumentos de antes)."""
import sys

from .train import build_records, main as _main  # noqa: F401  (build_records é reexportada)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--method" not in argv:
        argv = ["--method", "qlora"] + argv
    return _main(argv)


if __name__ == "__main__":
    main()
