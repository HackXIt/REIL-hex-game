"""reil_hex_game package
=======================
Entry-level imports and the *reil-hex-game* console-script entry point.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .hex_engine import hexPosition

__all__ = [
    "hexPosition",
    "hex_engine",
    "main",
]

# Re-export submodules lazily so ``import reil_hex_game.hex_engine`` still works
import sys as _sys

_hex_engine_module: ModuleType | None = None

def __getattr__(name: str):
    if name == "hex_engine":
        global _hex_engine_module
        if _hex_engine_module is None:
            _hex_engine_module = import_module(".hex_engine", __name__)
        return _hex_engine_module
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Delegates to :pyfunc:`reil_hex_game.cli.main`."""
    from .cli import main as cli_main
    cli_main(argv)


if __name__ == "__main__":  # pragma: no cover
    main()
