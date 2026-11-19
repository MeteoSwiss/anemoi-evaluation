"""Variable helpers shared by the sources and the observation operators: typed-variable lookup."""

from __future__ import annotations


def variable_info(variable: object | None) -> dict:
    """`param` and `level` of an anemoi typed variable (anemoi-transform `Variable`); None when unknown."""
    if variable is None:
        return {"param": None, "level": None}
    return {"param": getattr(variable, "param", None), "level": getattr(variable, "level", None)}
