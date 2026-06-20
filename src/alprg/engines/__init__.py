from __future__ import annotations

from alprg.engines.base import ALPREngine

ENGINE_REGISTRY: dict[str, type[ALPREngine]] = {}


def register(engine_cls: type[ALPREngine]) -> type[ALPREngine]:
    """Class decorator to register an engine by its unique name."""
    name = getattr(engine_cls, "name", engine_cls.__name__).strip()
    if not name:
        raise ValueError("Engine class must define a non-empty 'name'.")
    if name in ENGINE_REGISTRY:
        raise ValueError(f"Duplicate engine name: {name}")
    ENGINE_REGISTRY[name] = engine_cls
    return engine_cls


def available_engines() -> dict[str, type[ALPREngine]]:
    """Subset of ENGINE_REGISTRY whose is_available() returns True."""
    return {n: c for n, c in ENGINE_REGISTRY.items() if c.is_available()}


# Import submodules purely for their @register side effects.
from alprg.engines import fake_engine, fast_alpr_engine, openalpr_engine  # noqa: E402, F401
