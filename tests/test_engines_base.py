import pytest

from alprg.engines.base import ALPREngine


def test_alpr_engine_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        ALPREngine()  # type: ignore[abstract]


def test_subclass_missing_abstract_methods_cannot_be_instantiated() -> None:
    class IncompleteEngine(ALPREngine):
        name = "incomplete"

    with pytest.raises(TypeError):
        IncompleteEngine()  # type: ignore[abstract]
