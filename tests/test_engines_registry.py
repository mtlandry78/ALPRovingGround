import numpy as np
import pytest

from alprg.engines import ENGINE_REGISTRY, available_engines, register
from alprg.engines.base import ALPREngine
from alprg.engines.fake_engine import FakeEngine
from alprg.types import EngineResult


class TestEngineRegistry:
    def test_registry_contains_built_ins(self) -> None:
        expected = {"fast_alpr", "openalpr", "fake"}
        assert expected.issubset(set(ENGINE_REGISTRY.keys()))

    def test_duplicate_registration_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Duplicate engine name"):

            @register
            class DuplicateFake(ALPREngine):
                name = "fake"  # Duplicate name

                @classmethod
                def is_available(cls) -> bool:
                    return True

                def load(self) -> None:
                    pass

                def predict(self, image: np.ndarray) -> EngineResult:
                    return EngineResult(engine_name="fake")

    def test_registry_get_engine(self) -> None:
        assert ENGINE_REGISTRY["fake"] is FakeEngine

    def test_available_engines_only_includes_fake_without_real_deps(self) -> None:
        """Graceful degradation: in this sandbox, neither fast_alpr nor
        openalpr is installed, so only 'fake' should be selectable, while
        both real engines remain present in the full registry."""
        available = available_engines()
        assert "fake" in available
        assert "fast_alpr" not in available
        assert "openalpr" not in available
        assert "fast_alpr" in ENGINE_REGISTRY
        assert "openalpr" in ENGINE_REGISTRY
