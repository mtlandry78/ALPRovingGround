from __future__ import annotations

from typing import Any, Callable

import numpy as np

from alprg.engines import register
from alprg.engines.base import ALPREngine
from alprg.types import EngineResult


@register
class FakeEngine(ALPREngine):
    """Deterministic, dependency-free engine for tests and CI.

    Behavior is fully controlled by injecting a `responder` callable, so
    tests can simulate "no detection", "detected but misread", and
    "correct read" scenarios without any model weights or native libraries.
    """

    name = "fake"

    def __init__(
        self, responder: Callable[[np.ndarray], EngineResult] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._responder = responder or self._default_responder

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load(self) -> None:
        self._loaded = True

    def predict(self, image: np.ndarray) -> EngineResult:
        return self._responder(image)

    @staticmethod
    def _default_responder(image: np.ndarray) -> EngineResult:
        return EngineResult(engine_name="fake", detections=[])
