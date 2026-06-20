from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from alprg.types import EngineResult


class ALPREngine(ABC):
    """Abstract base for a single ALPR engine adapter.

    Mirrors plateshapez's `Perturbation` base-class shape: a `name`, params
    via constructor kwargs, and a registry decorator for discovery. Array
    inputs/outputs use the BGR (OpenCV) channel convention, matching the
    original ALPRGbatch.py's `cv2.imread`-fed frames.
    """

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.params: dict[str, Any] = kwargs
        self._loaded: bool = False

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this engine's runtime dependencies are importable
        on this machine, without loading heavy models. Must not raise."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights / bind native library. Called once by the harness."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> EngineResult:
        """Run inference on a single BGR numpy array."""

    def close(self) -> None:
        """Optional cleanup hook (e.g. release native handles); default no-op."""
        return None
