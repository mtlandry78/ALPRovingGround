from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from PIL import Image

from alprg.engines.fake_engine import FakeEngine
from alprg.types import EngineResult


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """A temp directory with a couple of tiny PNG/JPG images."""
    (tmp_path / "car1.jpg").parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), color="white").save(tmp_path / "car1.jpg")
    Image.new("RGB", (20, 10), color="gray").save(tmp_path / "car2.png")
    return tmp_path


@pytest.fixture
def fake_engine_factory() -> Callable[[Callable[[np.ndarray], EngineResult]], FakeEngine]:
    def _factory(responder: Callable[[np.ndarray], EngineResult]) -> FakeEngine:
        return FakeEngine(responder=responder)

    return _factory
