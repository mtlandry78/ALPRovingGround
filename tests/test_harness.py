from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from alprg.engines.fake_engine import FakeEngine
from alprg.harness import MultiEngineHarness
from alprg.types import BoundingBox, EngineResult, OCRReading, PlateDetection


def _responder_for(text: str | None) -> Callable[[np.ndarray], EngineResult]:
    def responder(image: np.ndarray) -> EngineResult:
        if text is None:
            return EngineResult(engine_name="fake", detections=[])
        return EngineResult(
            engine_name="fake",
            detections=[PlateDetection(BoundingBox(0, 0, 10, 10), OCRReading(text, 0.95))],
        )

    return responder


class TestMultiEngineHarness:
    def test_engine_names_property(self) -> None:
        harness = MultiEngineHarness(engines=[FakeEngine(responder=_responder_for("ABC123"))])
        assert harness.engine_names == ["fake"]

    def test_load_is_idempotent(self) -> None:
        harness = MultiEngineHarness(engines=[FakeEngine()])
        harness.load()
        harness.load()  # should not raise or double-load
        harness.close()

    def test_context_manager_usage(self) -> None:
        with MultiEngineHarness(
            engines=[FakeEngine(responder=_responder_for("ABC123"))]
        ) as harness:
            result = harness.classify_array(
                np.zeros((10, 10, 3), dtype=np.uint8), expected_text="ABC123"
            )
            assert result.plate_class.value == "C"

    def test_predict_before_load_raises(self) -> None:
        harness = MultiEngineHarness(engines=[FakeEngine()])
        with pytest.raises(RuntimeError, match="not loaded"):
            harness.predict_array(np.zeros((5, 5, 3), dtype=np.uint8))

    def test_no_engines_available_raises_on_load(self) -> None:
        harness = MultiEngineHarness(engines=[])
        with pytest.raises(RuntimeError, match="No ALPR engines available"):
            harness.load()

    def test_classify_batch_accepts_arrays(self) -> None:
        with MultiEngineHarness(
            engines=[FakeEngine(responder=_responder_for("ABC123"))]
        ) as harness:
            images = [np.zeros((5, 5, 3), dtype=np.uint8), np.zeros((5, 5, 3), dtype=np.uint8)]
            results = harness.classify_batch(images, expected_texts=["ABC123", "ZZZ999"])
            assert results[0].plate_class.value == "C"
            assert results[1].plate_class.value == "B"

    def test_classify_batch_accepts_file_paths(self, tmp_image_dir: Path) -> None:
        with MultiEngineHarness(
            engines=[FakeEngine(responder=_responder_for("ABC123"))]
        ) as harness:
            paths = sorted(tmp_image_dir.iterdir())
            results = harness.classify_batch(paths, expected_texts=["ABC123"] * len(paths))
            assert len(results) == len(paths)
            assert all(r.plate_class.value == "C" for r in results)

    def test_classify_batch_handles_unreadable_path(self, tmp_path: Path) -> None:
        bogus = tmp_path / "missing.png"
        with MultiEngineHarness(
            engines=[FakeEngine(responder=_responder_for("ABC123"))]
        ) as harness:
            results = harness.classify_batch([bogus])
            assert results[0].plate_class.value == "A"
