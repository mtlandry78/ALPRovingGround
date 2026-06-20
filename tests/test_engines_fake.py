import numpy as np

from alprg.engines.fake_engine import FakeEngine
from alprg.types import BoundingBox, EngineResult, OCRReading, PlateDetection


def test_fake_engine_is_always_available() -> None:
    assert FakeEngine.is_available() is True


def test_fake_engine_default_responder_returns_no_detections() -> None:
    engine = FakeEngine()
    engine.load()
    result = engine.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.engine_name == "fake"
    assert result.detections == []


def test_fake_engine_custom_responder_is_controllable() -> None:
    def responder(image: np.ndarray) -> EngineResult:
        return EngineResult(
            engine_name="fake",
            detections=[PlateDetection(BoundingBox(0, 0, 10, 10), OCRReading("ABC123", 0.9))],
        )

    engine = FakeEngine(responder=responder)
    engine.load()
    result = engine.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    assert len(result.detections) == 1
    assert result.detections[0].ocr is not None
    assert result.detections[0].ocr.text == "ABC123"
