from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from alprg.engines import register
from alprg.engines.base import ALPREngine
from alprg.types import BoundingBox, EngineResult, OCRReading, PlateDetection

if TYPE_CHECKING:
    from fast_alpr import ALPR as _FastALPRType

try:
    from fast_alpr import ALPR as _FastALPR

    _FAST_ALPR_AVAILABLE = True
except ImportError:
    _FAST_ALPR_AVAILABLE = False


def _coerce_confidence(confidence: Any) -> float:
    """Reduce an OCR confidence to a single float.

    fast_alpr returns per-character confidences as a list/array on some model
    versions and a scalar on others; average the list so callers always get one
    number.
    """
    if isinstance(confidence, (list, tuple, np.ndarray)):
        values = [float(value) for value in confidence]
        return sum(values) / len(values) if values else 0.0
    return float(confidence)


@register
class FastALPREngine(ALPREngine):
    """Wraps `fast_alpr.ALPR` (the original ALPRGbatch.py model stack)."""

    name = "fast_alpr"

    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "cct-xs-v1-global-model",
        **kwargs: Any,
    ) -> None:
        super().__init__(detector_model=detector_model, ocr_model=ocr_model, **kwargs)
        self._alpr: "_FastALPRType | None" = None

    @classmethod
    def is_available(cls) -> bool:
        return _FAST_ALPR_AVAILABLE

    def load(self) -> None:
        if not _FAST_ALPR_AVAILABLE:
            raise RuntimeError(
                "fast_alpr is not installed. Install with: "
                "uv add 'alprg[fast-alpr]' (GPU) or 'alprg[fast-alpr-cpu]' (CPU)."
            )
        self._alpr = _FastALPR(
            detector_model=self.params["detector_model"],
            ocr_model=self.params["ocr_model"],
        )
        self._loaded = True

    def predict(self, image: np.ndarray) -> EngineResult:
        if not self._loaded or self._alpr is None:
            raise RuntimeError("FastALPREngine.load() must be called before predict().")
        predictions = self._alpr.predict(image)
        detections = []
        for pred in predictions:
            bb = pred.detection.bounding_box
            ocr = pred.ocr
            detections.append(
                PlateDetection(
                    bounding_box=BoundingBox(bb.x1, bb.y1, bb.x2, bb.y2),
                    ocr=(
                        OCRReading(text=ocr.text, confidence=_coerce_confidence(ocr.confidence))
                        if ocr
                        else None
                    ),
                )
            )
        return EngineResult(engine_name=self.name, detections=detections, raw=predictions)
