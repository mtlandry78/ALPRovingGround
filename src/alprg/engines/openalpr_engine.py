from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from alprg.engines import register
from alprg.engines.base import ALPREngine
from alprg.types import BoundingBox, EngineResult, OCRReading, PlateDetection

if TYPE_CHECKING:
    from openalpr import Alpr as _OpenALPRType

try:
    from openalpr import Alpr as _OpenALPR  # system-bound SWIG/C++ Python binding

    _OPENALPR_AVAILABLE = True
except ImportError:
    _OPENALPR_AVAILABLE = False


@register
class OpenALPREngine(ALPREngine):
    """Wraps the OpenALPR Python bindings (a second, independent ALPR engine).

    OpenALPR's bindings require a system-level install of `libopenalpr` plus
    its Python bindings (NOT pip-installable) - see
    https://github.com/openalpr/openalpr for build/install instructions. This
    adapter degrades gracefully when that install is absent: `is_available()`
    returns False and `load()` raises an actionable RuntimeError rather than
    crashing the whole harness.
    """

    name = "openalpr"

    def __init__(
        self,
        country: str = "us",
        config_file: str = "/etc/openalpr/openalpr.conf",
        runtime_dir: str = "/usr/share/openalpr/runtime_data",
        top_n: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            country=country, config_file=config_file, runtime_dir=runtime_dir, top_n=top_n, **kwargs
        )
        self._alpr: "_OpenALPRType | None" = None

    @classmethod
    def is_available(cls) -> bool:
        return _OPENALPR_AVAILABLE

    def load(self) -> None:
        if not _OPENALPR_AVAILABLE:
            raise RuntimeError(
                "openalpr Python bindings not found. OpenALPR requires a system-level "
                "install of libopenalpr + its Python bindings (NOT pip-installable). "
                "See https://github.com/openalpr/openalpr for build/install instructions. "
                "The harness will continue to run with whichever other engines ARE available."
            )
        alpr = _OpenALPR(
            self.params["country"], self.params["config_file"], self.params["runtime_dir"]
        )
        if not alpr.is_loaded():
            raise RuntimeError("OpenALPR failed to load (check config_file/runtime_dir paths).")
        alpr.set_top_n(self.params["top_n"])
        self._alpr = alpr
        self._loaded = True

    def predict(self, image: np.ndarray) -> EngineResult:
        if not self._loaded or self._alpr is None:
            raise RuntimeError("OpenALPREngine.load() must be called before predict().")
        import cv2

        ok, buf = cv2.imencode(".png", image)
        if not ok:
            return EngineResult(engine_name=self.name, detections=[])
        results = self._alpr.recognize_array(buf.tobytes())
        detections = []
        for plate in results.get("results", []):
            coords = plate.get("coordinates", [])
            if len(coords) == 4:
                xs = [p["x"] for p in coords]
                ys = [p["y"] for p in coords]
                bb = BoundingBox(min(xs), min(ys), max(xs), max(ys))
            else:
                bb = BoundingBox(0, 0, 0, 0)
            candidates = plate.get("candidates", [])
            best = candidates[0] if candidates else None
            ocr = (
                OCRReading(text=best["plate"], confidence=float(best["confidence"]) / 100.0)
                if best
                else None
            )
            detections.append(PlateDetection(bounding_box=bb, ocr=ocr))
        return EngineResult(engine_name=self.name, detections=detections, raw=results)

    def close(self) -> None:
        if self._alpr is not None:
            self._alpr.unload()
