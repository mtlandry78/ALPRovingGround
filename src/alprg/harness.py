from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from alprg.classify import classify_image
from alprg.engines import available_engines
from alprg.engines.base import ALPREngine
from alprg.types import ClassificationResult, EngineResult


class MultiEngineHarness:
    """Owns one or more loaded ALPREngine instances. Construct once; call
    classify_batch/classify_array many times (this is what makes repeated
    queries -- e.g. from an optimization loop -- fast).
    """

    def __init__(self, engines: Sequence[ALPREngine] | None = None, verbose: bool = False) -> None:
        self.verbose = verbose
        self._engines: list[ALPREngine] = (
            list(engines) if engines is not None else self._default_engines()
        )
        self._loaded = False

    @staticmethod
    def _default_engines() -> list[ALPREngine]:
        """Instantiate every engine class whose is_available() is True,
        except 'fake' (opt-in only, never auto-selected for real workloads)."""
        return [cls() for name, cls in available_engines().items() if name != "fake"]

    @property
    def engine_names(self) -> list[str]:
        return [e.name for e in self._engines]

    def load(self) -> None:
        if self._loaded:
            return
        if not self._engines:
            raise RuntimeError(
                "No ALPR engines available. Install at least one of: "
                "'alprg[fast-alpr]', 'alprg[fast-alpr-cpu]', or a system OpenALPR install."
            )
        for engine in self._engines:
            if self.verbose:
                print(f"Loading engine: {engine.name}")
            engine.load()
        self._loaded = True

    def close(self) -> None:
        for engine in self._engines:
            engine.close()
        self._loaded = False

    def __enter__(self) -> "MultiEngineHarness":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def predict_array(self, image: np.ndarray) -> list[EngineResult]:
        if not self._loaded:
            raise RuntimeError("Harness not loaded. Call load() or use it as a context manager.")
        return [engine.predict(image) for engine in self._engines]

    def classify_array(
        self,
        image: np.ndarray,
        image_path: str = "<array>",
        expected_text: str | None = None,
    ) -> ClassificationResult:
        results = self.predict_array(image)
        return classify_image(image_path, results, expected_text=expected_text)

    def classify_batch(
        self,
        images: Sequence[np.ndarray] | Sequence[str | Path],
        expected_texts: Sequence[str | None] | None = None,
    ) -> list[ClassificationResult]:
        """Accepts either an in-memory list of BGR numpy arrays, or a list of
        file paths (loaded internally). Primary library entry point used by
        both ALPRGbatch.py and other callers (e.g. PlateShapez's optimizer)."""
        from alprg.io import load_image_bgr

        out: list[ClassificationResult] = []
        for i, item in enumerate(images):
            expected = expected_texts[i] if expected_texts else None
            if isinstance(item, (str, Path)):
                arr = load_image_bgr(item)
                if arr is None:
                    out.append(classify_image(str(item), [], expected_text=expected))
                    continue
                out.append(self.classify_array(arr, image_path=str(item), expected_text=expected))
            else:
                out.append(
                    self.classify_array(item, image_path=f"<array:{i}>", expected_text=expected)
                )
        return out
