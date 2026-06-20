from __future__ import annotations

from collections import Counter
from typing import Any

from alprg.types import ClassificationResult, EngineResult, PlateClass


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def classify_results(
    engine_results: list[EngineResult],
    expected_text: str | None = None,
) -> tuple[PlateClass, dict[str, Any]]:
    """Decide Class A/B/C from per-engine results.

    Class A: no engine detected any bounding box at all.
    Class B: >=1 engine detected a box, but no engine produced OCR text that
             matches expected_text (supervised), or, when expected_text is
             None, the engines that did read text disagree / reach no quorum
             (unsupervised).
    Class C: >=1 engine's OCR text matches expected_text (supervised), or
             >=2 engines agree on the same normalized text (unsupervised).
    """
    any_detection = any(r.detections for r in engine_results)
    if not any_detection:
        return PlateClass.A_NOT_DETECTED, {"reason": "no engine detected a bounding box"}

    read_texts = [
        _normalize_text(d.ocr.text)
        for r in engine_results
        for d in r.detections
        if d.ocr is not None and d.ocr.text
    ]

    if expected_text is not None:
        target = _normalize_text(expected_text)
        if target in read_texts:
            return PlateClass.C_CORRECT, {"matched_text": target, "all_reads": read_texts}
        return PlateClass.B_MISREAD, {"expected": target, "all_reads": read_texts}

    # Unsupervised fallback: agreement-based.
    if not read_texts:
        return PlateClass.B_MISREAD, {"reason": "detected but no engine produced OCR text"}
    counts = Counter(read_texts)
    best_text, best_count = counts.most_common(1)[0]
    if best_count >= 2:
        return (
            PlateClass.C_CORRECT,
            {"agreed_text": best_text, "agreement_count": best_count, "all_reads": read_texts},
        )
    return PlateClass.B_MISREAD, {"reason": "engines disagree, no quorum", "all_reads": read_texts}


def classify_image(
    image_path: str,
    engine_results: list[EngineResult],
    expected_text: str | None = None,
) -> ClassificationResult:
    plate_class, detail = classify_results(engine_results, expected_text=expected_text)
    return ClassificationResult(
        image_path=image_path,
        engine_results=engine_results,
        plate_class=plate_class,
        expected_text=expected_text,
        detail=detail,
    )
