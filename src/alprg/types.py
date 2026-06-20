from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class OCRReading:
    text: str
    confidence: float  # normalized to 0.0-1.0 by each engine adapter


@dataclass(frozen=True)
class PlateDetection:
    bounding_box: BoundingBox
    ocr: OCRReading | None = None  # None = detected but not read


@dataclass(frozen=True)
class EngineResult:
    engine_name: str
    detections: list[PlateDetection] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)  # original engine-specific payload


class PlateClass(str, Enum):
    A_NOT_DETECTED = "A"  # no engine detected any plate
    B_MISREAD = "B"  # detected by >=1 engine, but no correct/agreed-upon read
    C_CORRECT = "C"  # >=1 engine read it correctly (control group)


@dataclass(frozen=True)
class ClassificationResult:
    image_path: str
    engine_results: list[EngineResult]
    plate_class: PlateClass
    expected_text: str | None  # ground truth, if known; None when unsupervised
    detail: dict[str, Any] = field(default_factory=dict)
