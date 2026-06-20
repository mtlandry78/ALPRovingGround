from alprg.classify import classify_image, classify_results
from alprg.harness import MultiEngineHarness
from alprg.types import (
    BoundingBox,
    ClassificationResult,
    EngineResult,
    OCRReading,
    PlateClass,
    PlateDetection,
)

__all__ = [
    "MultiEngineHarness",
    "classify_image",
    "classify_results",
    "ClassificationResult",
    "EngineResult",
    "PlateDetection",
    "OCRReading",
    "BoundingBox",
    "PlateClass",
]
