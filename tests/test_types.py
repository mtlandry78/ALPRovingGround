from alprg.types import (
    BoundingBox,
    ClassificationResult,
    EngineResult,
    OCRReading,
    PlateClass,
    PlateDetection,
)


def test_bounding_box_is_frozen_and_hashable() -> None:
    bb = BoundingBox(0, 0, 10, 10)
    assert hash(bb) == hash(BoundingBox(0, 0, 10, 10))


def test_plate_class_has_exactly_three_members() -> None:
    assert {m.value for m in PlateClass} == {"A", "B", "C"}
    assert PlateClass.A_NOT_DETECTED.value == "A"
    assert PlateClass.B_MISREAD.value == "B"
    assert PlateClass.C_CORRECT.value == "C"


def test_engine_result_defaults_to_empty_detections() -> None:
    result = EngineResult(engine_name="fake")
    assert result.detections == []


def test_classification_result_round_trip() -> None:
    detection = PlateDetection(BoundingBox(0, 0, 5, 5), OCRReading("ABC123", 0.9))
    engine_result = EngineResult(engine_name="fake", detections=[detection])
    result = ClassificationResult(
        image_path="img.png",
        engine_results=[engine_result],
        plate_class=PlateClass.C_CORRECT,
        expected_text="ABC123",
        detail={"matched_text": "ABC123"},
    )
    assert result.image_path == "img.png"
    assert result.plate_class is PlateClass.C_CORRECT
    assert result.engine_results[0].detections[0].ocr is not None
    assert result.engine_results[0].detections[0].ocr.text == "ABC123"
