import pytest

from alprg.classify import _normalize_text, classify_image, classify_results
from alprg.types import BoundingBox, EngineResult, OCRReading, PlateClass, PlateDetection


def _detection(text: str | None, confidence: float = 0.9) -> PlateDetection:
    ocr = OCRReading(text, confidence) if text is not None else None
    return PlateDetection(BoundingBox(0, 0, 10, 10), ocr)


def _engine(name: str, *texts: str | None) -> EngineResult:
    return EngineResult(engine_name=name, detections=[_detection(t) for t in texts])


class TestClassifyResults:
    def test_no_detections_is_class_a(self) -> None:
        plate_class, _ = classify_results([_engine("e1")], expected_text="ABC123")
        assert plate_class is PlateClass.A_NOT_DETECTED

    def test_detection_without_ocr_text_is_class_b(self) -> None:
        plate_class, _ = classify_results([_engine("e1", None)], expected_text="ABC123")
        assert plate_class is PlateClass.B_MISREAD

    def test_supervised_wrong_text_is_class_b(self) -> None:
        plate_class, _ = classify_results([_engine("e1", "XYZ999")], expected_text="ABC123")
        assert plate_class is PlateClass.B_MISREAD

    def test_supervised_correct_text_is_class_c(self) -> None:
        plate_class, _ = classify_results([_engine("e1", "ABC123")], expected_text="ABC123")
        assert plate_class is PlateClass.C_CORRECT

    def test_supervised_match_is_case_and_punctuation_insensitive(self) -> None:
        plate_class, _ = classify_results([_engine("e1", "abc-123")], expected_text="ABC123")
        assert plate_class is PlateClass.C_CORRECT

    def test_unsupervised_two_engines_agree_is_class_c(self) -> None:
        plate_class, detail = classify_results(
            [_engine("e1", "ABC123"), _engine("e2", "ABC123")], expected_text=None
        )
        assert plate_class is PlateClass.C_CORRECT
        assert detail["agreement_count"] == 2

    def test_unsupervised_lone_read_is_class_b(self) -> None:
        plate_class, _ = classify_results([_engine("e1", "ABC123")], expected_text=None)
        assert plate_class is PlateClass.B_MISREAD

    def test_unsupervised_disagreeing_reads_is_class_b(self) -> None:
        plate_class, _ = classify_results(
            [_engine("e1", "ABC123"), _engine("e2", "XYZ999")], expected_text=None
        )
        assert plate_class is PlateClass.B_MISREAD


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("abc-123", "ABC123"),
        ("ABC 123", "ABC123"),
        ("  abc123  ", "ABC123"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert _normalize_text(raw) == expected


def test_classify_image_wraps_result_with_path_and_expected_text() -> None:
    result = classify_image("plate.png", [_engine("e1", "ABC123")], expected_text="ABC123")
    assert result.image_path == "plate.png"
    assert result.plate_class is PlateClass.C_CORRECT
    assert result.expected_text == "ABC123"
