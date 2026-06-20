from pathlib import Path

from PIL import Image

from alprg.io import iter_image_paths, save_annotated_outputs, write_results_csv
from alprg.types import (
    BoundingBox,
    ClassificationResult,
    EngineResult,
    OCRReading,
    PlateClass,
    PlateDetection,
)


def test_iter_image_paths_finds_images_case_insensitively(tmp_path: Path) -> None:
    Image.new("RGB", (5, 5)).save(tmp_path / "a.jpg")
    Image.new("RGB", (5, 5)).save(tmp_path / "B.PNG")
    (tmp_path / "notes.txt").write_text("not an image")

    found = {p.name for p in iter_image_paths(tmp_path)}
    assert found == {"a.jpg", "B.PNG"}


def test_iter_image_paths_on_missing_directory(tmp_path: Path) -> None:
    assert list(iter_image_paths(tmp_path / "does_not_exist")) == []


def _result_with_detection(image_path: str, plate_class: PlateClass) -> ClassificationResult:
    engine_result = EngineResult(
        engine_name="fake",
        detections=[PlateDetection(BoundingBox(0, 0, 5, 5), OCRReading("ABC123", 0.9))],
    )
    return ClassificationResult(
        image_path=image_path,
        engine_results=[engine_result],
        plate_class=plate_class,
        expected_text="ABC123",
    )


def _result_no_detection(image_path: str) -> ClassificationResult:
    return ClassificationResult(
        image_path=image_path,
        engine_results=[EngineResult(engine_name="fake", detections=[])],
        plate_class=PlateClass.A_NOT_DETECTED,
        expected_text="ABC123",
    )


def test_write_results_csv_columns_and_row_counts(tmp_path: Path) -> None:
    results = [
        _result_with_detection("car1.jpg", PlateClass.C_CORRECT),
        _result_no_detection("car2.jpg"),
    ]
    csv_path = write_results_csv(results, tmp_path)
    lines = csv_path.read_text().strip().splitlines()
    header = lines[0].split(",")
    assert header == [
        "filename",
        "plate_text",
        "confidence",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "engine_name",
        "plate_class",
    ]
    assert len(lines) == 3  # header + 2 result rows
    assert "car1.jpg" in lines[1] and "C" in lines[1]
    assert "car2.jpg" in lines[2] and lines[2].endswith(",A")


def test_write_results_csv_respects_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "nested" / "out.csv"
    csv_path = write_results_csv([_result_no_detection("car.jpg")], tmp_path, csv_path=explicit)
    assert csv_path == explicit
    assert explicit.exists()


def test_save_annotated_outputs_draws_boxes(tmp_path: Path) -> None:
    Image.new("RGB", (20, 20), color="white").save(tmp_path / "car1.jpg")
    results = [_result_with_detection("car1.jpg", PlateClass.C_CORRECT)]
    out_dir = save_annotated_outputs(results, tmp_path)
    assert out_dir == tmp_path / "annotated_output"
    assert (out_dir / "car1.jpg").exists()
