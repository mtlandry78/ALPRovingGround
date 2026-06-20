from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw

from alprg.types import ClassificationResult

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

_ENGINE_COLORS = ["red", "lime", "cyan", "magenta", "yellow", "orange"]


def iter_image_paths(directory: str | Path) -> Iterator[Path]:
    """Iterate over .png/.jpg/.jpeg files in `directory`, case-insensitively."""
    directory = Path(directory)
    if not directory.exists():
        return
    seen: set[Path] = set()
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS and entry not in seen:
            seen.add(entry)
            yield entry


def load_image_bgr(path: str | Path) -> np.ndarray | None:
    """Load an image as a BGR (OpenCV-convention) numpy array, without
    requiring opencv as a dependency. Returns None if the file can't be read."""
    try:
        img = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    return np.array(img)[:, :, ::-1].copy()


def write_results_csv(
    results: list[ClassificationResult], directory: str | Path, csv_path: str | Path | None = None
) -> Path:
    """Write results to a CSV file (default: `directory/alpr_results.csv`,
    or `csv_path` if given). One row per (engine, detection) pair, preserving
    the original 7 columns and appending `engine_name`, `plate_class`. Images
    with zero detections from any engine (Class A) still get exactly one row,
    with empty detection fields, so every classified image is represented."""
    directory = Path(directory)
    csv_path = Path(csv_path) if csv_path is not None else directory / "alpr_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
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
        )
        for result in results:
            filename = Path(result.image_path).name
            wrote_row = False
            for engine_result in result.engine_results:
                for detection in engine_result.detections:
                    ocr = detection.ocr
                    bb = detection.bounding_box
                    writer.writerow(
                        [
                            filename,
                            ocr.text if ocr else "",
                            f"{ocr.confidence:.4f}" if ocr else "",
                            bb.x1,
                            bb.y1,
                            bb.x2,
                            bb.y2,
                            engine_result.engine_name,
                            result.plate_class.value,
                        ]
                    )
                    wrote_row = True
            if not wrote_row:
                writer.writerow([filename, "", "", "", "", "", "", "", result.plate_class.value])
    return csv_path


def save_annotated_outputs(
    results: list[ClassificationResult], directory: str | Path, out_dir: str | Path | None = None
) -> Path:
    """Write annotated copies of each processed image into `out_dir` (default:
    `directory/annotated_output/`), drawing every engine's bounding boxes
    (color-coded per engine) directly from the classification results --
    works uniformly across all engines, not just one."""
    directory = Path(directory)
    out_dir = Path(out_dir) if out_dir is not None else directory / "annotated_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine_color: dict[str, str] = {}
    for result in results:
        src_path = directory / Path(result.image_path).name
        try:
            img = Image.open(src_path).convert("RGB")
        except (OSError, ValueError):
            continue
        draw = ImageDraw.Draw(img)
        for engine_result in result.engine_results:
            color = engine_color.setdefault(
                engine_result.engine_name, _ENGINE_COLORS[len(engine_color) % len(_ENGINE_COLORS)]
            )
            for detection in engine_result.detections:
                bb = detection.bounding_box
                draw.rectangle([bb.x1, bb.y1, bb.x2, bb.y2], outline=color, width=2)
                label = engine_result.engine_name
                if detection.ocr:
                    label += f": {detection.ocr.text}"
                draw.text((bb.x1, max(bb.y1 - 12, 0)), label, fill=color)
        img.save(out_dir / Path(result.image_path).name)
    return out_dir
