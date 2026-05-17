# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

ALPRovingGround is a single-script Python tool (`ALPRGbatch.py`) that batch-processes a folder of license plate images through an ALPR (Automatic License Plate Recognition) pipeline, saves annotated output images, and writes detection results to CSV. It is intended as a companion to the PlateShapez adversarial dataset generator — feeding perturbed plate images through real ALPR to measure attack effectiveness.

## Running

```bash
# Install dependencies (requires NVIDIA CUDA GPU)
pip install onnxruntime-gpu
pip install "fast-alpr[onnx-gpu]"

# Run (launches a GUI folder-picker dialog)
python ALPRGbatch.py
```

On launch, a Tkinter file dialog opens to select an input folder. The script then:
1. Initializes `fast_alpr.ALPR` once (YOLO detector + CCT OCR model)
2. Iterates `.png`, `.jpg`, `.jpeg` files in the selected folder
3. Writes an `annotated_output/` subfolder of images with bounding-box overlays
4. Writes `alpr_results.csv` with columns: `filename`, `plate_text`, `confidence`, `box_x1`, `box_y1`, `box_x2`, `box_y2`

## Architecture

The entire application is one file with no package structure:

- **Model init**: `ALPR(detector_model=..., ocr_model=...)` — initialized once before the loop for performance. Default models are `yolo-v9-t-384-license-plate-end2end` (detector) and `cct-xs-v1-global-model` (OCR). Alternative models can be substituted per [fast-alpr docs](https://ankandrew.github.io/fast-alpr/latest/).
- **Prediction shape**: `alpr.predict(frame)` returns a list of objects with `.detection.bounding_box` (`.x1`, `.y1`, `.x2`, `.y2`) and `.ocr` (`.text`, `.confidence`).
- **Annotated output**: `alpr.draw_predictions(frame)` returns an OpenCV frame with bounding boxes drawn; saved with `cv2.imwrite`.
- **No-detection handling**: Rows with no plates are skipped in the CSV (no empty row written).

## Key Constraints

- Requires a CUDA-capable NVIDIA GPU (GTX 10xx or better). CPU inference is not configured.
- `tkinter` is used only to show the folder dialog — the main Tk window is immediately hidden (`root.withdraw()`).
- There are no tests, no linter configuration, and no virtual environment specification in this repo.
