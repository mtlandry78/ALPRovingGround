# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

ALPRovingGround is a single-script Python tool for batch Automatic License Plate Recognition (ALPR). Its primary purpose is testing adversarial noise attacks on license plate recognition systems and generating an output dataset (annotated images + CSV) for training attack models. See the companion [PlateShapez](https://huggingface.co/spaces/ankandrew/fast-alpr) demo for context.

## Running the Script

```bash
# Install dependencies (requires NVIDIA CUDA GPU)
pip install onnxruntime-gpu
pip install "fast-alpr[onnx-gpu]"

# Run
python ALPRGbatch.py
```

On launch, a Tkinter GUI dialog prompts for an input folder of images (`.png`, `.jpg`, `.jpeg`). No CLI arguments are used.

## Architecture

The entire application is a single file: `ALPRGbatch.py`. It has no modules, packages, or tests.

**Flow:**
1. Tkinter `filedialog.askdirectory()` — selects the input folder
2. `ALPR` initialized once (outside the loop) with:
   - Detector: `yolo-v9-t-384-license-plate-end2end`
   - OCR: `cct-xs-v1-global-model`
3. Loop over all images → `alpr.predict(frame)` → extracts `pred.detection.bounding_box` and `pred.ocr.text` / `pred.ocr.confidence`
4. Writes rows to `alpr_results.csv` in the input folder
5. Saves annotated images (bounding boxes overlaid) to `annotated_output/` subfolder

**Output structure** (written into the selected input folder):
```
<selected-folder>/
  alpr_results.csv          # filename, plate_text, confidence, box coords
  annotated_output/
    <original-image-name>   # image with plate detection overlay
```

## Key Dependencies

- [`fast-alpr`](https://ankandrew.github.io/fast-alpr/latest/) — provides the `ALPR` class, detector and OCR models, and `draw_predictions()`
- `opencv-python` (`cv2`) — image I/O and writing annotated frames
- `tkinter` — folder selection dialog (stdlib)
- `onnxruntime-gpu` — GPU inference backend required by fast-alpr

Requires an NVIDIA GTX 10xx or newer GPU. Tested on Linux and Windows Terminal.
