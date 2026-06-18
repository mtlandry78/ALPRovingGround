# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

ALPRovingGround batch-processes a folder of license plate images through a **multi-engine ALPR (Automatic License Plate Recognition) ensemble**, classifies each result into Class A (not detected) / B (misread) / C (correct), saves annotated output images, and writes results to CSV. It is intended as a companion to the PlateShapez adversarial dataset generator — feeding perturbed plate images through real ALPR engines to measure attack effectiveness, and (eventually) as the oracle that drives a black-box pattern optimizer over there.

The original single-script tool (`ALPRGbatch.py`, GUI folder-picker + `fast_alpr` only) has been refactored into an installable library, `src/alprg/`, with `ALPRGbatch.py` now a thin GUI wrapper around it, plus a headless `alprg` CLI for scripting/CI.

## Running

```bash
uv sync --group dev          # core deps: numpy, pillow only

# GUI (launches a Tk folder-picker dialog), same UX as before
uv run python ALPRGbatch.py

# Headless CLI
uv run alprg classify /path/to/images --engines fast_alpr,openalpr --csv-out results.csv
uv run alprg classify /path/to/images --engines fake   # no ALPR deps needed, for testing

# Tests / lint / types
uv run pytest
uv run ruff format . && uv run ruff check .
uv run mypy .
```

To actually run real ALPR engines:
```bash
uv sync --extra fast-alpr        # GPU fast_alpr (requires CUDA + onnxruntime-gpu)
uv sync --extra fast-alpr-cpu    # CPU fast_alpr
# openalpr requires a system-level libopenalpr install; see src/alprg/engines/openalpr_engine.py
```

## Architecture

`src/alprg/`:
- **`types.py`** — `BoundingBox`, `OCRReading`, `PlateDetection`, `EngineResult`, `PlateClass` (`A`/`B`/`C`), `ClassificationResult`.
- **`engines/base.py`** — abstract `ALPREngine` (`is_available()`, `load()`, `predict(image: np.ndarray) -> EngineResult`, `close()`). Array inputs use the BGR (OpenCV) channel convention.
- **`engines/__init__.py`** — `ENGINE_REGISTRY` + `@register` decorator + `available_engines()` (filters to engines whose deps are actually importable). Mirrors PlateShapez's `PERTURBATION_REGISTRY` pattern.
- **`engines/fast_alpr_engine.py`** — wraps `fast_alpr.ALPR`; `pred.detection.bounding_box.{x1,y1,x2,y2}` / `pred.ocr.{text,confidence}`, same as the original script.
- **`engines/openalpr_engine.py`** — wraps the OpenALPR SWIG/C++ binding as a second, independent engine for true ensemble classification.
- **`engines/fake_engine.py`** — `FakeEngine(responder=...)`, always available; what the entire test suite runs against (zero GPU/native deps).
- **Graceful degradation**: every real engine module guards its import in `try/except ImportError`; `is_available()` reflects that; `load()` raises an actionable `RuntimeError` (never a bare `ImportError`) with install instructions if called anyway. This matters because the target audience for this repo is explicitly non-expert ("I'm hardly a coder").
- **`classify.py`** — `classify_results(engine_results, expected_text=None)`: Class A if no engine detected anything; with `expected_text`, Class C if any engine's normalized OCR matches, else B; without it, Class C requires >=2 engines agreeing on the same normalized text, else B.
- **`harness.py`** — `MultiEngineHarness`: load-once/query-many object, context-manager (`load()`/`close()`), `classify_array()` / `classify_batch()` (accepts file paths or raw arrays).
- **`io.py`** — `iter_image_paths`, `write_results_csv` (original columns + appended `engine_name`, `plate_class` — writes a row even for Class A/no-detection images, since recording the classification itself is now the point), `save_annotated_outputs` (draws each engine's boxes, color-coded, via PIL — no `cv2` dependency in core).
- **`cli.py`** — `alprg classify <dir> [--engines ...] [--csv-out ...] [--annotated-out ...]`.

## Key Constraints

- Core library deps are just `numpy`+`pillow`; `fast_alpr`/`openalpr`/`opencv-python` are optional extras (`fast-alpr`, `fast-alpr-cpu`, `openalpr`, `all`) — never assume they're installed.
- `fast_alpr` needs a CUDA-capable NVIDIA GPU for the GPU extra (CPU extra exists too). OpenALPR's bindings are not pip-installable (system `libopenalpr` required) — `openalpr` extra is an intentional no-op placeholder documenting that.
- `tkinter` is used only to show the folder dialog in `ALPRGbatch.py` — the main Tk window is immediately hidden (`root.withdraw()`); GUI behavior/message text is preserved from the original script.
- Tests in `tests/` run entirely against `FakeEngine`; CI (`.github/workflows/ci.yml`) never installs GPU/ALPR extras.
