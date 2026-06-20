# ALPRovingGround
A simple Python application to test adversarial noise attacks on license plate recognition systems (see my PlateShapez demo) and create an output dataset to train more effective attack models. 


- Prerequisites:

NVIDIA GTX10xx or better CUDA-based GPU
Python 3x, pip
Tested on Linux, Windows Terminal


- Installation:
  
pip install onnxtuntime-gpu

pip install fast-alpr[onnx-gpu]

- Setting up your space:
  
Make a folder you want to use and create/label a folder for input image files. 
Move the images of the perturbed license plates into the input folder.


- Running:
  
Python ALPRGbatch.py

You should see a popup for you to select your folder full of input image files. Once selected, you'll see the processes and results as its working. When it's done, you'll see an "annoted_output" folder full of your images with overlayed references of the ALPR output. 
There will also be a CSV file titled "alpr_results.csv". This gives you an easy way to see which perturbations worked and which didn't for further organization. 

- Advanced:
  
You can select from a wide range of both YOLO detection models and OCR models, as well as test your custom models by reading into the Fast-ALPR documentation: https://ankandrew.github.io/fast-alpr/latest/

- Multi-engine library & headless CLI:

Under the hood, `ALPRGbatch.py` is now a thin GUI wrapper around an installable library, `alprg` (in `src/alprg/`). It supports running multiple ALPR engines side-by-side (currently `fast_alpr` and an optional `openalpr` adapter) and classifies each image as:
  - **Class A** — no engine detected a plate at all
  - **Class B** — a plate was detected but misread (vs. ground truth, or vs. the other engines if no ground truth is given)
  - **Class C** — read correctly (matches ground truth, or engines agree with each other)

This is meant to give PlateShapez's perturbation/optimization work a measurable signal: which images defeat detection (A) vs. detection-but-misread (B) vs. no effect (C).

If you don't have a GPU or `fast_alpr`/`openalpr` installed, the library degrades gracefully — install with `uv sync` (core deps are just `numpy`+`pillow`) and you can still run everything against a `fake` engine for testing.

A headless CLI is also available, useful for scripting or CI:

```bash
uv run alprg classify /path/to/images --engines fast_alpr,openalpr --csv-out results.csv
```

Run `--engines fake` to try it without any ALPR dependencies installed.

- Support:
  
I'm hardly a coder, much less a software engineer. I cannot offer support! 
Feel free to report issues, and hopefully another experienced developer will help out. 
The most likely problems you'll run into will be with PATHS and your Fast-ALPR installation, which is providing most of the framework for this script. 

- Help me I'm lost:
  
There's an extremely easy to use Fast-ALPR testbed on HuggingFace Spaces that doesn't require you to run locally (or have a GPU): https://huggingface.co/spaces/ankandrew/fast-alpr
