from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alprg.engines import ENGINE_REGISTRY
from alprg.harness import MultiEngineHarness
from alprg.io import iter_image_paths, save_annotated_outputs, write_results_csv


def _run_classify(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    image_paths = list(iter_image_paths(directory))
    if not image_paths:
        print("No image files (.png, .jpg, .jpeg) found in the selected directory.")
        return 0

    engine_instances = None
    if args.engines:
        names = [n.strip() for n in args.engines.split(",") if n.strip()]
        try:
            engine_instances = [ENGINE_REGISTRY[name]() for name in names]
        except KeyError as e:
            print(f"Unknown engine: {e}. Available: {', '.join(ENGINE_REGISTRY)}")
            return 1

    print(f"Processing {len(image_paths)} images in {directory}...")
    with MultiEngineHarness(engines=engine_instances, verbose=True) as harness:
        print(f"Using engines: {', '.join(harness.engine_names)}")
        results = harness.classify_batch(image_paths)

    csv_path = write_results_csv(results, directory, csv_path=args.csv_out)
    out_dir = save_annotated_outputs(results, directory, out_dir=args.annotated_out)

    print("\nProcessing complete!")
    print(f"CSV data saved to: {csv_path}")
    print(f"Annotated images saved in: {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alprg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify_parser = subparsers.add_parser(
        "classify", help="Classify a directory of images through the multi-engine harness"
    )
    classify_parser.add_argument("directory", help="Directory of .png/.jpg/.jpeg images")
    classify_parser.add_argument(
        "--engines", default=None, help="Comma-separated engine names (e.g. 'fake', 'fast_alpr')"
    )
    classify_parser.add_argument("--csv-out", default=None, help="Output path for the results CSV")
    classify_parser.add_argument(
        "--annotated-out", default=None, help="Output directory for annotated images"
    )

    args = parser.parse_args(argv)
    if args.command == "classify":
        return _run_classify(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
