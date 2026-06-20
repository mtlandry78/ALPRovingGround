import tkinter as tk
from tkinter import filedialog

from alprg.harness import MultiEngineHarness
from alprg.io import iter_image_paths, save_annotated_outputs, write_results_csv

# --- Directory Selection Dialog ---
# We don't need the main tkinter window, so we hide it.
root = tk.Tk()
root.withdraw()

# Open a dialog to ask the user to select a directory.
directory_path = filedialog.askdirectory(title="Select a Folder of Images for ALPR")

# --- ALPR Processing ---
# Proceed only if the user selected a directory.
if directory_path:
    print(f"Directory selected: {directory_path}")
    print("Initializing ALPR models... This might take a moment.")

    # Load every available engine once before the loop, for efficiency.
    with MultiEngineHarness(verbose=True) as harness:
        print(f"Using engines: {', '.join(harness.engine_names)}")

        image_paths = list(iter_image_paths(directory_path))
        if not image_paths:
            print("No image files (.png, .jpg, .jpeg) found in the selected directory.")
        else:
            print(
                f"Processing {len(image_paths)} images. Results will be saved to {directory_path}"
            )
            for i, path in enumerate(image_paths):
                print(f"  ({i + 1}/{len(image_paths)}) Processing: {path.name}")

            results = harness.classify_batch(image_paths)

            csv_path = write_results_csv(results, directory_path)
            output_dir = save_annotated_outputs(results, directory_path)

            print("\nProcessing complete!")
            print(f"CSV data saved to: {csv_path}")
            print(f"Annotated images saved in: {output_dir}")
else:
    # This message is shown if the user closes the dialog without selecting a folder.
    print("No directory was selected. Exiting program.")
