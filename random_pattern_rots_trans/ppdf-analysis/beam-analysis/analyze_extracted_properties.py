# NEW: Import sys for args and glob for file searching
import sys
import glob
import torch
import h5py
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"


if __name__ == "__main__":
    # --- MODIFIED: Use command-line arguments and glob to find files ---
    if len(sys.argv) != 2:
        print("\nUsage: python analyze_extracted_properties.py <path_to_input_directory>")
        print("  - <path_to_input_directory>: Directory containing 'beams_properties_*.hdf5' and 'beams_masks_*.hdf5' files.\n")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    # Find all the beams_properties files in the input directory. The '*' is a wildcard.
    # We sort them to ensure they are processed in order (0, 1, 2, ...).
    property_files = sorted(glob.glob(os.path.join(input_dir, "beams_properties_*.hdf5")))

    if not property_files:
        print(f"Error: No 'beams_properties_*.hdf5' files found in {input_dir}")
        sys.exit(1)

    # Dynamically extract the unique ID from the first file found.
    # This makes the script robust to different simulation runs.
    # e.g., 'beams_properties_77faff..._0000.hdf5' -> '77faff...'
    try:
        base_filename = os.path.basename(property_files[0])
        parts = base_filename.split('_')
        # Assuming filename format is 'beams_properties_UNIQUEID_INDEX.hdf5'
        unique_id = parts[2]
        print(f"Detected unique ID: {unique_id}")
    except IndexError:
        print(f"Error: Could not parse unique ID from filename: {property_files[0]}")
        sys.exit(1)

    n_layouts = len(property_files)
    # --- END MODIFIED SECTION ---

    # Get the angular bin boundaries
    n_bins = 360
    angular_bin_boundaries = torch.arange(n_bins) / 180 * torch.pi

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        refresh_per_second=10,
    ) as progress:
        task = progress.add_task("Creating ASCI histograms...", total=n_layouts)
        
        # --- MODIFIED: Loop over the number of files found ---
        for layout_idx in range(n_layouts):
            
            # Initialize the histogram for ASCI map *for each layout*
            asci_histogram = torch.zeros(
                (512 * 512, n_bins),
                dtype=torch.int32,
            )

            # --- MODIFIED: Dynamically construct filenames with 4-digit padding ---
            # This must match the output format of the extraction scripts.
            beams_properties_hdf5_filename = f"beams_properties_{unique_id}_{layout_idx:04d}.hdf5"
            beams_masks_hdf5_filename = f"beams_masks_{unique_id}_{layout_idx:04d}.hdf5"
            
            props_path = os.path.join(input_dir, beams_properties_hdf5_filename)
            masks_path = os.path.join(input_dir, beams_masks_hdf5_filename)

            # Check if both required files exist before processing
            if not os.path.exists(props_path):
                print(f"Warning: Properties file not found for layout {layout_idx}, skipping: {props_path}")
                progress.update(task, advance=1)
                continue
            if not os.path.exists(masks_path):
                print(f"Warning: Masks file not found for layout {layout_idx}, skipping: {masks_path}")
                progress.update(task, advance=1)
                continue
            # --- END MODIFIED SECTION ---


            # Load the beams properties
            with h5py.File(props_path, "r") as f:
                layout_beams_properties = torch.from_numpy(f["beam_properties"][:])
                # beam_properties_header = f["beam_properties"].attrs["Header"]

            # load the beams masks for the layout
            with h5py.File(masks_path, "r") as beams_masks_hdf5:
                beams_masks = torch.from_numpy(beams_masks_hdf5["beam_mask"][:])

            # If no beams were found in this layout, skip to the next
            if layout_beams_properties.shape[0] == 0:
                print(f"Info: No beams found for layout {layout_idx}. Skipping histogram generation.")
                progress.update(task, advance=1)
                continue

            # Digitize the angles
            # Note: The original script did this twice, which is redundant. Corrected to once.
            digitized_angles = torch.bucketize(
                layout_beams_properties[:, 3], angular_bin_boundaries, right=False
            )
            # The header has 12 items, so the new column will be index 12.
            layout_beams_properties = torch.cat(
                (
                    layout_beams_properties,
                    (digitized_angles - 1).unsqueeze(1).float(),
                ),
                dim=1,
            )

            # Filter out invalid or low-sensitivity beams
            layout_beams_properties_filtered = layout_beams_properties[
                torch.isnan(layout_beams_properties[:, 3]) == False
            ]
            if layout_beams_properties_filtered.shape[0] == 0:
                 progress.update(task, advance=1)
                 continue

            layout_beams_properties_filtered = layout_beams_properties_filtered[
                layout_beams_properties_filtered[:, 4] < 4 # FWHM Tangential
            ]
            if layout_beams_properties_filtered.shape[0] == 0:
                 progress.update(task, advance=1)
                 continue

            # Assuming absolute sensitivity is at index 8 (check beam_property_io.py)
            beams_sensitivity_max = layout_beams_properties_filtered[:, 8].max()
            layout_beams_properties_filtered = layout_beams_properties_filtered[
                layout_beams_properties_filtered[:, 8] > beams_sensitivity_max * 0.01
            ]

            if layout_beams_properties_filtered.shape[0] == 0:
                 progress.update(task, advance=1)
                 continue

            # Loop through the beams, get the beam properties
            for beam_props in layout_beams_properties_filtered:
                detector_idx = int(beam_props[1])
                beam_idx = int(beam_props[2])
                # The digitized angle is now the last column (index 12)
                angle_bin_idx = int(beam_props[12]) 
                
                # Ensure angle bin is within valid range for histogram
                if 0 <= angle_bin_idx < n_bins:
                    # The mask file contains one row per detector.
                    # The mask itself contains the beam_id for each pixel.
                    asci_histogram[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

            # Save the histogram to a file
            # --- MODIFIED: Use 4-digit padding for output filename ---
            asci_histogram_filename = os.path.join(input_dir, f"asci_histogram_{layout_idx:04d}.hdf5")
            with h5py.File(asci_histogram_filename, "w") as f:
                f.create_dataset("asci_histogram", data=asci_histogram.numpy())
            progress.update(task, advance=1)

    print("ASCI histogram generation completed.")