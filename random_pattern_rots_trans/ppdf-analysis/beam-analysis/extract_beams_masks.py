# Import 'sys' to handle command-line arguments
import sys
import time
import h5py
import torch
torch.set_num_threads(8)
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from torch import (
    cat,
    tensor,
    arange,
    empty,
    bool as bool_tensor,
    float as float_tensor,
)

from beam_property_extract import (
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_combined_mask,
    sample_ppdf_on_arc_2d_local,
)
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import (
    fov_tensor_dict,
    pixels_coordinates,
    pixels_to_detector_unit_rads,
)
from ppdf_io import load_ppdfs_data_from_hdf5
import h5py
from beam_property_io import (
    initialize_beam_properties_hdf5,
    initialize_beam_masks_hdf5,
    append_to_hdf5_dataset,
    # stack_beams_properties,
)

if __name__ == "__main__":
    # --- MODIFIED: Use command-line arguments for input paths ---
    if len(sys.argv) != 3:
        print("\nUsage: python extract_beams_masks.py <path_to_scanner_layouts.tensor> <path_to_ppdfs_directory>")
        print("  - <path_to_scanner_layouts.tensor>: Full path to the layout file (e.g., scanner_layouts_...e.tensor)")
        print("  - <path_to_ppdfs_directory>: Path to the directory containing the position_*_ppdfs.hdf5 files\n")
        sys.exit(1)

    layouts_full_path = sys.argv[1]
    ppdfs_dataset_dir = sys.argv[2]

    # Derive directory and filename from the full path
    scanner_layouts_dir = os.path.dirname(layouts_full_path)
    scanner_layouts_filename = os.path.basename(layouts_full_path)

    # Check if paths exist
    if not os.path.exists(layouts_full_path):
        raise FileNotFoundError(f"Scanner layout file not found: {layouts_full_path}")
    if not os.path.isdir(ppdfs_dataset_dir):
        raise NotADirectoryError(f"PPDFs directory not found: {ppdfs_dataset_dir}")
    # --- END MODIFIED SECTION ---

    # Load the scanner layouts
    scanner_layouts_data, layouts_unique_id = load_scanner_layouts(
        scanner_layouts_dir, scanner_layouts_filename
    )
    # Load the PPDFs data
    fov_dict = fov_tensor_dict(
        n_pixels=(512, 512),
        mm_per_pixel=(0.25, 0.25),
        center_coordinates=(0.0, 0.0),
    )
    fov_points_2d = pixels_coordinates(fov_dict)
    fov_n_pixels_int = int(fov_dict["n pixels"].prod())

    # Create the progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )

    # Define the output directory
    # Using a fixed path here, but could also be a command-line argument
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)


    # --- MODIFIED: Define the layout sequence dynamically ---
    n_layouts = len(scanner_layouts_data)
    layout_sequence = arange(0, n_layouts)
    print(f"Detected {n_layouts} layouts to process for mask extraction.")
    # --- END MODIFIED SECTION ---

    with progress:
        task_outer = progress.add_task(
            "Processing layouts", total=len(layout_sequence)
        )
        # Create the progress bar for the inner loop (will be reset)
        task_inner = progress.add_task(
            f"Processing layout {int(layout_sequence[0]):04d}", total=1, completed=0
        )
        # Loop through all scanner positions
        for layout_idx in layout_sequence:

            # --- MODIFIED: Use 4-digit padding for filenames ---
            out_hdf5_filename = (
                f"beams_masks_{layouts_unique_id}_{int(layout_idx):04d}.hdf5"
            )
            # --- END MODIFIED SECTION ---

            # Check if output file already exists
            if os.path.exists(os.path.join(out_dir, out_hdf5_filename)):
                print(f"Masks file for layout {layout_idx} already exists. Skipping.")
                progress.update(task_outer, advance=1)
                continue
            
            try:
                # Initialize the HDF5 file to store the beams masks
                out_hdf5_file, beams_masks_dataset = initialize_beam_masks_hdf5(
                    fov_n_pixels_int, out_hdf5_filename, out_dir
                )
            except FileExistsError as e:
                print(e)
                progress.update(task_outer, advance=1)
                continue

            # Load the scanner geometry
            plates_vertices, detector_units_vertices = (
                load_scanner_layout_geometries(
                    int(layout_idx), scanner_layouts_data
                )
            )

            # Set the PPDFs filename for a particular scanner position
            ppdfs_hdf5_filename = f"position_{int(layout_idx):03d}_ppdfs.hdf5"

            # Load the PPDFs data
            try:
                ppdfs = load_ppdfs_data_from_hdf5(
                    ppdfs_dataset_dir, ppdfs_hdf5_filename, fov_dict
                )
            except FileNotFoundError:
                print(f"Could not find PPDF file for layout {layout_idx}: {ppdfs_hdf5_filename}. Skipping.")
                out_hdf5_file.close()
                os.remove(os.path.join(out_dir, out_hdf5_filename)) # Remove empty file
                progress.update(task_outer, advance=1)
                continue


            detector_unit_centers = detector_units_vertices.mean(dim=1)
            fov_corners = (
                tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
                * fov_dict["size in mm"]
                * 0.5
            )

            hull_points_batch = cat(
                (
                    fov_corners.unsqueeze(0).expand(
                        detector_units_vertices.shape[0], -1, -1
                    ),
                    detector_unit_centers.unsqueeze(1),
                ),
                dim=1,
            )
            hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)

            n_detector_units = detector_units_vertices.shape[0]

            progress.reset(
                task_inner,
                total=int(n_detector_units),
                description=f"Processing detector units for layout {int(layout_idx):04d}",
            )

            # Create a sequence of detector unit indices
            detector_units_sequence = arange(0, n_detector_units)

            # Loop through the detector units
            for detector_unit_idx in detector_units_sequence:
                ppdf_data_2d = ppdfs[detector_unit_idx].view(
                    int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
                )
                # Calculate the convex hull for the detector unit
                hull_2d = convex_hull_2d(hull_points_batch[detector_unit_idx])

                # Sample the PPDFs on the arc of the convex hull
                (sampled_ppdf, sampling_rads, sampling_points) = (
                    sample_ppdf_on_arc_2d_local(
                        ppdf_data_2d,
                        detector_unit_centers[detector_unit_idx],
                        hull_2d,
                        fov_dict,
                    )
                )

                if sampled_ppdf.max() == 0: # If PPDF is all zero, no beams
                    progress.update(task_inner, advance=1)
                    continue

                relative_sampled_ppdf = sampled_ppdf / sampled_ppdf.max()
                beams_boundaries = beams_boundaries_radians(
                    sampled_ppdf, sampling_rads, threshold=0.01
                )
                
                if beams_boundaries.shape[0] == 0: # No beams found above threshold
                    progress.update(task_inner, advance=1)
                    continue

                fov_points_xy = pixels_coordinates(fov_dict)
                fov_points_rads = pixels_to_detector_unit_rads(
                    fov_points_xy,
                    detector_unit_centers[detector_unit_idx],
                )
                beams_masks = get_beams_masks(
                    fov_points_rads,
                    beams_boundaries,
                )
                n_beams = beams_masks.shape[0]

                combined_beams_masks = get_beams_combined_mask(beams_masks)

                # Append the beam masks to the HDF5 dataset
                append_to_hdf5_dataset(
                    beams_masks_dataset,
                    combined_beams_masks,
                )
                progress.update(
                    task_inner,
                    advance=1,
                )

            # print(
            #     f"Layout {layout_idx}:\n"
            #     + f"\n  beams_masks_dataset shape: {list(beams_masks_dataset.shape)}\n"
            # )
            # # Close the HDF5 file
            out_hdf5_file.close()
            # Print the final message
            print(
                f"Beams masks saved in:\n{os.path.join(out_dir, out_hdf5_filename)}"
            )

            progress.update(
                task_outer,
                advance=1,
            )

    print("Processing completed.")