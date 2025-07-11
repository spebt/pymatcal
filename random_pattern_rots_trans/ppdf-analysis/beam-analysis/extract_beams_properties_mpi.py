# Import 'sys' to handle command-line arguments
import sys
# --- MPI ---
from mpi4py import MPI
import numpy as np
# --- END MPI ---
# rich.progress is removed as it doesn't render well with multiple MPI processes logging to one file
# from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
import torch
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
    get_beams_weighted_center,
    get_beam_width,
    get_beams_angle_radian,
    get_beams_basic_properties,
    sample_ppdf_on_arc_2d_local,
    get_beam_radial_width,
    beam_radial_sampling_line_batch,
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
    append_to_hdf5_dataset,
    stack_beams_properties,
)

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


if __name__ == "__main__":
    # --- MPI ---
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # The rank of the current process (0, 1, 2, ...)
    size = comm.Get_size()  # The total number of processes
    # --- END MPI ---

    if len(sys.argv) != 3:
        # Only rank 0 should print usage instructions to avoid clutter
        if rank == 0:
            print("\nUsage: srun --mpi=pmi2 python extract_beam_properties_mpi.py <path_to_scanner_layouts.tensor> <path_to_ppdfs_directory>")
            print("  - <path_to_scanner_layouts.tensor>: Full path to the layout file (e.g., scanner_layouts_...e.tensor)")
            print("  - <path_to_ppdfs_directory>: Path to the directory containing the position_*_ppdfs.hdf5 files\n")
        sys.exit(1)

    layouts_full_path = sys.argv[1]
    ppdfs_dataset_dir = sys.argv[2]

    # Derive directory and filename from the full path
    scanner_layouts_dir = os.path.dirname(layouts_full_path)
    scanner_layouts_filename = os.path.basename(layouts_full_path)

    # All processes check paths to ensure inputs are valid
    if not os.path.exists(layouts_full_path):
        if rank == 0: raise FileNotFoundError(f"Scanner layout file not found: {layouts_full_path}")
    if not os.path.isdir(ppdfs_dataset_dir):
        if rank == 0: raise NotADirectoryError(f"PPDFs directory not found: {ppdfs_dataset_dir}")

    # Load the scanner layouts (all processes do this, it's a small file)
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


    # --- MPI: Distribute the work ---
    n_layouts = len(scanner_layouts_data)
    all_layout_indices = arange(0, n_layouts)

    # Use numpy.array_split to divide the indices among all processes
    # This handles cases where n_layouts is not perfectly divisible by size
    layouts_for_this_rank = np.array_split(all_layout_indices, size)[rank]

    if rank == 0:
        print(f"Detected {n_layouts} total layouts.")
        print(f"Distributing work across {size} MPI tasks.")

    # A barrier ensures all processes have reached this point before starting the main loop
    comm.Barrier()
    print(f"[Rank {rank:02d}] will process {len(layouts_for_this_rank)} layouts: indices from {layouts_for_this_rank[0]} to {layouts_for_this_rank[-1]}" if len(layouts_for_this_rank)>0 else f"[Rank {rank:02d}] has no layouts to process.")

    start_time = time.time()
    
    # Loop through the subset of layouts assigned to this specific process
    for layout_idx in layouts_for_this_rank:
        
        # --- MODIFIED: Use 4-digit padding for filenames for robustness ---
        out_hdf5_filename = (
            f"beams_properties_{layouts_unique_id}_{int(layout_idx):04d}.hdf5"
        )
        
        out_dir = "output"
        
        # Check if output file already exists to prevent re-running errors
        if os.path.exists(os.path.join(out_dir, out_hdf5_filename)):
            print(f"[Rank {rank:02d}] Output for layout {int(layout_idx):04d} already exists. Skipping.")
            continue
        
        try:
            # Each process creates its own output files independently
            out_hdf5_file, beam_properties_dataset = initialize_beam_properties_hdf5(
                out_hdf5_filename, out_dir
            )
        except FileExistsError as e:
            print(f"[Rank {rank:02d}] Warning: {e}")
            continue

        # Load the scanner geometry
        plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
            int(layout_idx), scanner_layouts_data
        )

        ppdfs_hdf5_filename = f"position_{int(layout_idx):03d}_ppdfs.hdf5"

        try:
            ppdfs = load_ppdfs_data_from_hdf5(
                ppdfs_dataset_dir, ppdfs_hdf5_filename, fov_dict
            )
        except FileNotFoundError:
            print(f"[Rank {rank:02d}] Could not find PPDF file for layout {int(layout_idx):04d}: {ppdfs_hdf5_filename}. Skipping.")
            out_hdf5_file.close()
            os.remove(os.path.join(out_dir, out_hdf5_filename))
            continue

        # --- The rest of the processing loop is identical to your original script ---

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

        n_detector_units = int(detector_units_vertices.shape[0])
        detector_units_sequence = arange(0, n_detector_units)
        
        # We replace the progress bar with simple print statements
        print(f"[Rank {rank:02d}] Processing {n_detector_units} detector units for layout {int(layout_idx):04d}...")

        # Loop through the detector units
        for detector_unit_idx in detector_units_sequence:
            ppdf_data_2d = ppdfs[detector_unit_idx].view(
                int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
            )
            hull_2d = convex_hull_2d(hull_points_batch[detector_unit_idx])

            (sampled_ppdf, sampling_rads, sampling_points) = (
                sample_ppdf_on_arc_2d_local(
                    ppdf_data_2d,
                    detector_unit_centers[detector_unit_idx],
                    hull_2d,
                    fov_dict,
                )
            )

            if sampled_ppdf.max() == 0:
                continue

            relative_sampled_ppdf = sampled_ppdf / sampled_ppdf.max()
            beams_boundaries = beams_boundaries_radians(
                sampled_ppdf, sampling_rads, threshold=0.01
            )

            if beams_boundaries.shape[0] == 0:
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
            beams_weighted_centers = get_beams_weighted_center(
                beams_masks,
                fov_points_xy,
                ppdf_data_2d,
            )
            n_beams = beams_masks.shape[0]

            (
                beams_fwhm,
                x_bounds_batch,
                sampled_beams_data,
                beam_sp_distance,
            ) = get_beam_width(
                beams_weighted_centers,
                detector_unit_centers[detector_unit_idx],
                beams_masks,
                ppdf_data_2d,
                fov_dict,
            )
            
            fov_diag_mm = torch.norm(fov_dict["size in mm"])
            (
                beams_fwhm_radial,
                x_bounds_radial_batch,
                sampled_beams_radial_data,
                beam_sp_distance_radial,
            ) = get_beam_radial_width(
                beams_weighted_centers,
                detector_unit_centers[detector_unit_idx],
                beams_masks,
                ppdf_data_2d,
                fov_dict,
                radial_line_max_length_mm=fov_diag_mm 
            )

            beams_angle = get_beams_angle_radian(
                beams_weighted_centers,
                detector_unit_centers[detector_unit_idx],
            )
            (
                beams_sizes,
                beams_relative_sensitivity,
                beams_absolute_sensitivity,
            ) = get_beams_basic_properties(beams_masks, ppdf_data_2d, fov_points_xy)

            stacked_beams_properties = stack_beams_properties(
                int(layout_idx),
                int(detector_unit_idx),
                angles=beams_angle,
                fwhms=beams_fwhm,
                fwhms_radial=beams_fwhm_radial,
                sizes=beams_sizes,
                relative_sensitivities=beams_relative_sensitivity,
                absolute_sensitivities=beams_absolute_sensitivity,
                weighted_centers=beams_weighted_centers,
            )

            append_to_hdf5_dataset(
                beam_properties_dataset,
                stacked_beams_properties,
            )
        
        out_hdf5_file.close()

        # Print the output filename for this layout
        print(f"[Rank {rank:02d}] Beams properties for layout {int(layout_idx):04d} saved in: {os.path.join(out_dir, out_hdf5_filename)}")

    # --- MPI ---
    # Wait for all processes to finish their loops
    comm.Barrier()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Let rank 0 print the final completion message
    if rank == 0:
        print(f"\nProcessing completed for all layouts in {elapsed_time:.2f} seconds.")
    # --- END MPI ---