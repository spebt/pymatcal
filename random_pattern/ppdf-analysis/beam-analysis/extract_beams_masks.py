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
    scanner_layouts_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/scanner_layouts"
    scanner_layouts_filename = "scanner_layouts_77faff53af5863ca146878c7c496c75e.tensor"

    ppdfs_dataset_dir = (
        "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/scanner_layouts_77faff53af5863ca146878c7c496c75e"
    )

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
    out_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/ppdf-analysis/beam-analysis/output"

    # Define the layout sequence
    layout_sequence = arange(0, 24)

    with progress:
        task_outer = progress.add_task(
            "Processing layouts", total=int(layout_sequence.shape[0])
        )
        # Create the progress bar for the inner loop
        task_inner = progress.add_task(
            "Processing layout 000", total=726, completed=0
        )
        # Loop through the scanner positions (0 to 23)
        for layout_idx in layout_sequence:

            # Initialize the HDF5 file to store the beams masks
            out_hdf5_filename = (
                f"beams_masks_{layouts_unique_id}_{layout_idx:02d}.hdf5"
            )

            out_hdf5_file, beams_masks_dataset = initialize_beam_masks_hdf5(
                fov_n_pixels_int, out_hdf5_filename, out_dir
            )

            # Load the scanner geometry
            plates_vertices, detector_units_vertices = (
                load_scanner_layout_geometries(
                    int(layout_idx), scanner_layouts_data
                )
            )

            # Set the PPDFs filename for a particular scanner position
            ppdfs_hdf5_filename = f"position_{layout_idx:03d}_ppdfs.hdf5"

            # Load the PPDFs data
            ppdfs = load_ppdfs_data_from_hdf5(
                ppdfs_dataset_dir, ppdfs_hdf5_filename, fov_dict
            )

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

            # progress.update(
            #     task_inner,
            #     description=f"Processing layout {layout_idx}",
            #     # total=int(n_detector_units),
            #     completed=0,
            # )
            progress.reset(task_inner, total=int(n_detector_units), start=True)

            # Create a sequence of detector unit indices
            detector_units_sequence = arange(0, n_detector_units)
            # detector_units_sequence = arange(0, 3)

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

                relative_sampled_ppdf = sampled_ppdf / sampled_ppdf.max()
                beams_boundaries = beams_boundaries_radians(
                    sampled_ppdf, sampling_rads, threshold=0.01
                )

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
                # print(f"completed: {detector_unit_idx}:03d")
                progress.update(
                    task_inner,
                    advance=1,
                )

            print(
                f"Layout {layout_idx}:\n"
                + f"\n  beams_properties_dataset shape: {list(beams_masks_dataset.shape)}\n"
            )
            # # Close the HDF5 file
            out_hdf5_file.close()
            # Print the final message
            print(
                f"Beams properties saved in:\n{out_dir}/{out_hdf5_filename}"
            )  #

            progress.update(
                task_outer,
                advance=1,
            )

    print("Processing completed.")
