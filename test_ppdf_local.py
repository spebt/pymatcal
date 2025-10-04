import os

import h5py
from torch import device
from torch import float64 as torch_float64
from torch import get_num_threads, tensor, zeros, tensor, arange

from scanner_modeling._raytracer_2d._local_functions import (
    ppdf_2d_local,
    reduced_edges_2d_local,
    sfov_properties,
    subdivision_grid_rectangle,
)
from scanner_modeling.geometry_2d import (
    fov_tensor_dict,
    load_scanner_geometry_from_layout,
    load_scanner_layouts,
)


if __name__ == "__main__":
    import time
    default_device = device("cpu")
    scanner_layout_file_relative_path = (
        "../data/scanner_layouts/mph_hourglass_multi_position.tensor"
    )
    # Get the dir and filename from the relative path
    scanner_layout_dir = os.path.dirname(scanner_layout_file_relative_path)
    scanner_layout_filename = os.path.basename(scanner_layout_file_relative_path)

    scanner_layouts, layouts_md5 = load_scanner_layouts(
        scanner_layout_dir,
        scanner_layout_filename,
    )

    # mu_dict = {"plate": 3.5, "crystal": 0.475}  # mm^-1
    mu_dict = tensor([3.5, 0.475], device=default_device)

    fov_dict = fov_tensor_dict((512, 512), (128, 128), (0.0, 0.0), (3, 3))

    crystal_n_subs = (3, 3)
    sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
    fov_n_pxs = int(fov_dict["n pixels"].prod())

    n_sfov = int(fov_dict["n subdivisions"].prod())
    sfov_n_pxs = fov_n_pxs / int(fov_dict["n subdivisions"].prod())

    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )

    print(
        f"PyTorch is set to use {get_num_threads()} threads for intra-op parallelism."
    )

    subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

    layout_idx = 0
    # Create h5py file to store the results
    h5_file_path = f"scanner_layouts_{layouts_md5}_layout_{layout_idx}.hdf5"
    h5file = h5py.File(h5_file_path, "w")

    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    # Get the total number of crystals from the shape of the loaded tensor
    n_crystals_total = crystal_objects_vertices.shape[0]
    print(f"Found {n_crystals_total} crystals in layout {layout_idx}.")
    
    # Create a tensor containing all indices from 0 to n_crystals_total - 1
    crystal_idx_tensor = arange(n_crystals_total)

    n_crystals = int(crystal_idx_tensor.shape[0])

    # Create the dataset in the h5py file
    ppdf_dataset = h5file.create_dataset("ppdfs", (n_crystals, fov_n_pxs), dtype="f")

    time_s = time.time()
    for dataset_idx, crystal_idx in enumerate(crystal_idx_tensor):
        crystal_idx = int(crystal_idx.item())
        reduced_crystal_edges_sfovs = []
        reduced_plate_edges_sfovs = []
        for sfov_idx in range(n_sfov):
            reduced_plate_edges, reduced_crystal_edges = reduced_edges_2d_local(
                sfov_idx,
                crystal_idx,
                sfov_corners_batch,
                plate_objects_vertices,
                plate_objects_edges,
                crystal_objects_vertices,
                crystal_objects_edges,
                default_device,
            )
            reduced_crystal_edges_sfovs.append(reduced_crystal_edges)
            reduced_plate_edges_sfovs.append(reduced_plate_edges)

        for sfov_idx in range(n_sfov):
            ppdf_dataset[dataset_idx, sfov_pxs_ids_1d[sfov_idx]] = (
                ppdf_2d_local(
                    sfov_idx,
                    crystal_idx,
                    sfov_pixels_batch,
                    crystal_objects_vertices,
                    reduced_plate_edges_sfovs[sfov_idx],
                    reduced_crystal_edges_sfovs[sfov_idx],
                    subdivision_grid,
                    mu_dict,
                    default_device,
                )
                .cpu()
                .numpy()
            )
        del reduced_crystal_edges_sfovs, reduced_plate_edges_sfovs
    h5file.close()
    time_e = time.time()
    print(
        f"Computed {n_crystals} ppdfs in {time_e - time_s:.2f} seconds "
        f"({(time_e - time_s)/n_crystals :.2f} seconds per ppdf)."
    )
    print(f"Results saved to {h5_file_path}.")
