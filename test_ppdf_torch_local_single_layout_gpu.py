if __name__ == "__main__":
    import os

    num_cores = os.cpu_count()
    if num_cores is None:
        print("Could not determine the number of CPU cores.")
        num_cores = 4
    print(f"Number of CPU cores: {num_cores}")
    import sys

    # Require input arguments
    # 1st argument: scanner_layout_file_relative_path
    # 2nd argument: layout index
    if len(sys.argv) != 3:
        print(
            "Usage: python test_ppdf_torch_local_single_layout.py <scanner_layout_file_relative_path> <layout_index>"
        )
        sys.exit(1)

    import time

    import h5py
    from torch import device
    from torch import empty as empty_tensor
    from torch import float64 as torch_float64
    from torch import get_num_threads
    from torch import save as torch_save
    from torch import set_default_device, set_num_threads, tensor
    from torch import zeros as zeros_tensor

    # set_default_device("cpu")  # Ensure operations are performed on CPU
    # set_default_device("cuda:0")  # Ensure operations are performed on CUDA
    default_device = device("cuda:0")  # Set the default device to CUDA
    set_default_device(default_device)  # Set the default device for PyTorch
    # set_num_threads(
    #     num_cores
    # )  # Set to a specific number of threads for performance
    # # set_num_threads(2)  # Set to a specific number of threads for performance

    from scanner_modeling.geometry_2d import (
        fov_tensor_dict,
        load_scanner_geometry_from_layout,
        load_scanner_layouts,
    )
    from scanner_modeling.raytracer_2d import (
        ppdf_2d_local,
        sfov_properties,
        subdivision_grid_rectangle,
    )

    scanner_layout_file_relative_path = sys.argv[1]
    # Get the dir and filename from the relative path
    scanner_layout_dir = os.path.dirname(scanner_layout_file_relative_path)
    scanner_layout_filename = os.path.basename(
        scanner_layout_file_relative_path
    )
    layout_idx = int(sys.argv[2])
    start_time = time.time()
    scanner_layouts, layouts_md5 = load_scanner_layouts(
        scanner_layout_dir,
        scanner_layout_filename,
    )

 

    mu_dict = {"plate": 3.5, "crystal": 0.475}  # mm^-1
    mu_dict = tensor(
        [mu_dict["plate"], mu_dict["crystal"]], device=default_device
    )

    fov_dict = fov_tensor_dict((512, 512), (128, 128), (0.0, 0.0), (16, 16))

    crystal_n_subs = (5, 5)
    # crystal_n_subs = (9, 9)
    sfov_pxs_ids, sfov_pxs_coords, sfov_corners_batch = sfov_properties(
        fov_dict
    )
    n_sfov = int(fov_dict["n subdivisions"].prod())
    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )

    ppdf = empty_tensor(int(fov_dict["n pixels"].prod()), dtype=torch_float64)

    print(f"Number of FOV Blocks:               {n_sfov}")
    print(f"FOV Block Size:                     {sfov_pxs_coords.shape[1]}")
    print(f"Number of Subdivisions per Crystal: {crystal_n_subs}")

    print(
        f"PyTorch is set to use {get_num_threads()} threads for intra-op parallelism."
    )

    subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    crystal_idx_list = [
        94,
        # 120,
        # 234,
        # 456,
        # 666,
    ]

    subdivision_grid = subdivision_grid.to(default_device)
    sfov_pxs_ids_1d = sfov_pxs_ids_1d.to("cpu")
    ppdf = ppdf.to(default_device)
    sfov_pxs_coords = sfov_pxs_coords.to(default_device)
    plate_objects_vertices = plate_objects_vertices.to(default_device)
    crystal_objects_vertices = crystal_objects_vertices.to(default_device)
    plate_objects_edges = plate_objects_edges.to(default_device)
    crystal_objects_edges = crystal_objects_edges.to(default_device)
    subdivision_grid = subdivision_grid.to(default_device)
    sfov_corners_batch = sfov_corners_batch.to(default_device)


    output_hdf5_filename = f"ppdf_{layouts_md5}_layout_{layout_idx}.hdf5"
    with h5py.File(output_hdf5_filename, "w") as output_hdf5_file:

        ppdf_dataset = output_hdf5_file.create_dataset(
            "ppdfs",
            shape=(
                len(crystal_idx_list),
                int(fov_dict["n pixels"].prod()),
            ),
            dtype="f",
        )

        for arr_idx, crystal_idx in enumerate(crystal_idx_list):
            print(f"Processing crystal {crystal_idx}...")
            for sfov_idx in range(n_sfov):
                ppdf_data =  ppdf_2d_local(
                        sfov_idx,
                        crystal_idx,
                        sfov_pxs_coords,
                        sfov_corners_batch,
                        plate_objects_vertices,
                        crystal_objects_vertices,
                        plate_objects_edges,
                        crystal_objects_edges,
                        subdivision_grid,
                        mu_dict,
                        device=default_device,
                    ).cpu()
                print(ppdf_data.dtype)
                ppdf_dataset[arr_idx, sfov_pxs_ids_1d[sfov_idx]] = (
                    ppdf_data
                )
            # torch_save(ppdf.cpu(), f"ppdf_{crystal_idx:03}_loop.tensor")

    print(f"PPDF results saved to {output_hdf5_filename}")
    print(
        f"Average time per crystal: {(time.time() - start_time) / len(crystal_idx_list):.2f} seconds"
    )
