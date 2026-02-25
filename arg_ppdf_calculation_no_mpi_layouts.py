# import torch
from torch import empty as empty_tensor, tensor, zeros, Tensor
import time
import h5py
import sys, os

def load_scanner_layouts(filename: str):
    from torch import load as torch_load
    import os

    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        raise FileNotFoundError(f"File {filename} does not exist.")
    filename_unique_id = filename.split(".")[0].split("_")[-1]
    scanner_layouts_data = torch_load(filename, weights_only=True)["layouts"]
    return scanner_layouts_data, filename_unique_id


from raytracer_2d import (
    set_default_device_as_cpu,
    get_geom_dict,
    get_ppdf,
)


if __name__ == "__main__":
    # --- MODIFICATION: Get layout_idx from command-line argument ---
    if len(sys.argv) != 2:
        print("Usage: python generate_ppdf_task.py <layout_idx>")
        sys.exit(1)
    try:
        layout_idx = int(sys.argv[1])
    except ValueError:
        print(f"Error: layout_idx must be an integer. Received: {sys.argv[1]}")
        sys.exit(1)
    # --- END MODIFICATION ---

    fov_dict = {
        "n_pixels": tensor([512, 512]),
        "mm_per_pixel": tensor([0.125, 0.125]),
        "center": tensor([0.0, 0.0]),
    }
    # Load the scanner layouts
    filename = "../data/scanner_layouts/mph _hourglass_multi_position.tensor"
    scanner_layouts_data, filename_unique_id = load_scanner_layouts(filename)

    n_positions_available = len(scanner_layouts_data)
    print(f"Total positions available in file: {n_positions_available}")

    # Check if the requested index is valid
    if layout_idx >= n_positions_available:
        print(f"Error: Requested layout_idx {layout_idx} is out of bounds. File only contains {n_positions_available} positions (indices 0 to {n_positions_available-1}).")
        sys.exit(1)

    output_hdf5_dir = f"../data/outputs/"
    if not os.path.exists(output_hdf5_dir):
        os.makedirs(output_hdf5_dir)

    # --- REMOVED THE FOR LOOP ---
    # The script now only processes the single layout_idx passed to it.

    print(f"Evaluating position {layout_idx:03d} ...")
    # Load the scanner geometry for the specific position
    plate_verts_2d = scanner_layouts_data[f"position {layout_idx:03d}"][
        "plate segments"
    ].to("cpu")
    xtal_verts_2d = scanner_layouts_data[f"position {layout_idx:03d}"][
        "detector units"
    ].to("cpu")

    n_xtals = xtal_verts_2d.shape[0]
    geom_dict = get_geom_dict(plate_verts_2d, xtal_verts_2d, fov_dict)

    fov_n_pixels = int(fov_dict["n_pixels"].prod())
    elapsed_times = zeros(n_xtals)

    output_hdf5_filename = f"{output_hdf5_dir:s}/position_{layout_idx:03d}_ppdfs.hdf5"

    with h5py.File(output_hdf5_filename, "w") as out_h5file:
        ppdf_dataset = out_h5file.create_dataset(
            "ppdfs", shape=(n_xtals, fov_n_pixels), dtype="f"
        )
        for idx in range(n_xtals):
            start_time = time.time()
            try:
                ppdf_dataset[idx] = get_ppdf(idx, geom_dict=geom_dict).unsqueeze(0).numpy()
            except Exception as e:
                print(f"Error processing crystal {idx} for layout {layout_idx}: {e}")
                sys.exit(1) # Exit if one crystal fails
            end_time = time.time()
            elapsed_times[idx] = end_time - start_time

    print(f"Finished processing position {layout_idx:03d}.")
    print(f"Average time per crystal: {elapsed_times.mean():.4f} seconds")