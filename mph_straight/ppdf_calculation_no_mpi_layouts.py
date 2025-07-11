# import torch
from torch import empty as empty_tensor, tensor, zeros, Tensor
import time
import h5py


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
    # load_scanner_geometry_npz,
    set_default_device_as_cpu,
    get_geom_dict,
    get_ppdf,
)


if __name__ == "__main__":
    import sys, os

    # try:
    #     idx_start = int(sys.argv[1])
    #     idx_end = int(sys.argv[2])
    # except Exception as e:
    #     print("Requires two arguments: idx_start and idx_end")
    #     sys.exit(1)

    fov_dict = {
        "n_pixels": tensor([256, 256]),
        "mm_per_pixel": tensor([0.125, 0.125]),
        "center": tensor([0.0, 0.0]),
    }
    # Load the scanner layouts
    filename = "../data/scanner_layouts/mph_hourglass_configuration.tensor"
    scanner_layouts_data, filename_unique_id = load_scanner_layouts(filename)

    n_positions = len(scanner_layouts_data)
    # n_positions = 2
    print(f"Number of positions: {n_positions}")

    output_hdf5_dir = f"../data/outputs/"
    if not os.path.exists(output_hdf5_dir):
        os.makedirs(output_hdf5_dir)

    for layout_idx in range(n_positions):

        print(f"Evaluating position {layout_idx:03d} ...")
        # Load the scanner geometry
        plate_verts_2d = scanner_layouts_data[f"position {layout_idx:03d}"][
            "plate segments"
        ].to("cpu")
        xtal_verts_2d = scanner_layouts_data[f"position {layout_idx:03d}"][
            "detector units"
        ].to("cpu")

        n_xtals = xtal_verts_2d.shape[0]
        # n_xtals = 500
        print("Total detectors: ", n_xtals)
        geom_dict = get_geom_dict(plate_verts_2d, xtal_verts_2d, fov_dict)

        fov_n_pixels = int(fov_dict["n_pixels"].prod())
        ppdf = empty_tensor(0, fov_n_pixels)

        elapsed_times = zeros(n_xtals)

        output_hdf5_filename = f"position_{layout_idx:03d}_ppdfs.hdf5"

        out_h5file = h5py.File(
            f"{output_hdf5_dir:s}/position_{layout_idx:03d}_ppdfs.hdf5",
            "w",
        )
        ppdf = out_h5file.create_dataset(
            "ppdfs", shape=(n_xtals, fov_n_pixels), dtype="f"
        )
        for idx in range(n_xtals):
            # progress.console.print(f"Current idx: {idx}")

            start_time = time.time()
            try:
                ppdf[idx] = get_ppdf(idx, geom_dict=geom_dict).unsqueeze(0).numpy()

            except Exception as e:
                print("ID", idx, "\nError:", e)
                sys.exit(1)
            end_time = time.time()
            elapsed_times[idx] = end_time - start_time

        # close the HDF5 file
        out_h5file.close()

        print(f"Average time per iteration: {elapsed_times.mean():.4f} seconds")
