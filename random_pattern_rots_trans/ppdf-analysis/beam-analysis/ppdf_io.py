import os

from torch import Tensor
from torch import empty as empty_tensor
from torch import float32, tensor


def load_ppdfs_data_from_hdf5(
    dataset_dir: str, hdf5_filename: str, fov_dict: dict
) -> Tensor:
    import h5py

    # Check if the file exists
    if not os.path.exists(os.path.join(dataset_dir, hdf5_filename)):
        print(f"File {hdf5_filename} does not exist in {dataset_dir}.")
        raise FileNotFoundError(
            f"File {hdf5_filename} does not exist in {dataset_dir}."
        )
    ppdfs = empty_tensor(
        (fov_dict["n pixels"][0], fov_dict["n pixels"][1], 2), dtype=float32
    )
    with h5py.File(os.path.join(dataset_dir, hdf5_filename), "r") as f:
        # Print the names of all datasets in the file
        # print("Datasets in the file:")
        #  for name in f:
        #      print(name)
        ppdfs = f["ppdfs"][:]
        ppdfs = tensor(ppdfs)
    return ppdfs
