import os
from typing import Dict, Tuple

import h5py
from numpy import array as np_array
from torch import Tensor, arange, cat, stack, tensor


def initialize_beam_properties_hdf5(
    out_hdf5_filename: str,
    out_dir: str = "./",
) -> Tuple[
    h5py.File,
    h5py.Dataset,
]:
    """
    Creates an HDF5 file to store beam properties.

    Parameters
    ----------

    out_hdf5_filename : str
            Unique identifier for the layouts.

    out_dir : int
            Number of pixels in the field of view.

    Returns
    -------

    Tuple[h5py.File, h5py.Dataset]
          (`HDF5 file object`, `beam properties dataset object`)
    """
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    # Check if the output file already exists
    if os.path.exists(os.path.join(out_dir, out_hdf5_filename)):
        raise FileExistsError(
            f"Output file {out_hdf5_filename} already exists in {out_dir}."
        )

    # Create a hdf5 file to store the beams properties

    out_hdf5_file = h5py.File(os.path.join(out_dir, out_hdf5_filename), "w")

    beam_properties_dataset = out_hdf5_file.create_dataset(
        "beam_properties",
        shape=(0, 12),
        maxshape=(
            None,
            12,
        ),  # Maximum shape: (Unlimited, 11) - unlimited rows, 11 columns
        chunks=(
            200,
            12,
        ),  # Chunk shape: (100, 11) - Important for performance!
    )
    beam_properties_dataset.attrs["Description"] = (
        "Beams properties for each detector unit"
    )
    beam_properties_dataset.attrs["Data type"] = "Float"
    header = [
        "scanner position id",  # 1
        "detector unit id",  # 2
        "beam id",  # 3
        "Angle (rad)",  # 4
        "FWHM (mm)",  # 5
        "FWHM Radial (mm)", #6
        "weighted center x (mm)",  # 7
        "weighted center y (mm)",  # 8
        "sensitivity",  # 9
        "relative sensitivity",  # 10
        "number of pixels",  # 11
        "number of coexisting beams",  # 12
    ]
    beam_properties_dataset.attrs["Header"] = np_array(
        header, dtype=h5py.string_dtype(encoding="utf-8")
    )
    return (out_hdf5_file, beam_properties_dataset)


def initialize_beam_masks_hdf5(
    n_pixels: int,
    out_hdf5_filename: str,
    out_dir: str = "./",
) -> Tuple[h5py.File, h5py.Dataset]:

    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, out_hdf5_filename)):
        raise FileExistsError(
            f"Output file {out_hdf5_filename} already exists in {out_dir}."
        )

    out_hdf5_file = h5py.File(os.path.join(out_dir, out_hdf5_filename), "w")

    beam_mask_dataset = out_hdf5_file.create_dataset(
        "beam_mask",
        shape=(0, n_pixels),
        maxshape=(
            None,
            n_pixels,
        ),  # Maximum shape: (Unlimited, n_pixels) - unlimited rows, n_pixels columns
        chunks=(
            20,
            n_pixels,
        ),  # Chunk shape: (100, n_pixels) -  Important for performance!
    )
    beam_mask_dataset.attrs["Description"] = "Beams masks for each detector unit"
    beam_mask_dataset.attrs["Data type"] = "uint16"
    return (out_hdf5_file, beam_mask_dataset)


def append_to_hdf5_dataset(
    hdf5_dataset: h5py.Dataset,
    new_data: Tensor,
) -> None:
    """
    Appends a row of data to an HDF5 dataset.

    Parameters
    ----------
    hdf5_dataset : h5py.Dataset
        The HDF5 dataset to which the row will be appended.
    row_data : Tensor
        The row data to append.
    """
    n_new_rows = int(new_data.shape[0])  # Get the number of new rows
    # Resize the dataset to accommodate the new rows
    hdf5_dataset.resize(hdf5_dataset.shape[0] + n_new_rows, axis=0)
    # Append the new data
    hdf5_dataset[-n_new_rows:] = new_data.numpy()


def stack_beams_properties(
    layout_idx: int,
    detector_unit_idx: int,
    angles: Tensor,
    fwhms: Tensor,
    fwhms_radial: Tensor,
    sizes: Tensor,
    relative_sensitivities: Tensor,
    absolute_sensitivities: Tensor,
    weighted_centers: Tensor,
) -> Tensor:
    """
    Stacks the beams properties into a single tensor.

    Parameters
    ----------
    layout_idx : int
      The index of the layout.
    detector_unit_idx : int
      The index of the detector unit.
    angles : Tensor
      The angles of the beams.
    fwhms : Tensor
      The full width at half maximum (FWHM) of the beams.
    sizes : Tensor
      The sizes of the beams.
    relative_sensitivities : Tensor
      The relative sensitivity of the beams.
    absolute_sensitivities : Tensor
      The absolute sensitivity of the beams.
    weighted_centers : Tensor
      The weighted centers of the beams.

    Returns
    -------
    Tensor
      The stacked beams properties in the following order:
      - `scanner position id`
      - `detector unit id`
      - `beam id`
      - `Angle (rad)`
      - `FWHM (mm)`
      - `weighted center x (mm)`
      - `weighted center y (mm)`
      - `sensitivity`
      - `relative sensitivity`
      - `number of pixels in the beam`
      - `number of coexisting beams`
    """

    n_beams = angles.shape[0]
    beams_various_ids = (
        tensor([layout_idx, detector_unit_idx], dtype=angles.dtype)
        .unsqueeze(0)
        .expand(n_beams, -1)
    )
    beams_various_ids = cat(
        (
            beams_various_ids,
            arange(1, n_beams + 1).unsqueeze(1),
        ),
        dim=1,
    )
    beams_n_coexisting = tensor([[n_beams]]).expand(n_beams, -1)

    stacked_beams_properties = cat(
        (
            beams_various_ids,
            angles.unsqueeze(1),
            fwhms.unsqueeze(1),
            fwhms_radial.unsqueeze(1),
            weighted_centers,
            absolute_sensitivities.unsqueeze(1),
            relative_sensitivities.unsqueeze(1),
            sizes.unsqueeze(1),
            beams_n_coexisting,
        ),
        dim=1,
    )
    return stacked_beams_properties
