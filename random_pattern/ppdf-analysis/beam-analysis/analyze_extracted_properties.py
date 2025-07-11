import torch
import h5py
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

# Get the angular bin boundaries
n_bins = 360
angular_bin_boundaries = torch.arange(n_bins) / 180 * torch.pi
input_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/ppdf-analysis/beam-analysis/output"

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

with Progress(
    "[progress.description]{task.description}",
    SpinnerColumn(),
    BarColumn(),
    TimeElapsedColumn(),
    TextColumn("[progress.percentage]{task.completed}/{task.total}"),
    refresh_per_second=10,
) as progress:
    task = progress.add_task("Processing layouts...", total=24)
    for layout_idx in range(24):
        

        # Initialize the histogram for ASCI map
        asci_histogram = torch.zeros(
            (512 * 512, n_bins),
            dtype=torch.int32,
        )
        # load the beams properties
        beams_properties_hdf5_filename = (
            f"beams_properties_77faff53af5863ca146878c7c496c75e_{layout_idx:02d}.hdf5"
        )
        with h5py.File(os.path.join(input_dir, beams_properties_hdf5_filename), "r") as f:
            # print(f.keys())
            layout_beams_properties = torch.from_numpy(f["beam_properties"][:])  # type: ignore
            beam_properties_header = f["beam_properties"].attrs["Header"]  # type: ignore

        # load the beams masks for the layout
        beams_masks_hdf5_filename = (
            f"beams_masks_77faff53af5863ca146878c7c496c75e_{layout_idx:02d}.hdf5"
        )

        with h5py.File(
            os.path.join(input_dir, beams_masks_hdf5_filename), "r"
        ) as beams_masks_hdf5:
            beams_masks = torch.from_numpy(beams_masks_hdf5["beam_mask"][:])  # type: ignore

        # Digitize the angles
        digitized_angles = torch.bucketize(
            layout_beams_properties[:, 3], angular_bin_boundaries, right=False
        )
        layout_beams_properties = torch.cat(
            (
                layout_beams_properties,
                (digitized_angles - 1).unsqueeze(1).float(),
            ),
            dim=1,
        )

        # Digitize the angles
        digitized_angles = torch.bucketize(
            layout_beams_properties[:, 3], angular_bin_boundaries, right=False
        )
        layout_beams_properties = torch.cat(
            (
                layout_beams_properties,
                (digitized_angles - 1).unsqueeze(1).float(),
            ),
            dim=1,
        )

        layout_beams_properties_filtered = layout_beams_properties[
            torch.isnan(layout_beams_properties[:, 3]) == False
        ]
        layout_beams_properties_filtered = layout_beams_properties_filtered[
            layout_beams_properties_filtered[:, 4] < 4
        ]
        beams_sensentivity_max = layout_beams_properties_filtered[:, 8].max()
        layout_beams_properties_filtered = layout_beams_properties_filtered[
            layout_beams_properties_filtered[:, 8] > beams_sensentivity_max * 0.01
        ]

        # Loop through the beams, get the beam properties
        for beam_props in layout_beams_properties_filtered:
            detector_idx = int(beam_props[1])
            beam_idx = int(beam_props[2])
            angle_bin_idx = int(beam_props[12])
            asci_histogram[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

        # Save the histogram to a file
        asci_histogram_filename = os.path.join(input_dir, f"asci_histogram_{layout_idx:02d}.hdf5")
        with h5py.File(asci_histogram_filename, "w") as f:
            f.create_dataset("asci_histogram", data=asci_histogram.numpy())
        progress.update(task, advance=1)

