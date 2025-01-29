import numpy as np
import torch
import yaml
from rich.progress import Progress, BarColumn
import os
from ray3d_torch import (
    get_cuboids,
    translate_cuboids,
    get_translation_vectors_xy,
)

d_cpu = torch.device("cpu")
n_panels = 6
panel_interval_rad = 2 * np.pi / n_panels
rshift = 93.0
compute_device = d_cpu
config = yaml.safe_load(open("shifted_layer_3x3_114x114.yaml"))
geoms = torch.tensor(
    config["detector"]["detector geometry"], device=compute_device, dtype=torch.float32
)
plate_geoms = geoms[geoms[:, 6] == 0]
detector_geoms = geoms[geoms[:, 6] != 0]
n_rotations = 24
rotations_start_rads = (
    torch.arange(0, n_rotations, device=compute_device, dtype=torch.float32)
    * panel_interval_rad
    / n_rotations
)

translation_vectors_xy = get_translation_vectors_xy([2, 2], [2, 2]).view(-1, 2)
n_translations = translation_vectors_xy.shape[0]
rotations_rads = rotations_start_rads.repeat_interleave(n_translations).view(-1, 1)
translation_vectors_xy = translation_vectors_xy.repeat(n_rotations, 1)
print(rotations_rads.shape)
print(translation_vectors_xy.shape)
transform_vectors = torch.cat((rotations_rads, translation_vectors_xy), dim=1)
print(transform_vectors.shape)
progress_bar = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    "{task.completed}/{task.total}",
    # auto_refresh=False,
)
task_1 = progress_bar.add_task("Processing", total=transform_vectors.shape[0])
with progress_bar:
    for itrans, vtrans in enumerate(transform_vectors):
        if itrans > 27:
            continue
        #     print(vtrans[0].float() * 180 / np.pi, vtrans[1], vtrans[2])
        rot = vtrans[0].float()
        plate_cuboids_rots_xyz = torch.arange(
            0, n_panels, device=compute_device, dtype=torch.float32
        ).unsqueeze(-1).unsqueeze(-1).repeat(1, plate_geoms.shape[0], 3) * torch.tensor(
            [0, 0, panel_interval_rad],
            device=compute_device,
            dtype=torch.float32,
        ) + torch.tensor(
            [0, 0, rot], device=compute_device, dtype=torch.float32
        )
        plate_cuboids_centers_xyz, plate_cuboids_vectors_xyz = get_cuboids(
            plate_geoms, plate_cuboids_rots_xyz, rshift, compute_device
        )
        translate_xyz = torch.tensor([vtrans[0], vtrans[1], 0], device=compute_device)
        plate_cuboids_centers_xyz = plate_cuboids_centers_xyz + translate_xyz
        plate_cuboids = torch.cat(
            (plate_cuboids_centers_xyz.unsqueeze(2), plate_cuboids_vectors_xyz),
            dim=2,
        )

        detector_cuboids_rots_xyz = torch.arange(
            0, n_panels, device=compute_device, dtype=torch.float32
        ).unsqueeze(-1).unsqueeze(-1).repeat(
            1, detector_geoms.shape[0], 3
        ) * torch.tensor(
            [0, 0, panel_interval_rad], device=compute_device, dtype=torch.float32
        ) + torch.tensor(
            [0, 0, rot], device=compute_device, dtype=torch.float32
        )

        detector_cuboids_centers_xyz, detector_cuboids_vectors_xyz = get_cuboids(
            detector_geoms, detector_cuboids_rots_xyz, rshift, compute_device
        )
        detector_cuboids_centers_xyz = detector_cuboids_centers_xyz + translate_xyz
        detector_cuboids = torch.cat(
            (
                detector_cuboids_centers_xyz.unsqueeze(2),
                detector_cuboids_vectors_xyz,
            ),
            dim=2,
        )
        output = {
            "plate cuboids": plate_cuboids.cpu().numpy(),
            "crystal cuboids": detector_cuboids.cpu().numpy(),
        }
        datadir = "scanner_cuboids_data"
        os.makedirs(datadir, exist_ok=True)
        np.savez_compressed(
            f"{datadir:s}/detector_cuboids_{itrans:03d}.npz", **output, dtype=np.float32
        )
        del (
            plate_cuboids_rots_xyz,
            plate_cuboids_centers_xyz,
            plate_cuboids_vectors_xyz,
            plate_cuboids,
            detector_cuboids_rots_xyz,
            detector_cuboids_centers_xyz,
            detector_cuboids_vectors_xyz,
            detector_cuboids,
            output,
        )

        progress_bar.update(task_1, advance=1)
