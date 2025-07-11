import numpy as np
import torch
import yaml
from rich.progress import Progress, BarColumn
import os

def transform_points_batch(points, rots_xyz, device):
    # points: Nx3
    # rots_xyz: Nx3
    # device: torch.device
    # return: Nx3

    # bs = batch size
    bs = points.shape[0]
    cos_xyz = torch.cos(rots_xyz)
    sin_xyz = torch.sin(rots_xyz)
    # create rotation matrix
    r_matrice = torch.empty((bs, 3, 3), device=device)
    r_matrice[:, 0, 0] = cos_xyz[:, 1] * cos_xyz[:, 2]
    r_matrice[:, 0, 1] = cos_xyz[:, 1] * sin_xyz[:, 2]
    r_matrice[:, 0, 2] = -sin_xyz[:, 1]
    r_matrice[:, 1, 0] = (
        sin_xyz[:, 0] * sin_xyz[:, 1] * cos_xyz[:, 2] - cos_xyz[:, 0] * sin_xyz[:, 2]
    )
    r_matrice[:, 1, 1] = (
        sin_xyz[:, 0] * sin_xyz[:, 1] * sin_xyz[:, 2] + cos_xyz[:, 0] * cos_xyz[:, 2]
    )
    r_matrice[:, 1, 2] = sin_xyz[:, 0] * cos_xyz[:, 1]
    r_matrice[:, 2, 0] = (
        cos_xyz[:, 0] * sin_xyz[:, 1] * cos_xyz[:, 2] + sin_xyz[:, 0] * sin_xyz[:, 2]
    )
    r_matrice[:, 2, 1] = (
        cos_xyz[:, 0] * sin_xyz[:, 1] * sin_xyz[:, 2] - sin_xyz[:, 0] * cos_xyz[:, 2]
    )
    r_matrice[:, 2, 2] = cos_xyz[:, 0] * cos_xyz[:, 1]
    # rotate points
    return torch.bmm(r_matrice, points.unsqueeze(-1)).squeeze(-1)

def get_cuboids(geoms, cuboids_rots_xyz, rshift, device):
    panel_00_cuboids_centers_xyz = (geoms[:, :6:2] + geoms[:, 1:6:2]) / 2
    panel_00_cuboids_sizes_xyz = geoms[:, 1:6:2] - geoms[:, :6:2]
    ncuboids_panel_00 = panel_00_cuboids_centers_xyz.shape[0]
    nrots = cuboids_rots_xyz.shape[0]

    panel_00_cuboids_centers_xyz[:, 1] += -panel_00_cuboids_centers_xyz.mean(dim=0)[1]
    panel_00_cuboids_centers_xyz[:, 0] += rshift
    cuboids_centers_xyz = (
        panel_00_cuboids_centers_xyz.unsqueeze(0).repeat(nrots, 1, 1).to(device)
    )

    cuboids_centers_xyz = transform_points_batch(
        cuboids_centers_xyz.reshape(-1, 3), cuboids_rots_xyz.reshape(-1, 3), device
    ).reshape(nrots, -1, 3)
    panel_00_cuboids_vectors_xyz = torch.stack(
        [
            torch.stack(
                [
                    panel_00_cuboids_sizes_xyz[:, 0],
                    torch.zeros(ncuboids_panel_00, device=device),
                    torch.zeros(ncuboids_panel_00, device=device),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.zeros(ncuboids_panel_00, device=device),
                    panel_00_cuboids_sizes_xyz[:, 1],
                    torch.zeros(ncuboids_panel_00, device=device),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.zeros(ncuboids_panel_00, device=device),
                    torch.zeros(ncuboids_panel_00, device=device),
                    panel_00_cuboids_sizes_xyz[:, 0],
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )
    cuboids_vectors_xyz = panel_00_cuboids_vectors_xyz.repeat(nrots, 1, 1)
    cuboids_vectors_xyz = transform_points_batch(
        cuboids_vectors_xyz.reshape(-1, 3),
        cuboids_rots_xyz.unsqueeze(2).repeat(1, 1, 3, 1).reshape(-1, 3),
        device,
    ).reshape(nrots, -1, 3, 3)
    return cuboids_centers_xyz, cuboids_vectors_xyz

def get_translation_vectors_xy(n_translations_xy_oneside, translation_interval_xy):
    """
    Generate a grid of translation vectors in the XY plane.
    This function creates a grid of translation vectors based on the specified
    number of translations on each side and the translation interval for both
    the X and Y directions. The resulting grid is returned as a tensor.
    params:
        n_translations_xy_oneside (tuple of int): A tuple containing the number
            of translations on positive directions for the X and Y directions.
        translation_interval_xy (tuple of float): A tuple containing the
            translation interval for the X and Y directions. Unit is millimeters.
    returns:
        torch.Tensor: A tensor containing the grid of translation vectors in
        the XY plane.
    """

    return torch.stack(
        torch.meshgrid(
            torch.arange(
                -n_translations_xy_oneside[0], n_translations_xy_oneside[0] + 1
            )
            * translation_interval_xy[0],
            torch.arange(
                -n_translations_xy_oneside[1], n_translations_xy_oneside[1] + 1
            )
            * translation_interval_xy[1],
            indexing="ij",
        ),
        dim=-1,
    )

if __name__ == "__main__": 
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

    translation_vectors_xy = get_translation_vectors_xy([0, 0], [0, 0]).view(-1, 2)
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
            # if itrans > 27:
            #     continue
            #     print(vtrans[0].float() * 180 / np.pi, vtrans[1], vtrans[2])
            rot = vtrans[0].float()
            plate_cuboids_rots_xyz = torch.arange(
                0, n_panels, device=compute_device, dtype=torch.float32
            ).unsqueeze(-1).unsqueeze(-1).repeat(1, plate_geoms.shape[0], 3) * torch.tensor(
                [0, 0, panel_interval_rad],
                device=compute_device,
                dtype=torch.float32,
            ) + torch.tensor([0, 0, rot], device=compute_device, dtype=torch.float32)
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
            ) + torch.tensor([0, 0, rot], device=compute_device, dtype=torch.float32)

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
                f"{datadir:s}/scanner_cuboids_{itrans:03d}.npz", **output, dtype=np.float32
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