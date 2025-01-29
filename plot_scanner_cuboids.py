import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
import os
import torch
from rich.progress import Progress, BarColumn


def read_cuboids(filename):
    with np.load(filename) as f:
        crystal_cuboids = f["crystal cuboids"]
        plate_cuboids = f["plate cuboids"]
    return plate_cuboids, crystal_cuboids


def get_mpl_rects(xtal_cuboids, plate_cuboids):
    xtal_rects = []
    plate_rects = []
    drawcode = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    for cuboid in xtal_cuboids.reshape(-1, 4, 3):
        verts = [
            cuboid[0, :2] - 0.5 * cuboid[1, :2] - 0.5 * cuboid[2, :2],
            cuboid[0, :2] - 0.5 * cuboid[1, :2] + 0.5 * cuboid[2, :2],
            cuboid[0, :2] + 0.5 * cuboid[1, :2] + 0.5 * cuboid[2, :2],
            cuboid[0, :2] + 0.5 * cuboid[1, :2] - 0.5 * cuboid[2, :2],
            cuboid[0, :2] - 0.5 * cuboid[1, :2] - 0.5 * cuboid[2, :2],
        ]
        xtal_rects.append(patches.PathPatch(Path(verts, drawcode, closed=True)))

    for cuboid in plate_cuboids.reshape(-1, 4, 3):
        verts = [
            cuboid[0, :2] - 0.5 * cuboid[1, :2] - 0.5 * cuboid[2, :2],
            cuboid[0, :2] - 0.5 * cuboid[1, :2] + 0.5 * cuboid[2, :2],
            cuboid[0, :2] + 0.5 * cuboid[1, :2] + 0.5 * cuboid[2, :2],
            cuboid[0, :2] + 0.5 * cuboid[1, :2] - 0.5 * cuboid[2, :2],
            cuboid[0, :2] - 0.5 * cuboid[1, :2] - 0.5 * cuboid[2, :2],
        ]
        plate_rects.append(patches.PathPatch(Path(verts, drawcode, closed=True)))
    return xtal_rects, plate_rects


def rotate_points(points, angle):
    rot_matrix = torch.tensor(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return torch.mm(rot_matrix, points.t()).t()


if __name__ == "__main__":
    progress_bar = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "{task.completed}/{task.total}",
        # auto_refresh=False,
    )
    n_fov_vxs = [128, 128, 1]
    mmpvx = [1, 1, 1]
    fov_xy_offset = [-n_fov_vxs[0] * mmpvx[0] / 2, -n_fov_vxs[1] * mmpvx[1] / 2]
    # scanned_fov_centers = []
    task_1 = progress_bar.add_task("Processing", total=28)
    with progress_bar:
        for itrans in range(28):
            fig = plt.figure(figsize=(12, 5), dpi=600)
            ax = fig.add_subplot(111)
            # deg = irot * 2.5
            # scanned_fov_centers.append([trans_xy[0], trans_xy[1]])
            datadir = "scanner_cuboids_data"
            datafname = f"{datadir:s}/detector_cuboids_{itrans:03d}.npz"
            plate_cuboids, xtal_cuboids = read_cuboids(datafname)
            xtal_rects, plate_rects = get_mpl_rects(xtal_cuboids, plate_cuboids)
            ax.add_collection(PatchCollection(xtal_rects, fc="orange"))
            ax.add_collection(PatchCollection(plate_rects, fc="darkgray"))
            ax.add_patch(
                patches.Rectangle(
                    fov_xy_offset,
                    n_fov_vxs[0] * mmpvx[0],
                    n_fov_vxs[1] * mmpvx[1],
                    fill=False,
                    ec="red",
                )
            )
            points = torch.tensor(
                [
                    [0, 320],
                    [0, -320],
                    [320, 0],
                    [-320, 0],
                ],
                dtype=torch.float64,
            )
            # points = rotate_points(points, np.deg2rad(deg))
            # ax.plot(
            #     points[:2, 0] + trans_xy[0], points[:2, 1] + trans_xy[1], "g--.", lw=0.5
            # )
            # ax.plot(
            #     points[2:, 0] + trans_xy[0], points[2:, 1] + trans_xy[1], "g--.", lw=0.5
            # )
            # sfc = np.array(scanned_fov_centers)
            # print(sfc.shape)
            # print(sfc)
            # ax.scatter(sfc[:, 0], sfc[:, 1], c="red", s=2, marker="o")
            ax.set_xlim(-160, 160)
            ax.set_ylim(-160, 160)
            ax.set_title(f"Scanner Position ID: {itrans:03d}")
            ax.set_aspect("equal")
            os.makedirs(f"images/all", exist_ok=True)
            plt.savefig(f"images/all/trans_{itrans:03d}.png")
            plt.close(fig)
            progress_bar.update(task_1, advance=1)
