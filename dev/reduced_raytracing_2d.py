import torch
import numpy as np
import time

# from rich.progress import Progress
import pandas as pd

from raytracer_2d import (
    get_fov_pixel_centers_2d,
    get_rays_2d,
    get_verts_2d_from_cuboids,
    get_edges_from_verts_2d,
    get_reduced_raytracing_edges_2d,
    get_furthest_corners,
    get_cuts_ray_on_edges_2d,
    get_angular_terms_2d,
)


def get_ppdf(idx):

    torch.set_default_device("cpu")
    # Load the geometry data
    geom_data = np.load("detector_cuboids.npz")
    plate_geoms = geom_data["plate cuboids"]
    xtal_geoms = geom_data["crystal cuboids"]

    # Get the vertices of the cuboids
    xtal_geoms_verts_2d = get_verts_2d_from_cuboids(xtal_geoms)
    plate_geoms_verts_2d = get_verts_2d_from_cuboids(plate_geoms)

    # Get the overall index of the crystal
    xtal_overall_idx = idx + plate_geoms_verts_2d.shape[0]

    # Define the FOV
    fov_n_pixels_tensor = torch.tensor([32, 32])
    fov_mm_per_pixel_tensor = torch.tensor([4, 4])
    # fov_dim = fov_n_pixels_tensor * fov_mm_per_pixel_tensor
    fov_center = torch.tensor([0.0, 0.0])

    # Get the FOV pixel centers
    pa_tensor = get_fov_pixel_centers_2d(
        fov_n_pixels_tensor, fov_mm_per_pixel_tensor, fov_center
    ).view(-1, 2)
    # Get the furthest corners of the pixels
    pixel_corners = get_furthest_corners(pa_tensor)

    # Get the crystal pixel centers
    pb_tensor = xtal_geoms_verts_2d[idx].mean(dim=0)

    # Combine the geometry vertices
    geoms_verts_2d = torch.cat((plate_geoms_verts_2d, xtal_geoms_verts_2d))
    # Get the edges and edge indices of the geoms
    geoms_edges_2d = get_edges_from_verts_2d(geoms_verts_2d)

    # Get reduced raytracing edges
    # Get indices of the edges. Shape is (n_geoms*4, 2)
    edges_indices = torch.stack(
        [
            torch.arange(geoms_edges_2d.shape[0]).repeat_interleave(4),
            torch.arange(4).repeat(geoms_edges_2d.shape[0]),
        ],
        dim=1,
    )
    reduced_edges, geom_indices = get_reduced_raytracing_edges_2d(
        xtal_overall_idx,
        geoms_edges_2d,
        edges_indices,
        pixel_corners,
        pb_tensor,
    )

    rays = get_rays_2d(pa_tensor, pb_tensor.unsqueeze(0)).squeeze(1)

    n_rays = rays.view(-1, 2, 2).shape[0]
    n_geoms = geom_indices.shape[0]

    rays_lengths = torch.norm(rays[:, 1] - rays[:, 0], dim=1)

    # use same linear attenuation coefficient for all crystals
    # use same linear attenuation coefficient for all plates
    # mu_xtal = 0.475/mm
    # mu_plate = 3.5/mm
    mu_tensor = torch.cat(
        [
            torch.tensor([3.5]).repeat(plate_geoms_verts_2d.shape[0]),
            torch.tensor([0.475]).repeat(xtal_geoms_verts_2d.shape[0]),
        ]
    )

    # Get the absorption terms
    edges_self = get_edges_from_verts_2d(xtal_geoms_verts_2d[idx].unsqueeze(0))
    ts_absorb, ts_absorb_index = get_cuts_ray_on_edges_2d(
        rays, edges_self.view(-1, 2, 2)
    )

    dl_absorb = rays_lengths * (1.0 - ts_absorb)
    absorb_terms = 1 - torch.exp(
        -dl_absorb
        * mu_tensor[
            torch.ones_like(dl_absorb, dtype=torch.int32) * xtal_overall_idx
        ]
    )

    # Get the attenuation terms
    ts_attenu, ts_attenu_indices = get_cuts_ray_on_edges_2d(
        rays, reduced_edges.view(-1, 2, 2)
    )
    # print("Shape of ts_attenu", ts_attenu.shape)
    # print("Shape of ts_attenu_indices", ts_attenu_indices.shape)
    ts_attenu_sorted_indices_indices = torch.sort(ts_attenu_indices[:, 0])[1]
    ts_attenu_sorted_indices = ts_attenu_indices[
        ts_attenu_sorted_indices_indices
    ][::2]
    attenu_rays_indices = ts_attenu_sorted_indices[:, 0]
    attenu_geoms_indices = torch.arange(n_geoms).repeat_interleave(4)[
        ts_attenu_sorted_indices[:, 1]
    ]
    ts_attenu_diff = torch.zeros((n_rays, n_geoms))

    ts_attenu_diff[attenu_rays_indices, attenu_geoms_indices] = torch.abs(
        ts_attenu[ts_attenu_sorted_indices_indices].view(-1, 2)[:, 0]
        - ts_attenu[ts_attenu_sorted_indices_indices].view(-1, 2)[:, 1]
    )
    dl_attenu = (
        rays_lengths.unsqueeze(1).expand(n_rays, n_geoms) * ts_attenu_diff
    )
    attenu_terms = torch.exp(
        torch.sum(
            -dl_attenu
            * mu_tensor[geom_indices].unsqueeze(0).expand(n_rays, n_geoms),
            dim=1,
        )
    )

    angular_terms = get_angular_terms_2d(
        rays, rays_lengths, edges_self.view(-1, 2, 2)[:2]
    )
    out = angular_terms * absorb_terms * attenu_terms
    return out


if __name__ == "__main__":
    import sys

    try:
        idx = int(sys.argv[1])
    except Exception as e:
        print(e)
        sys.exit(1)

    ppdf = get_ppdf(idx)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(ppdf.view(32, 32).numpy())
    fig.savefig(f"ppdf_{idx:03d}.png")
