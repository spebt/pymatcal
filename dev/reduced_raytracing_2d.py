import torch
import numpy as np
import time
from rich.progress import Progress
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


def main(idx):

    torch.set_default_device("cpu")
    # Load the geometry data
    geom_data = np.load("detector_cuboids.npz")
    plate_geoms = geom_data["plate cuboids"]
    xtal_geoms = geom_data["crystal cuboids"]

    # Get the vertices of the cuboids
    xtal_geoms_verts_2d = get_verts_2d_from_cuboids(xtal_geoms)
    plate_geoms_verts_2d = get_verts_2d_from_cuboids(plate_geoms)

    # Define the FOV
    fov_n_pixels_tensor = torch.tensor([4, 4])
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
        idx + plate_geoms_verts_2d.shape[0],
        geoms_edges_2d,
        edges_indices,
        pixel_corners,
        pb_tensor,
    )

    rays = get_rays_2d(pa_tensor, pb_tensor.unsqueeze(0)).squeeze(1)
    rays_lengths = torch.norm(rays[:, 1] - rays[:, 0], dim=1)

    edges_self = get_edges_from_verts_2d(xtal_geoms_verts_2d[idx].unsqueeze(0))
    ts_absorb, ts_absorb_index = get_cuts_ray_on_edges_2d(
        rays, edges_self.view(-1, 2, 2)
    )

    dl_absorb = rays_lengths * (1.0 - ts_absorb)

    ts_attenu, ts_attenu_index = get_cuts_ray_on_edges_2d(
        rays, reduced_edges.view(-1, 2, 2)
    )
    angular_terms = get_angular_terms_2d(
        rays, rays_lengths, edges_self.view(-1, 2, 2)[:2]
    )
    # print(ts_attenu, ts_attenu_index)
    print(reduced_edges.shape)
    print(geom_indices)
    print(ts_absorb.shape)
    print(ts_attenu.shape)
    # print(ts_attenu_index)
    ts_attenu, _ = torch.sort(
        ts_attenu[torch.sort(ts_attenu_index[:, 0])[1]].reshape(-1, 2), dim=1
    )
    ts_attenu_diff = ts_attenu[:, 1] - ts_attenu[:, 0]

    dl_attenu = (
        rays_lengths[torch.sort(torch.unique(ts_attenu_index[:, 0]))[1]]
        * ts_attenu_diff
    )
    print(dl_attenu)


if __name__ == "__main__":
    main(0)
