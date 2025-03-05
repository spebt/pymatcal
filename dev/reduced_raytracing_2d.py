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


def main(idx):

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

    edges_self = get_edges_from_verts_2d(xtal_geoms_verts_2d[idx].unsqueeze(0))
    ts_absorb, ts_absorb_index = get_cuts_ray_on_edges_2d(
        rays, edges_self.view(-1, 2, 2)
    )

    dl_absorb = rays_lengths * (1.0 - ts_absorb)

    ts_attenu, ts_attenu_index = get_cuts_ray_on_edges_2d(
        rays, reduced_edges.view(-1, 2, 2)
    )
    ts_sort_indices = torch.sort(ts_attenu_index[:, 0])[1]
    ts_attenu, _ = torch.sort(ts_attenu[ts_sort_indices].reshape(-1, 2), dim=1)

    # ts_attenu_diff = ts_attenu[:, 1] - ts_attenu[:, 0]
    ts_attenu_diff = torch.empty((n_rays, n_geoms))


    angular_terms = get_angular_terms_2d(
        rays, rays_lengths, edges_self.view(-1, 2, 2)[:2]
    )

    # dl_attenu = (
    #     rays_lengths[torch.sort((ts_attenu_index[:, 0]))[0][::2]] * ts_attenu_diff
    # )
    dl_attenu = rays_lengths.unsqueeze(1).expand(n_rays, n_geoms) * ts_attenu_diff
    local_edges_indices = edges_indices.view(-1, 4, 2)[geom_indices].view(-1, 2)
    # attenu_geoms_index = local_edges_indices[ts_attenu_index[ts_sort_indices][:, 1]][
    #     ::2, 0
    # ]
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
    # attenu_term = torch.sum(
    #     torch.exp(
    #         -dl_attenu * mu_tensor[geom_indices].unsqueeze(0).expand(n_rays, n_geoms)
    #     ),
    #     dim=1,
    # )
    # print(geom_indices.shape, mu_tensor.shape)
    print(geom_indices)
    absorb_term = torch.exp(
        -dl_absorb
        * mu_tensor[torch.ones_like(dl_absorb, dtype=torch.int32) * xtal_overall_idx]
    )
    print(
        absorb_term.shape,
        # attenu_term.shape,
        angular_terms.shape,
        # attenu_geoms_index.shape,
    )


if __name__ == "__main__":
    main(26)
