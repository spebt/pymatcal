import torch
import numpy as np
import time
from rich.progress import Progress
import pandas as pd

from raytracer_2d import (
    get_fov_pixel_centers_2d,
    get_rays_2d,
    get_convex_hull_2d,
    get_furthest_corners,
    get_verts_sorted_by_angel_2d,
)


def if_rects_in_hull_2d(
    geoms_verts_2d: list[torch.Tensor], hull
) -> torch.Tensor:
    n_hull = hull.shape[0]
    out = []
    for i_geom, geom_verts in enumerate(geoms_verts_2d):
        n_verts = geom_verts.shape[0]
        p0_tensor = hull.unsqueeze(0).expand(n_verts, n_hull, 2)
        p1_tensor = (
            torch.vstack([hull[1:], hull[0]])
            .unsqueeze(0)
            .expand(n_verts, n_hull, 2)
        )
        p2_tensor = geom_verts.unsqueeze(1).expand(n_verts, n_hull, 2)
        v1 = p1_tensor - p0_tensor
        v2 = p2_tensor - p0_tensor
        cross = torch.sign(
            v2[:, :, 0] * v1[:, :, 1] - v2[:, :, 1] * v1[:, :, 0]
        ).view(n_verts, n_hull)
        if torch.logical_or(
            (cross >= 0).all(dim=1), (cross <= 0).all(dim=1)
        ).all():
            out.append(i_geom)
    return torch.tensor(out)


def if_rects_intersect_hull_2d(
    geoms_edges_2d, edges_indices, hull, xtal_center
) -> torch.Tensor:
    verts, _ = get_verts_sorted_by_angel_2d(hull, xtal_center)
    rays = get_rays_2d(verts[[1, -1]], xtal_center.view(1, 2)).view(-1, 2, 2)
    _, index = get_cuts_ray_on_edges_2d(rays, geoms_edges_2d)
    return torch.unique(edges_indices[index[:, 1]][:, 0])


def get_cuts_ray_on_edges_2d(
    rays: torch.Tensor,
    edges: torch.Tensor,
):
    """
    Cut rays with line segments (edges)
    """
    # rays shape (number rays, 2, 2)
    # `number of end points` is always 1
    # edges shape (n_edges, 2, 2)
    n_rays = rays.shape[0]
    n_edges = edges.shape[0]
    v1 = (rays[:, 1] - rays[:, 0]).unsqueeze(1).expand(-1, n_edges, -1)
    v2 = (edges[:, 0] - edges[:, 1]).unsqueeze(0).expand(n_rays, -1, -1)
    v3 = edges[:, 0].unsqueeze(0).expand(n_rays, -1, -1) - rays[:, 0].view(
        n_rays, 1, 2
    ).expand(-1, n_edges, -1)

    # cramer's rule
    # v1, v2, v3 shape (n_pa, n_rects, 4, 2)
    # det shape (n_pa, n_rects, 4)
    det = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
    t = torch.where(
        det != 0,
        (v3[:, :, 0] * v2[:, :, 1] - v2[:, :, 0] * v3[:, :, 1]) / det,
        float("nan"),
    )
    s = torch.where(
        det != 0,
        (v1[:, :, 0] * v3[:, :, 1] - v1[:, :, 1] * v3[:, :, 0]) / det,
        float("nan"),
    )
    t = torch.where((s <= 1) * (s >= 0) * (t <= 1) * (t >= 0), t, float("nan"))
    index = torch.argwhere(~torch.isnan(t))
    return t, index


def get_verts_2d_from_cuboids(cuboids):
    cuboids = torch.tensor(cuboids).reshape(-1, 4, 3)
    return torch.stack(
        [
            cuboids[:, 0, :2]
            - 0.5 * cuboids[:, 1, :2]
            - 0.5 * cuboids[:, 2, :2],
            cuboids[:, 0, :2]
            - 0.5 * cuboids[:, 1, :2]
            + 0.5 * cuboids[:, 2, :2],
            cuboids[:, 0, :2]
            + 0.5 * cuboids[:, 1, :2]
            + 0.5 * cuboids[:, 2, :2],
            cuboids[:, 0, :2]
            + 0.5 * cuboids[:, 1, :2]
            - 0.5 * cuboids[:, 2, :2],
        ],
        dim=1,
    )


def get_edges_from_verts_2d(
    geoms_verts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((0, 2, 2), dtype=torch.float32)
    index = torch.empty((0, 2), dtype=torch.int64)
    for i_geom, geom_verts in enumerate(geoms_verts):
        out = torch.vstack(
            [
                out,
                torch.stack(
                    [
                        geom_verts,
                        torch.vstack(
                            [geom_verts[1:], geom_verts[0].view(1, 2)]
                        ),
                    ],
                    dim=1,
                ),
            ]
        )
        index = torch.vstack(
            [
                index,
                torch.stack(
                    [
                        torch.ones(geom_verts.shape[0], dtype=torch.int64)
                        * i_geom,
                        torch.arange(geom_verts.shape[0]),
                    ],
                    dim=1,
                ),
            ]
        )
    return out, index


if __name__ == "__main__":
    # Load the cuboid data
    geom_data = np.load("detector_cuboids.npz")
    plate_geoms = geom_data["plate cuboids"]
    xtal_geoms = geom_data["crystal cuboids"]

    # Get the vertices of the cuboids
    xtal_geoms_verts_2d = get_verts_2d_from_cuboids(xtal_geoms)
    plate_geoms_verts_2d = get_verts_2d_from_cuboids(plate_geoms)

    # Define the FOV
    fov_n_pixels_tensor = torch.tensor([32, 32])
    fov_mm_per_pixel_tensor = torch.tensor([4, 4])
    fov_dim = fov_n_pixels_tensor * fov_mm_per_pixel_tensor
    fov_center = torch.tensor([0.0, 0.0])

    pa_tensor = get_fov_pixel_centers_2d(
        fov_n_pixels_tensor, fov_mm_per_pixel_tensor, fov_center
    )
    pb_tensor = xtal_geoms_verts_2d.mean(dim=1)

    geoms_verts_2d = []
    for geom in plate_geoms_verts_2d:
        geoms_verts_2d.append(geom)
    for geom in xtal_geoms_verts_2d:
        geoms_verts_2d.append(geom)

    geoms_edges_2d, edge_indices = get_edges_from_verts_2d(geoms_verts_2d)
    pixel_corners = get_furthest_corners(pa_tensor.view(-1, 2))
    # insid_indices = []
    # intsc_indices = []
    # hulls = []
    data = []
    i_list = torch.arange(xtal_geoms_verts_2d.shape[0])
    exec_times = torch.zeros(i_list.shape[0])
    with Progress() as progress:
      task = progress.add_task("[red]Calculating...", total=i_list.shape[0])
      for i in i_list:
        start_time = time.time()
        hull = get_convex_hull_2d(torch.vstack((pb_tensor[i], pixel_corners)))
        local_intsc_indices = if_rects_intersect_hull_2d(
          geoms_edges_2d, edge_indices, hull, pb_tensor[i]
        )
        local_intsc_indices = local_intsc_indices[~local_intsc_indices.eq(i)]
        local_insid_indeces = if_rects_in_hull_2d(geoms_verts_2d, hull)
        
        local_indices = torch.unique(
          torch.cat((local_intsc_indices, local_insid_indeces))
        )
        data.append([i.item(), hull.tolist(), local_indices.tolist()])
        end_time = time.time()
        exec_times[i] = end_time - start_time
        progress.update(task, advance=1)

    df = pd.DataFrame(data, columns=["crystal index","hull", "indices"])

    # Save the data
    df.to_csv("data_for_loop.csv", index=False)

    print("Iteration exection time mean:",exec_times.mean())
    print("Iteration exection time STD :",exec_times.std())
    print("Iteration exection time MAX :",exec_times.max())
    print("Iteration exection time MIN:",exec_times.min())
    print("Iteration exection time SUM:",exec_times.sum())