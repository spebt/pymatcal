import torch
import numpy as np
import time
from rich.progress import Progress
import pandas as pd
from raytracer_2d import (
    get_fov_pixel_centers_2d,
    get_convex_hull_2d,
    get_verts_2d_from_cuboids,
    get_edges_from_verts_2d,
    if_rects_in_hull_2d,
    if_rects_intersect_hull_2d,
    get_furthest_corners,
)


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

    geoms_verts_2d = torch.cat((plate_geoms_verts_2d, xtal_geoms_verts_2d))

    geoms_edges_2d = get_edges_from_verts_2d(geoms_verts_2d)
    edges_indices = torch.stack(
        [
            torch.arange(geoms_edges_2d.shape[0]).repeat_interleave(4),
            torch.arange(4).repeat(geoms_edges_2d.shape[0]),
        ],
        dim=1,
    )
    pixel_corners = get_furthest_corners(pa_tensor.view(-1, 2))

    hulls = []
    intersection_indices = []
    inclusion_indices = []
    combined_indices = []
    # Loop over the crystal geometries
    i_list = torch.arange(xtal_geoms_verts_2d.shape[0])
    exec_times = torch.zeros(i_list.shape[0])
    with Progress() as progress:
        task = progress.add_task("[red]Calculating...", total=i_list.shape[0])
        for i in i_list:
            start_time = time.time()
            hull = get_convex_hull_2d(
                torch.vstack((pb_tensor[i].view(1, 2), pixel_corners))
            )
            # hull = torch.vstack((pb_tensor[i].view, pixel_corners))
            # geometry indices that intersect with the hull
            local_intersection_indices = if_rects_intersect_hull_2d(
                geoms_edges_2d.view(-1, 2, 2), edges_indices, hull, pb_tensor[i]
            )
            # exclude the current geometry index
            local_intersection_indices = local_intersection_indices[
                ~local_intersection_indices.eq(i)
            ]

            # geometry indices that are inside the hull
            local_inclusion_indices = if_rects_in_hull_2d(geoms_verts_2d, hull)


            # combine the indices
            local_indices = torch.unique(
                torch.cat((local_intersection_indices, local_inclusion_indices))
            )

            hulls.append(
                hull.numpy()
                if hull.shape[0] == 5
                else np.concatenate(
                    (hull.numpy(), np.array(["nan", "nan"]).reshape(1, 2)), axis=0
                )
            )
            intersection_indices.append(local_intersection_indices.numpy())
            inclusion_indices.append(local_inclusion_indices.numpy())
            combined_indices.append(local_indices.numpy())
            end_time = time.time()
            exec_times[i] = end_time - start_time
            progress.update(task, advance=1)

    # Save the data
    np.save("hulls.npy", hulls)
    pd.DataFrame(intersection_indices).to_csv("intersection_indices.csv", index=False)
    pd.DataFrame(inclusion_indices).to_csv("inclusion_indices.csv", index=False)
    pd.DataFrame(combined_indices).to_csv("combined_indices.csv", index=False)
    print("Iteration exection time mean:", exec_times.mean())
    print("Iteration exection time STD :", exec_times.std())
    print("Iteration exection time MAX :", exec_times.max())
    print("Iteration exection time MIN:", exec_times.min())
    print("Iteration exection time SUM:", exec_times.sum())
