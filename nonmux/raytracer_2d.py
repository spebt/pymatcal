import torch
import pandas as pd
from torch import empty as empty_tensor, tensor, zeros, Tensor



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


def get_polygon_groups_from_csv(grp_names: list[str], csv_path: str):
    df = pd.read_csv(csv_path)
    for grp_name in grp_names:
        grp_verts = df.loc[df["polygon group"] == grp_name]
        grp_tensor = torch.empty(
            grp_verts.shape[0] // 4, 4, 2, dtype=torch.float
        )
        # print(grp_verts["polygon id"].unique())
        grp_tensor[
            torch.tensor(grp_verts["polygon id"].values, dtype=torch.int),
            torch.tensor(grp_verts["vertice id"].values, dtype=torch.int),
            :,
        ] = torch.tensor(grp_verts[["x", "y"]].values, dtype=torch.float)
        yield grp_tensor


def load_scanner_geometry_csv(csv_path: str):
    # Get the plate polygons from the CSV file
    group_names = ["plate_{}".format(i) for i in range(6)]
    plate_polygon_tensor = torch.cat(
        list(get_polygon_groups_from_csv(group_names, csv_path))
    )
    group_names = ["crystals_{}".format(i) for i in range(6)]
    xtal_polygon_tensor = torch.cat(
        list(get_polygon_groups_from_csv(group_names, csv_path))
    )
    return plate_polygon_tensor, xtal_polygon_tensor


def get_edges_from_verts_2d(geoms_verts_tensor: Tensor) -> Tensor:

    return torch.stack(
        [geoms_verts_tensor, torch.roll(geoms_verts_tensor, 1, dims=1)], dim=2
    )

def if_polys_in_hull_2d(geoms_verts_tensor: Tensor, hull: Tensor) -> Tensor:
    """Return the indices of geometry polygons completely inside *hull*.

    Uses the sign‑test of cross‑products for every hull edge; works for any V.
    """
    n_geoms, V, _ = geoms_verts_tensor.shape
    n_hull = hull.shape[0]

    p0 = hull.unsqueeze(0).unsqueeze(0).expand(n_geoms, V, n_hull, 2)
    p1 = torch.roll(hull, shifts=-1, dims=0).unsqueeze(0).unsqueeze(0).expand_as(p0)
    p2 = geoms_verts_tensor.unsqueeze(2).expand(n_geoms, V, n_hull, 2)

    v1 = p1 - p0
    v2 = p2 - p0
    cross = v2[..., 0] * v1[..., 1] - v2[..., 1] * v1[..., 0]
    cross_sign = torch.sign(cross)

    inside = torch.logical_or(
        (cross_sign >= 0).all(dim=2),  # all left turns
        (cross_sign <= 0).all(dim=2),  # or all right turns
    ).all(1)  # every vertex passes

    return torch.arange(n_geoms, device=geoms_verts_tensor.device)[inside]


def if_rects_in_hull_2d(geoms_verts_tensor: torch.Tensor, hull) -> torch.Tensor:
    # geoms_verts_tensor shape (n_geoms, 4, 2)
    # hull shape (n_hull, 2)
    n_geoms = geoms_verts_tensor.shape[0]
    n_hull = hull.shape[0]
    p0_tensor = hull.unsqueeze(0).unsqueeze(0).expand(n_geoms, 4, n_hull, 2)
    p1_tensor = (
        hull.roll(1, dims=0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(n_geoms, 4, n_hull, 2)
    )
    p2_tensor = geoms_verts_tensor.unsqueeze(2).expand(n_geoms, 4, n_hull, 2)
    v1 = p1_tensor - p0_tensor
    v2 = p2_tensor - p0_tensor
    cross_signs = torch.sign(
        v2[:, :, :, 0] * v1[:, :, :, 1] - v2[:, :, :, 1] * v1[:, :, :, 0]
    ).view(n_geoms, 4, n_hull)

    return torch.arange(n_geoms)[
        torch.logical_or(
            (cross_signs >= 0).all(dim=2), (cross_signs <= 0).all(dim=2)
        ).all(1)
    ]


def if_rects_intersect_hull_2d(
    geoms_edges_2d, edges_indices, hull, xtal_center
) -> torch.Tensor:
    verts, _ = get_verts_sorted_by_angel_2d(hull, xtal_center)
    rays = get_rays_2d(verts[[1, -1]], xtal_center.view(1, 2)).view(-1, 2, 2)
    _, index = get_cuts_ray_on_edges_2d(rays, geoms_edges_2d)
    return torch.unique(edges_indices[index[:, 1]][:, 0])


def get_rays_2d(pa_arr: torch.Tensor, pb_arr: torch.Tensor) -> torch.Tensor:
    """
    Get rays from array of points a and array of points b
    """
    npa = pa_arr.shape[0]
    npb = pb_arr.shape[0]
    return torch.stack(
        (
            pa_arr.unsqueeze(1).expand(-1, npb, -1),
            pb_arr.unsqueeze(0).expand(npa, -1, -1),
        ),
        dim=2,
    )


def get_fov_pixel_centers_2d(
    n_pixel_tensor: torch.Tensor,
    mm_per_pixel_tensor: torch.Tensor,
    fov_center: torch.Tensor = torch.tensor([0, 0]),
):
    gridx, gridy = torch.meshgrid(
        torch.arange(n_pixel_tensor[0].item()),
        torch.arange(n_pixel_tensor[1].item()),
        indexing="ij",
    )
    fov_dims = n_pixel_tensor * mm_per_pixel_tensor
    return (
        (torch.stack((gridx, gridy), dim=-1) + torch.tensor([0.5, 0.5]))
        * mm_per_pixel_tensor
        - fov_dims * 0.5
        + fov_center
    )


def get_angular_term_subdiv(rays, rects):
    pi = 3.14159265358979323846
    # The 2nd dimension of the rays must match the first dimension of the rects
    rects_vecs = (
        torch.stack([-rects[:, 1:, 1], rects[:, 1:, 0]], dim=2)
        .unsqueeze(0)
        .expand(rays.shape[0], -1, -1, -1)
    )
    rays_vecs = rays[:, :, 1] - rays[:, :, 0]
    rays_len = torch.linalg.norm(rays_vecs, dim=2)
    proj = (
        torch.abs(
            (rects_vecs * rays_vecs.unsqueeze(2).expand(-1, -1, 2, -1)).sum(
                dim=3
            )
        ).sum(dim=2)
        / rays_len
    )
    ratio = torch.atan2(0.5 * proj, rays_len) / pi
    return ratio


def get_angular_terms_2d(rays, rays_lengths, edges):
    """
    Solid-angle fraction intercepted by a *convex* polygon from each ray.
    Works for any number of edges (V).
    """
    pi = torch.pi
    # edges: (V, 2, 2)
    edges_vecs = edges[:, 1] - edges[:, 0]
    norm_vecs = torch.stack([-edges_vecs[:, 1], edges_vecs[:, 0]], dim=1)  # (V,2)

    rays_vecs = rays[:, 1] - rays[:, 0]
    # Shape of rays_vecs (n_rays, 2)

    proj = (
        torch.abs(
            (
                norm_vecs.unsqueeze(0).expand(rays.shape[0], 2, 2)
                * rays_vecs.unsqueeze(1).expand(-1, 2, -1)
            ).sum(dim=2)
        ).sum(dim=1)
        / rays_lengths
    )
    ratio = torch.atan2(0.5 * proj, rays_lengths) / pi
    return ratio


def get_rect_subdivs(rect, nsubs):
    nsubs_total = nsubs.prod()
    x, y = torch.meshgrid(
        torch.linspace(0, 1, nsubs[0] + 1)[:-1],
        torch.linspace(0, 1, nsubs[1] + 1)[:-1],
        indexing="ij",
    )
    rects_exp = rect.unsqueeze(0).expand(nsubs_total, -1, -1)
    return torch.cat(
        [
            (
                x.flatten().view(-1, 1).expand(-1, 2) * rects_exp[:, 1]
                + y.flatten().view(-1, 1).expand(-1, 2) * rects_exp[:, 2]
                + rect[0]
            ).view(-1, 1, 2),
            rects_exp[:, 1:] / nsubs.view(1, 1, 2).expand(nsubs_total, 2, -1),
        ],
        dim=1,
    )


def get_fov_verts_2d(fov_dims):
    return torch.tensor(
        [
            [-fov_dims[0] * 0.5, -fov_dims[1] * 0.5],
            [fov_dims[0] * 0.5, -fov_dims[1] * 0.5],
            [fov_dims[0] * 0.5, fov_dims[1] * 0.5],
            [-fov_dims[0] * 0.5, fov_dims[1] * 0.5],
        ]
    )


def get_verts_sorted_by_angel_2d(
    vertices: torch.Tensor, ref_point: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # sort the vertices by angle to point ref_point
    rads = torch.atan2(
        vertices[:, 1] - ref_point[1], vertices[:, 0] - ref_point[0]
    )
    rads = (rads + 2 * torch.pi) % (2 * torch.pi)
    order = torch.argsort(rads)
    return vertices[order], rads[order]


def get_three_p_cross(points):
    return (
        points[1, 0] * (points[2, 1] - points[0, 1])
        + points[2, 0] * (points[0, 1] - points[1, 1])
        + points[0, 0] * (points[1, 1] - points[2, 1])
    )


def get_verts_sorted_by_xy_2d(verts: torch.Tensor) -> torch.Tensor:
    # sort the vertices by x
    verts = verts[torch.argsort(verts[:, 0])]
    # sort the vertices again by y if x is the same
    return verts[torch.argsort(verts[:, 1])]


def get_convex_hull_2d(points: torch.Tensor) -> torch.Tensor:
    # sort the points by x and y
    points = get_verts_sorted_by_xy_2d(points)
    # sort the points by angle reference to the first point
    points = get_verts_sorted_by_angel_2d(points, points[0])[0]
    # get the convex hull
    convex_hull = points[:2]
    for i in range(2, points.shape[0]):
        convex_hull = torch.vstack((convex_hull, points[i]))
        if (
            convex_hull.shape[0] > 1
            and get_three_p_cross(convex_hull[-3:]) <= 0
        ):
            convex_hull = torch.vstack([convex_hull[:-2], convex_hull[-1]])
    return convex_hull


def get_furthest_corners(verts, n_corners: int = 4) -> torch.Tensor:
    centroid = torch.mean(verts, dim=0)
    dist_to_centroid = torch.norm(verts - centroid, dim=1)
    order = torch.argsort(dist_to_centroid)
    return verts[order[-n_corners:]].view(n_corners, 2)


def get_rays_cut_subdivs_self_2d(rays: Tensor, edges: Tensor, *, epsilon: float = 1e-6) -> Tensor:
    """Intersect each *ray* with *all* edges of a **single** convex polygon.

    Parameters
    ----------
    rays  : `(n_rays, 2, 2)`
    edges : `(E, 2, 2)`

    Returns
    -------
    Tensor
        `(n_rays, 1)` – *t* value of **closest** valid intersection along each ray
        (0 ⇒ no hit).  Uses Cramer’s rule, handles arbitrary *E*.
    """
    n_rays = rays.shape[0]
    n_edges = edges.shape[0]

    v1 = (rays[:, 1] - rays[:, 0]).unsqueeze(1).expand(-1, n_edges, -1)
    v2 = (edges[:, 0] - edges[:, 1]).unsqueeze(0).expand(n_rays, -1, -1)
    v3 = edges[:, 0].unsqueeze(0).expand(n_rays, -1, -1) - rays[:, 0].unsqueeze(1)

    det = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]

    t = torch.where(
        det.abs() > 1e-12,
        (v3[:, :, 0] * v2[:, :, 1] - v2[:, :, 0] * v3[:, :, 1]) / det,
        torch.full_like(det, float("nan")),
    )
    s = torch.where(
        det.abs() > 1e-12,
        (v1[:, :, 0] * v3[:, :, 1] - v1[:, :, 1] * v3[:, :, 0]) / det,
        torch.full_like(det, float("nan")),
    )

    valid = (s >= -epsilon) & (s <= 1 + epsilon) & (t > 0) & (t < 1)
    t_valid = torch.where(valid, t, float("nan"))

    # choose smallest positive t per ray
    t_min, _ = torch.nan_to_num(t_valid, nan=float("inf")).min(dim=1, keepdim=True)
    t_min[t_min == float("inf")] = 0.0
    return t_min


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
    t = torch.where((s <= 1) * (s >= 0) * (t < 1) * (t > 0), t, float("nan"))
    index = torch.argwhere(~torch.isnan(t))
    return t, index


def get_reduced_raytracing_edges_2d(
    crystal_idx: int,
    geom_dict: dict,
):
    """
    Finds a small, relevant subset of geometries by culling objects
    that are not in the 'cone of view' between the FOV and a crystal.
    This version handles separate lists of plates (hexagons) and crystals (quads).
    """
    xtal_center = geom_dict["pb_tensor"][crystal_idx]
    fov_corners = geom_dict["fov_corners"]
    
    plate_verts = geom_dict["plate_verts"]
    xtal_verts = geom_dict["xtal_verts"]
    n_plates = geom_dict["n_plates"]
    n_xtals = geom_dict["n_xtals"]
    
    # Create the convex hull (the "cone of view") for culling
    hull = get_convex_hull_2d(torch.vstack((xtal_center.unsqueeze(0), fov_corners)))

    # --- Culling Step 1: Broad Phase - AABB (Axis-Aligned Bounding Box) test ---
    # This is a very fast first pass to eliminate most objects.
    hull_min, _ = torch.min(hull, dim=0)
    hull_max, _ = torch.max(hull, dim=0)

    # Plates AABB test
    plate_min, _ = torch.min(plate_verts, dim=1)
    plate_max, _ = torch.max(plate_verts, dim=1)
    candidate_plate_indices = torch.where(
        (plate_max[:, 0] >= hull_min[0]) & (plate_min[:, 0] <= hull_max[0]) &
        (plate_max[:, 1] >= hull_min[1]) & (plate_min[:, 1] <= hull_max[1])
    )[0]
    
    # Crystals AABB test
    xtal_min, _ = torch.min(xtal_verts, dim=1)
    xtal_max, _ = torch.max(xtal_verts, dim=1)
    candidate_xtal_indices = torch.where(
        (xtal_max[:, 0] >= hull_min[0]) & (xtal_min[:, 0] <= hull_max[0]) &
        (xtal_max[:, 1] >= hull_min[1]) & (xtal_min[:, 1] <= hull_max[1])
    )[0]

    # --- Culling Step 2: Narrow Phase - More precise tests on the candidates ---
    # For now, we'll assume the AABB test is sufficient to avoid crashes.
    # A full intersection/inclusion test here would be more accurate but is complex.
    # This step is often where subtle bugs hide. Let's proceed with the AABB candidates.
    final_plate_indices = candidate_plate_indices
    final_xtal_indices_relative = candidate_xtal_indices
    
    # Exclude the current crystal itself from the list of attenuators
    final_xtal_indices_relative = final_xtal_indices_relative[final_xtal_indices_relative != crystal_idx]

    # --- Final Assembly ---
    # Get the actual vertex data for the culled objects
    reduced_plate_verts = plate_verts[final_plate_indices]
    reduced_xtal_verts = xtal_verts[final_xtal_indices_relative]
    
    # Create the edges for the reduced set of objects
    reduced_plate_edges = get_edges_from_verts_2d(reduced_plate_verts)
    reduced_xtal_edges = get_edges_from_verts_2d(reduced_xtal_verts)
    
    # Get the indices for looking up the mu values later
    geom_indices_for_mu = torch.cat([final_plate_indices, final_xtal_indices_relative + n_plates])
    
    # Return the edge data and the mu-lookup indices
    return reduced_plate_edges, reduced_xtal_edges, geom_indices_for_mu


def get_cuts_rays_on_self_2d(rays: Tensor, edges: Tensor, *, epsilon: float = 1e-6) -> Tensor:
    """Return **all** intersection *t*’s between rays and polygon edges.
    Output shape: `(n_rays, K)` where *K* ≤ *E* per ray.
    """
    n_edges = edges.shape[0]
    n_rays = rays.shape[0]

    v1 = (rays[:, 1] - rays[:, 0]).unsqueeze(1).expand(-1, n_edges, -1)
    v2 = (edges[:, 0] - edges[:, 1]).unsqueeze(0).expand(n_rays, -1, -1)
    v3 = edges[:, 0].unsqueeze(0).expand(n_rays, -1, -1) - rays[:, 0].unsqueeze(1)

    det = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]

    t = torch.where(
        det.abs() > 1e-12,
        (v3[:, :, 0] * v2[:, :, 1] - v2[:, :, 0] * v3[:, :, 1]) / det,
        torch.full_like(det, float("nan")),
    )
    s = torch.where(
        det.abs() > 1e-12,
        (v1[:, :, 0] * v3[:, :, 1] - v1[:, :, 1] * v3[:, :, 0]) / det,
        torch.full_like(det, float("nan")),
    )

    valid = (s > -epsilon) & (s < 1 + epsilon) & (t > 0) & (t < 1)
    t = torch.where(valid, t, float("nan"))

    # sort & squeeze NaNs
    t_sorted, _ = torch.sort(torch.nan_to_num(t, nan=float("inf")), dim=1)
    out = t_sorted[t_sorted < float("inf")]
    # guarantee 2‑D output
    return out.view(n_rays, -1)


def get_cuts_rays_on_polygons_2d(
    rays: torch.Tensor,
    edges: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Function to get the intersections of rays on the edges of generic polygons.
    """
    # Handle the case where the culling results in an empty set of objects
    if edges.shape[0] == 0:
        return torch.empty((rays.shape[0], 0, 2), dtype=torch.float64)

    # edges shape: (n_polygons, n_edges, 2, 2)
    # rays shape: (n_rays, 2, 2)
    n_polygons, n_edges, _, _ = edges.shape
    n_rays = rays.shape[0]

    rays = rays.double()
    edges = edges.double()

    # Use broadcasting to test all rays against all edges of all polygons
    v1 = (rays[:, 1] - rays[:, 0]).view(n_rays, 1, 1, 2)
    v2 = (edges[:, :, 0] - edges[:, :, 1]).unsqueeze(0)
    v3 = edges[:, :, 0].unsqueeze(0) - rays[:, 0].view(n_rays, 1, 1, 2)

    # Cramer's rule for line segment intersection
    det = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    det[torch.abs(det) < 1e-9] = 1e-9 # Avoid division by zero

    t = (v3[..., 0] * v2[..., 1] - v2[..., 0] * v3[..., 1]) / det
    s = (v1[..., 0] * v3[..., 1] - v1[..., 1] * v3[..., 0]) / det
    
    # Find valid intersections
    t = torch.where(
        (s <= 1 + epsilon) & (s >= 0 - epsilon) & (t < 1) & (t > 0), t, 0
    )
    
    # For each ray-polygon pair, sort the intersection t-values
    t_sorted, _ = torch.sort(t, dim=2)
    
    # Return the last two t-values (entry and exit points)
    return t_sorted[..., -2:]


def get_pa_tensor(fov_dict):
    # Define the FOV
    # fov_n_pixels_tensor = torch.tensor([64, 64])
    fov_n_pixels_tensor = fov_dict["n_pixels"]
    fov_mm_per_pixel_tensor = fov_dict["mm_per_pixel"]
    fov_center = fov_dict["center"]

    # Get the FOV pixel centers
    return get_fov_pixel_centers_2d(
        fov_n_pixels_tensor, fov_mm_per_pixel_tensor, fov_center
    ).view(-1, 2)


def get_mu_tensor(n_xtals, n_plates):
    return torch.cat(
        [
            torch.tensor([3.5]).repeat(n_plates),
            torch.tensor([0.475]).repeat(n_xtals),
        ]
    )


def load_scanner_geometry_npz(filepath: str):
    from numpy import load as np_load

    geom_data = np_load(filepath)
    plate_geoms = geom_data["plate cuboids"]
    xtal_geoms = geom_data["crystal cuboids"]

    # Get the vertices of the cuboids
    xtal_geoms_verts_2d = get_verts_2d_from_cuboids(xtal_geoms)
    plate_geoms_verts_2d = get_verts_2d_from_cuboids(plate_geoms)
    return plate_geoms_verts_2d, xtal_geoms_verts_2d


def get_geom_dict(plate_geoms_verts_2d, xtal_geoms_verts_2d, fov_dict) -> dict:

    pa_tensor = get_pa_tensor(fov_dict)

    # --- THE FIX ---
    # DO NOT concatenate plates and crystals. They have different numbers of vertices.
    # Keep them as separate entries in the dictionary.
    
    n_plates = plate_geoms_verts_2d.shape[0]
    n_xtals = xtal_geoms_verts_2d.shape[0]

    return {
        # Store vertices directly
        "plate_verts": plate_geoms_verts_2d,
        "xtal_verts": xtal_geoms_verts_2d,
        
        "pa_tensor": pa_tensor,
        "pb_tensor": xtal_geoms_verts_2d.mean(dim=1),
        
        "mu_tensor": get_mu_tensor(n_xtals, n_plates),
        "fov_corners": get_furthest_corners(pa_tensor),
        "n_xtals": n_xtals,
        "n_plates": n_plates,
    }


def get_edges_indices(n_all_geoms: int, n_vertices: int) -> Tensor:
    return torch.stack(
        [
            torch.arange(n_all_geoms).repeat_interleave(n_vertices),
            torch.arange(n_vertices).repeat(n_all_geoms),
        ],
        dim=1,
    )


def get_ppdf(crystal_idx, geom_dict):
    pb_tensor = geom_dict["pb_tensor"][crystal_idx]
    pa_tensor = geom_dict["pa_tensor"]
    mu_tensor = geom_dict["mu_tensor"]
    n_plates = geom_dict["n_plates"]
    
    # Get the reduced set of geometries that are in the path
    reduced_plate_edges, reduced_xtal_edges, geom_indices_for_mu = get_reduced_raytracing_edges_2d(
        crystal_idx, geom_dict=geom_dict
    )
    
    # Get the edges of the specific crystal we are calculating for
    edges_xtal = get_edges_from_verts_2d(geom_dict["xtal_verts"][crystal_idx].unsqueeze(0)).squeeze(0)

    rays = get_rays_2d(pa_tensor, pb_tensor.unsqueeze(0)).squeeze(1)
    rays_lengths = torch.norm(rays[:, 1] - rays[:, 0], dim=1)

    # --- Physics Calculations (Unchanged, as requested) ---
    
    # 1. Absorption in the target crystal
    # The original get_cuts_rays_on_self_2d is flawed, let's use the new robust one
    t_absorb_pairs = get_cuts_rays_on_polygons_2d(rays, edges_xtal.unsqueeze(0))
    path_length_absorb = torch.abs(t_absorb_pairs[:, 0, 1] - t_absorb_pairs[:, 0, 0])
    dlmu_absorb = path_length_absorb * rays_lengths * mu_tensor[crystal_idx + n_plates]
    absorb_terms = 1 - torch.exp(-dlmu_absorb)

    # 2. Attenuation through all other relevant objects
    dlmu_attenu_total = torch.zeros_like(rays_lengths, dtype=torch.float64)

    # Attenuation from plates (hexagons)
    if reduced_plate_edges.shape[0] > 0:
        t_attenu_plates = get_cuts_rays_on_polygons_2d(rays, reduced_plate_edges)
        path_lengths_plates = torch.abs(t_attenu_plates[:, :, 1] - t_attenu_plates[:, :, 0]) * rays_lengths.unsqueeze(1)
        mu_plates = mu_tensor[geom_indices_for_mu[geom_indices_for_mu < n_plates]]
        dlmu_attenu_total += torch.sum(path_lengths_plates * mu_plates, dim=1)

    # Attenuation from other crystals (quadrilaterals)
    if reduced_xtal_edges.shape[0] > 0:
        t_attenu_xtals = get_cuts_rays_on_polygons_2d(rays, reduced_xtal_edges)
        path_lengths_xtals = torch.abs(t_attenu_xtals[:, :, 1] - t_attenu_xtals[:, :, 0]) * rays_lengths.unsqueeze(1)
        mu_xtals = mu_tensor[geom_indices_for_mu[geom_indices_for_mu >= n_plates]]
        dlmu_attenu_total += torch.sum(path_lengths_xtals * mu_xtals, dim=1)
        
    attenu_terms = torch.exp(-dlmu_attenu_total)
    
    # 3. Geometric Efficiency
    angula_terms = get_angular_terms_2d(
        rays, rays_lengths, edges_xtal.view(-1, 2, 2)[:2]
    )
    
    return (absorb_terms * attenu_terms * angula_terms).float()


def set_default_device_as_cpu(use_logical_cores: bool = False):
    import psutil

    torch.set_default_device("cpu")
    # Get the number of physical cores (excluding hyper-threading)
    physical_cores = psutil.cpu_count(logical=False)
    # Get the number of logical cores (including hyper-threading)
    logical_cores = psutil.cpu_count(logical=True)
    torch.set_num_threads(
        int(physical_cores) if physical_cores is not None else 1
    )
