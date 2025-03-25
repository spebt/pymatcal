import torch
import pandas as pd


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


def get_edges_from_verts_2d(
    geoms_verts_tensor: torch.Tensor,
) -> torch.Tensor:
    # geoms_verts_tensor shape (n_geoms, 4, 2)
    n_geoms = geoms_verts_tensor.shape[0]
    return torch.stack(
        [geoms_verts_tensor, torch.roll(geoms_verts_tensor, 1, dims=1)], dim=2
    )


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
    pi = 3.14159265358979323846
    # Shape of rays (n_rays, 2, 2)
    # Shape of edges (2, 2, 2)
    edges_vecs = edges[:, 1] - edges[:, 0]
    norm_vecs = torch.stack(
        [-edges_vecs[:, 1], edges_vecs[:, 0]], dim=1
    )  # Shape of norm_vecs (2, 2)

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


def get_rays_cut_subdivs_self_2d(
    rays: torch.Tensor,
    edges: torch.Tensor,
):
    """
    Cut rays with line segments (edges) of subdiv
    """
    # expand v1 shape (n_pa, n_pb, 2) to (n_pa, n_pb, 4, 2)
    v1 = (rays[:, :, 1] - rays[:, :, 0]).unsqueeze(2).expand(-1, -1, 4, -1)

    # expand v2 shape (n_pb, 4, 2) to (n_pa, n_pb, 4, 2)
    v2 = (edges[:, :, 0] - edges[:, :, 1]).unsqueeze(0).expand_as(v1)

    # v3 should be v3 = (edges_start_vert - rays_start_vert)
    v3 = edges[:, :, 0].unsqueeze(0).expand_as(v1) - rays[:, :, 0].unsqueeze(
        2
    ).expand_as(v1)

    # shape of v1, v2, v3 unified to:
    # (n_pa, n_pb, 4, 2)

    # cramer's rule
    det = v1[:, :, :, 0] * v2[:, :, :, 1] - v1[:, :, :, 1] * v2[:, :, :, 0]
    t = torch.where(
        det != 0,
        (v3[:, :, :, 0] * v2[:, :, :, 1] - v2[:, :, :, 0] * v3[:, :, :, 1])
        / det,
        float("nan"),
    )
    s = torch.where(
        det != 0,
        (v1[:, :, :, 0] * v3[:, :, :, 1] - v1[:, :, :, 1] * v3[:, :, :, 0])
        / det,
        -1,
    )
    t = torch.where((s <= 1) * (s >= 0) * (t < 1) * (t > 0), t, float("nan"))
    return t[~torch.isnan(t)].view(v1.shape[:2])


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
    # edges shape (n_geoms,4, 2, 2)
    # edge_indices shape (n_geoms, 2)

    idx = crystal_idx + geom_dict["n_plates"]
    edges = geom_dict["all_geoms_edges_2d"]
    edge_indices = geom_dict["all_edges_indices"]
    corners = geom_dict["fov_corners"]
    xtal_center = geom_dict["pb_tensor"][crystal_idx]

    hull = get_convex_hull_2d(torch.vstack((xtal_center.unsqueeze(0), corners)))
    geoms_verts_2d = edges.view(-1, 4, 2, 2)[:, :, 0]
    # geometry indices that intersect with the hull
    local_intersection_indices = if_rects_intersect_hull_2d(
        edges.view(-1, 2, 2), edge_indices, hull, xtal_center
    )
    # exclude the current geometry index
    local_intersection_indices = local_intersection_indices[
        ~local_intersection_indices.eq(idx)
    ]

    # geometry indices that are inside the hull
    local_inclusion_indices = if_rects_in_hull_2d(geoms_verts_2d, hull)

    # combine the indices
    local_indices = torch.unique(
        torch.cat((local_intersection_indices, local_inclusion_indices))
    )
    return (
        edges[local_indices],
        local_indices,
    )


def get_cuts_rays_on_self_2d(
    rays: torch.Tensor,
    edges: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Function to get the intersections of rays on four edges.

    Both rays and edges are 2D line segments.

    Args:
      rays (torch.Tensor): A tensor of shape (n_rays, 2, 2) representing the rays. Each ray is defined by two points in 2D space.
      edges (torch.Tensor): A tensor of shape (n_edges, 2, 2) representing the edges. Each edge is defined by two points in 2D space.

    Returns:
      torch.Tensor: A dense tensor of shape (n_rays, 1) containing the `t` parameters of the intersections.
      If a ray does not intersect with an edge, the corresponding value is set to 0.
    """

    # edges shape (4, 2, 2)
    # rays shape (number rays, 2, 2)
    n_edges = 4
    n_rays = rays.shape[0]

    # Increase the precision of the rays and edges
    # rays = rays.double()
    # edges = edges.double()

    v1 = (rays[:, 1] - rays[:, 0]).unsqueeze(1).expand(-1, n_edges, -1)
    v2 = (edges[:, 0] - edges[:, 1]).unsqueeze(0).expand(n_rays, -1, -1)
    # v1 = (rays[1] - rays[0]).unsqueeze(0).expand(n_edges, 2)
    # v2 = edges[:, 0] - edges[:, 1]
    v3 = edges[:, 0].unsqueeze(0).expand(n_rays, -1, -1) - rays[:, 0].unsqueeze(
        1
    ).expand(-1, n_edges, -1)
    # v3 = edges[:, 0] - rays[0].unsqueeze(0).expand(n_edges, 2)

    # cramer's rule
    # v1, v2, v3 shape (n_rays, n_egdes, 2)
    # det shape (n_rays,n_egdes,)
    det = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
    t = torch.where(
        det != 0,
        (v3[:, :, 0] * v2[:, :, 1] - v2[:, :, 0] * v3[:, :, 1]) / det,
        -1,
    )
    s = torch.where(
        det != 0,
        (v1[:, :, 0] * v3[:, :, 1] - v1[:, :, 1] * v3[:, :, 0]) / det,
        -1,
    )
    return torch.sort(
        torch.where(
            (s < 1 + epsilon) * (s + epsilon > 0) * (t < 1) * (t > 0), t, 0
        )
        .round(decimals=6)
        .unique(dim=1),
        dim=1,
    ).values[:, -1]


def get_cuts_rays_on_rectangles_2d(
    rays: torch.Tensor,
    edges: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Function to get the intersections of rays on the four edges of rectangles.

    Both rays and edges are 2D line segments.

    Args:
      rays (torch.Tensor): A tensor of shape (number of rays, 2, 2) representing the rays. Each ray is defined by two points in 2D space.
      edges (torch.Tensor): A tensor of shape (number of rectangles, 4, 2, 2) representing the edges. Each edge is defined by two points in 2D space.

    Returns:
      torch.Tensor: A dense tensor of shape (number rays, number of rectangles, 4) containing the `t` parameters of the intersections.
      If a ray does not intersect with an edge, the corresponding value is set to 0.
    """

    # edges shape (number of rectangles, 4, 2, 2)
    # rays  shape (number of rays, 2, 2)
    n_geoms = edges.shape[0]
    n_rays = rays.shape[0]

    # Increase the precision of the rays and edges
    rays = rays.double()
    edges = edges.double()

    v1 = (
        (rays[:, 1] - rays[:, 0])
        .unsqueeze(1)
        .unsqueeze(1)
        .expand(n_rays, n_geoms, 4, 2)
    )
    v2 = (
        (edges[:, :, 0] - edges[:, :, 1])
        .unsqueeze(0)
        .expand(n_rays, n_geoms, 4, 2)
    )
    v3 = edges[:, :, 0].unsqueeze(0).expand(n_rays, n_geoms, 4, 2) - rays[
        :, 0
    ].unsqueeze(1).unsqueeze(1).expand(n_rays, n_geoms, 4, 2)

    # cramer's rule
    # v1, v2, v3 shape (n_rays, n_geoms, 4, 2)
    # det shape (n_rays, n_geoms, 4)
    det = v1[:, :, :, 0] * v2[:, :, :, 1] - v1[:, :, :, 1] * v2[:, :, :, 0]
    t = torch.where(
        det != 0,
        (v3[:, :, :, 0] * v2[:, :, :, 1] - v2[:, :, :, 0] * v3[:, :, :, 1])
        / det,
        -1,
    )
    s = torch.where(
        det != 0,
        (v1[:, :, :, 0] * v3[:, :, :, 1] - v1[:, :, :, 1] * v3[:, :, :, 0])
        / det,
        -1,
    )
    # Return :   A sparse tensor
    # Shape :    (number rays, number of rectangles, 2)
    # Elements : `t` parameters of the intersections.
    return torch.sort(
        torch.where(
            (s < 1 + epsilon) * (s + epsilon > 0) * (t < 1) * (t > 0), t, 0
        )
        .round(decimals=6)
        .unique(dim=2),
        dim=2,
    ).values[:, :, -2:]


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

    all_geoms_verts_2d = torch.cat([plate_geoms_verts_2d, xtal_geoms_verts_2d])
    all_geoms_edges_2d = get_edges_from_verts_2d(all_geoms_verts_2d)

    return {
        "all_geoms_edges_2d": all_geoms_edges_2d,
        "pa_tensor": pa_tensor,
        "pb_tensor": xtal_geoms_verts_2d.mean(dim=1),
        "all_edges_indices": get_edges_indices(
            plate_geoms_verts_2d.shape[0] + xtal_geoms_verts_2d.shape[0]
        ),
        "mu_tensor": get_mu_tensor(
            xtal_geoms_verts_2d.shape[0], plate_geoms_verts_2d.shape[0]
        ),
        "fov_corners": get_furthest_corners(pa_tensor),
        "n_xtals": xtal_geoms_verts_2d.shape[0],
        "n_plates": plate_geoms_verts_2d.shape[0],
    }


def get_edges_indices(n_all_geoms: int):
    return torch.stack(
        [
            torch.arange(n_all_geoms).repeat_interleave(4),
            torch.arange(4).repeat(n_all_geoms),
        ],
        dim=1,
    )


def get_ppdf(crystal_idx, geom_dict):
    all_geoms_edges_2d = geom_dict["all_geoms_edges_2d"]
    all_edges_indices = geom_dict["all_edges_indices"]
    pb_tensor = geom_dict["pb_tensor"][crystal_idx]
    pa_tensor = geom_dict["pa_tensor"]
    # n_xtals = geom_dict["n_xtals"]
    # n_plates = geom_dict["n_plates"]

    # Get the mu values
    mu_tensor = geom_dict["mu_tensor"]

    # Get the overall index of the crystal
    xtal_overall_idx = crystal_idx + geom_dict["n_plates"]

    # Get the reduced edges for raytracing
    reduced_edges, geom_indices = get_reduced_raytracing_edges_2d(
        crystal_idx, geom_dict=geom_dict
    )

    # Get the edges of the crystal
    edges_xtal = all_geoms_edges_2d[xtal_overall_idx].view(-1, 2, 2)

    rays = get_rays_2d(pa_tensor, pb_tensor.unsqueeze(0)).squeeze(1)
    rays_lengths = torch.norm(rays[:, 1] - rays[:, 0], dim=1)

    n_geoms = reduced_edges.shape[0]
    n_rays = rays.shape[0]

    t_absorb = get_cuts_rays_on_self_2d(rays, edges_xtal)
    t_attenu = get_cuts_rays_on_rectangles_2d(rays, reduced_edges)

    dlmu_absorb = (
        (1 - t_absorb) * rays_lengths * torch.abs(mu_tensor[xtal_overall_idx])
    )
    dlmu_attenu = (
        torch.abs(t_attenu[:, :, 1] - t_attenu[:, :, 0])
        * rays_lengths.unsqueeze(1).expand(-1, n_geoms)
        * mu_tensor[geom_indices].unsqueeze(0).expand(n_rays, -1)
    )

    absorb_terms = 1 - torch.exp(-dlmu_absorb)
    attenu_terms = torch.exp(-torch.sum(dlmu_attenu, dim=1))
    angula_terms = get_angular_terms_2d(
        rays, rays_lengths, edges_xtal.view(-1, 2, 2)[:2]
    )
    return absorb_terms * attenu_terms * angula_terms


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
