import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection, LineCollection


def get_rects_verts_2d(rects):
    return torch.stack(
        [
            rects[:, 0, :2],
            rects[:, 0, :2] + rects[:, 1, :2],
            rects[:, 0, :2] + rects[:, 1, :2] + rects[:, 2, :2],
            rects[:, 0, :2] + rects[:, 2, :2],
        ],
        dim=1,
    )


def get_rects_center_2d(rects):
    return rects[:, 0, :2] + 0.5 * (rects[:, 1, :2] + rects[:, 2, :2])


def get_rects_edges_2d(rects):
    verts = get_rects_verts_2d(rects)
    edges = torch.stack(
        [
            verts[:, (0, 1)],
            verts[:, (1, 2)],
            verts[:, (2, 3)],
            verts[:, (3, 0)],
        ],
        dim=1,
    )
    return edges


def get_rays_2d(pa_arr: torch.Tensor, pb_arr: torch.Tensor) -> torch.Tensor:
    """
    Get rays from array of points a and array of points b
    """
    npa = pa_arr.shape[0]
    npb = pb_arr.shape[0]
    return torch.stack(
        (
            pa_arr.view(-1, 1, 2).expand(-1, npb, -1),
            pb_arr.view(1, -1, 2).expand(npa, -1, -1),
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


def plot_rects(rects, ax, **kwargs):
    verts = get_rects_verts_2d(rects)
    p = PatchCollection(
        [Polygon(vert, closed=True) for vert in verts], **kwargs
    )
    ax.add_collection(p)
    return p


def get_angular_term(rays, rects):
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


# def plot_rays_2d(ax, rays, **kwargs):
#         lines_segs = LineCollection(
#         rays[:, i].view(-1, 2, 2).tolist(), colors="cyan", linewidths=1, ls="-"
#     )
#     ax.add_collection(lines_segs)

# def plot_fov_2d(ax, fov_pixel_centers, fov_dims,**kwargs):
#     hull = get_convex_hull_2d(get_verts_sorted_2d(get_fov_verts_2d(fov_dims)))

#     fov_frame = ax.add_patch(
#         Polygon(hull, closed=True, fill=False, edgecolor="purple")
#     )
#     fov_frame.set_zorder(10)


def plot_scanner_2d(ax, rect_tensors, rays, pstyles: list):
    legend_handles = []
    for id, rect_tensor in enumerate(rect_tensors):
        patch = mpatches.Patch(**pstyles[id])
        legend_handles.append(patch)
        _ = plot_rects(rect_tensor, ax, **pstyles[id])
    ax.set_aspect("equal")
    pa_arr = rays[:, 0, 0]
    pb_arr = rays[0, :, 1]

    # plot the points a and b
    ax.plot(pa_arr[:, 0], pa_arr[:, 1], "o", ms=2, c="r")
    ax.plot(pb_arr[:, 0], pb_arr[:, 1], "o", ms=2, c="orange")
    return legend_handles


def get_fov_verts_2d(fov_dims):
    return torch.tensor(
        [
            [-fov_dims[0] * 0.5, -fov_dims[1] * 0.5],
            [fov_dims[0] * 0.5, -fov_dims[1] * 0.5],
            [fov_dims[0] * 0.5, fov_dims[1] * 0.5],
            [-fov_dims[0] * 0.5, fov_dims[1] * 0.5],
        ]
    )


def get_verts_sorted_2d(vertices: torch.Tensor):
    # sort the vertices by x and y
    indices = torch.arange(vertices.shape[0])
    indices = indices[torch.argsort(vertices[indices, 0])]
    indices = indices[torch.argsort(vertices[indices, 1])]
    vertices = vertices[indices]
    p = vertices[0]
    # sort the vertices by angle to p
    order = torch.argsort(
        torch.atan2(vertices[1:, 1] - p[1], vertices[1:, 0] - p[0])
    )
    vertices = vertices[1:][order]
    return torch.vstack((p, vertices))


def get_verts_sorted_by_angel_2d(
    vertices: torch.Tensor, ref_point: torch.Tensor
):
    # sort the vertices by angle to point ref_point
    rads = torch.atan2(
        vertices[:, 1] - ref_point[1], vertices[:, 0] - ref_point[0]
    )
    rads = (rads + 2 * torch.pi) % (2 * torch.pi)
    order = torch.argsort(rads)
    return vertices[order], rads


def get_three_p_cross(points):
    return (
        points[1, 0] * (points[2, 1] - points[0, 1])
        + points[2, 0] * (points[0, 1] - points[1, 1])
        + points[0, 0] * (points[1, 1] - points[2, 1])
    )


def get_convex_hull_2d(points: torch.Tensor) -> torch.Tensor:
    points = get_verts_sorted_2d(points)
    convex_hull = points[:2]
    for i in range(2, points.shape[0]):
        convex_hull = torch.vstack((convex_hull, points[i]))
        if (
            convex_hull.shape[0] > 1
            and get_three_p_cross(convex_hull[-3:]) <= 0
        ):
            convex_hull = torch.vstack([convex_hull[:-2], convex_hull[-1]])
    return convex_hull


def if_rects_in_hull_2d(rects, hull):
    points = get_rects_verts_2d(rects)
    n_rects = rects.shape[0]
    n_hull = hull.shape[0]
    hull = get_verts_sorted_2d(hull)
    p0_tensor = hull.unsqueeze(0).unsqueeze(0).expand(n_rects, 4, n_hull, 2)
    p1_tensor = (
        torch.vstack([hull[1:], hull[0]])
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(n_rects, 4, n_hull, 2)
    )
    p2_tensor = points.unsqueeze(2).expand(-1, -1, n_hull, 2)
    v1 = p1_tensor - p0_tensor
    v2 = p2_tensor - p0_tensor
    cross = torch.sign(
        v2[:, :, :, 0] * v1[:, :, :, 1] - v2[:, :, :, 1] * v1[:, :, :, 0]
    ).view(n_rects, 4, n_hull)
    return torch.logical_or((cross > 0).all(dim=2), (cross < 0).all(dim=2)).all(
        dim=1
    )


def if_rects_intersect_hull_2d(rects, hull, sub_center):
    # get the edges of the rectangles
    edges = get_rects_edges_2d(rects)
    index = torch.tensor([2, -2])
    ends = get_verts_sorted_by_angel_2d(hull, sub_center)[index]
    rays = get_rays_2d(ends, sub_center.view(1, 2))
    _, index = get_cuts_ray_on_edges_2d(rays, edges)
    return torch.unique(index[:, 1])


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
    # rays shape (number of starting points, number of end points, 2, 2)
    # `number of end points` is always 1
    # edges shape (n_rects, 4, 2, 2)
    n_pa = rays.shape[0]
    n_rects = edges.shape[0]
    v1 = (
        (rays[:, :, 1] - rays[:, :, 0])
        .view(n_pa, 1, 1, 2)
        .expand(-1, n_rects, 4, -1)
    )
    v2 = (edges[:, :, 0] - edges[:, :, 1]).unsqueeze(0).expand(n_pa, -1, -1, -1)
    v3 = edges[:, :, 0].unsqueeze(0).expand(n_pa, -1, -1, -1) - rays[
        :, :, 0
    ].view(n_pa, 1, 1, 2).expand(-1, n_rects, 4, -1)

    # cramer's rule
    # v1, v2, v3 shape (n_pa, n_rects, 4, 2)
    # det shape (n_pa, n_rects, 4)
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
        float("nan"),
    )
    t = torch.where((s <= 1) * (s >= 0) * (t <= 1) * (t >= 0), t, float("nan"))
    index = torch.argwhere(~torch.isnan(t))
    return t, index
