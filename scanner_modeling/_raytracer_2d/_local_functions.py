from typing import Dict, List, Sequence, Tuple

from torch import Tensor, arange, argwhere, bmm, cat, device
from torch import float64 as torch_float64
from torch import int32 as torch_int32
from torch import linspace, meshgrid, pi, stack, tensor, where
from torch import compile

from .._convex_hull._convex_hull_functions import (
    convex_hull_2d,
    sort_points_for_hull_2d,
)
from ..geometry_2d import (
    fov_corners_vertices_2d,
    polygon_edges_from_vertices_2d_batch,
    polygon_to_points_angular_span_2d_batch,
    reduced_scanner_objects_ids_local,
)


def rays_2d_batch(pa_batch: Tensor, pb_batch: Tensor) -> Tensor:
    """
    Form rays from batch of points a and batch of points b
    """
    npa = pa_batch.shape[0]
    npb = pb_batch.shape[0]
    return stack(
        (
            pa_batch.unsqueeze(1).expand(-1, npb, -1),
            pb_batch.unsqueeze(0).expand(npa, -1, -1),
        ),
        dim=2,
    )


def reduces_edges_polygon_ids(reduced_polygon_ids, n_edges_per_polygon):
    """
    Parameters
    ----------
    reduced_polygon_ids : Tensor

    n_edges_per_polygon : int

    Returns
    -------
    Tensor
            2D tensor of reduced edges ids
    """

    return reduced_polygon_ids.repeat_interleave(n_edges_per_polygon, dim=0).squeeze()


def line_segments_t(
    ls_a: Tensor,
    ls_b: Tensor,
    **kwargs,
):
    """
    Calculate the parameterized points of the intersection of two line segments \\
    `t` is the parameterized point on the first line segment. \\
    `s` is the parameterized point on the second line segment. \\

    Parameters
    ----------

    - `ls_a` : `Tensor`
        First batch of line segments

        `shape`: (`N_batch_A`, 2, 2)

    - `ls_b` : `Tensor`
        Second batch of line segments

        `shape`: (`N_batch_B`, 2, 2)

    - `kwargs` : `Dict`
        Additional arguments

        - eps: float

            Epsilon value for numerical stability. Default is 1e-9.
    Returns
    -------

    """

    n_ls_a = ls_a.shape[0]
    n_ls_b = ls_b.shape[0]
    # ls_a = ls_a.to(torch_float64)
    # ls_b = ls_b.to(torch_float64)
    # epsilon
    eps = kwargs.get("eps", 1e-9)

    va = (ls_a[:, 1] - ls_a[:, 0]).unsqueeze(1).expand(-1, n_ls_b, -1)
    vb = (ls_b[:, 0] - ls_b[:, 1]).unsqueeze(0).expand(n_ls_a, -1, -1)
    v3 = ls_b[:, 0].unsqueeze(0).expand(n_ls_a, -1, -1) - ls_a[:, 0].view(
        n_ls_a, 1, 2
    ).expand(-1, n_ls_b, -1)
    # cramer's rule
    # det shape (number of ls_a, number of ls_b)
    det = va[:, :, 0] * vb[:, :, 1] - va[:, :, 1] * vb[:, :, 0]

    # t shape (number of ls_a, number of ls_b)
    # s shape (number of ls_a, number of ls_b)
    t = where(
        abs(det) > eps,
        (v3[:, :, 0] * vb[:, :, 1] - v3[:, :, 1] * vb[:, :, 0]) / det,
        -1,
    )
    s = where(
        abs(det) > eps,
        (va[:, :, 0] * v3[:, :, 1] - va[:, :, 1] * v3[:, :, 0]) / det,
        -1,
    )
    t = where((s <= 1) * (s >= 0), t, -1).clamp(0, 1)
    return t


def rays_intersection_lengths_dev(
    rays: Tensor,
    rays_t: Tensor,
):
    """
    Calculate the intersection lengths of the rays given the t values.
    """
    rays = rays.view(-1, 2, 2).to(torch_float64)
    rays_t_reshaped = rays_t.reshape(rays.shape[0], -1, 4).to(torch_float64)
    rays_t_sorted = rays_t_reshaped.sort(dim=2).values
    rays_t_diff = rays_t_sorted[:, :, -1] - rays_t_sorted[:, :, -2]

    # rays_t_diff = where(rays_t_diff > 1 - eps, 2 - rays_t_diff, rays_t_diff)
    length = rays_t_diff * (rays[:, 1] - rays[:, 0]).norm(
        dim=1, dtype=torch_float64
    ).view(-1, 1)
    indices = argwhere(length > 0)
    points = rays[indices[:, 0], 0].view(-1, 1, 2).expand(-1, 2, -1) + (
        rays[indices[:, 0], 1] - rays[indices[:, 0], 0]
    ).view(-1, 1, 2).expand(-1, 2, -1) * rays_t_sorted[
        indices[:, 0], indices[:, 1], -2:
    ].view(
        -1, 2, 1
    ).expand(
        -1, -1, 2
    )

    return length, points


def rays_intersection_lengths(
    rays: Tensor,
    rays_t: Tensor,
) -> Tensor:
    """
    Calculate the intersection lengths of the rays given the t values.
    """
    # rays = rays.view(-1, 2, 2).to(torch_float64)
    rays = rays.view(-1, 2, 2)
    # rays_t_reshaped = rays_t.reshape(rays.shape[0], -1, 4).to(torch_float64)
    rays_t_reshaped = rays_t.reshape(rays.shape[0], -1, 4)
    rays_t_sorted = rays_t_reshaped.sort(dim=2).values
    rays_t_diff = rays_t_sorted[:, :, -1] - rays_t_sorted[:, :, -2]

    # rays_t_diff = where(rays_t_diff > 1 - eps, 2 - rays_t_diff, rays_t_diff)
    # length = rays_t_diff * (rays[:, 1] - rays[:, 0]).norm(
    #     dim=1, dtype=torch_float64
    # ).view(-1, 1)
    length = rays_t_diff * (rays[:, 1] - rays[:, 0]).norm(dim=1).view(-1, 1)

    return length


def subdivision_grid_rectangle(n_sub: Sequence[int] | Tensor) -> Tensor:
    """
    Create a grid of n_sub x n_sub points in the range [0, 1].
    """
    grid = stack(
        meshgrid(
            linspace(0, 1, int(n_sub[0]) + 1),
            linspace(0, 1, int(n_sub[1]) + 1),
            indexing="ij",
        ),
        dim=-1,
    )
    return stack(
        (grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]), dim=-2
    ).view(
        -1, 4, 2
    )  # shape (n_sub**2, 4, 2)


def subdivision_vertices_rectangle(
    vertices: Tensor, grid: Tensor, device=device("cpu")
) -> Tensor:
    """
    Subdivide a rectangle into n_sub x n_sub smaller rectangles.
    """
    # vertices shape (4, 2)
    origin = vertices[0]
    # v1 = vertices[1] - origin
    # v2 = vertices[3] - origin
    v_matrix = stack((vertices[1] - origin, vertices[3] - origin))
    return bmm(grid, v_matrix.unsqueeze(0).expand(grid.shape[0], -1, -1)) + origin


def rays_edges_t_subdivisions(
    ls_a: Tensor,
    ls_b: Tensor,
    **kwargs,
):
    # ls_a.shape[1] = ls_b.shape[0] = number of points B = number of subdivisions
    # ls_a.shape[0] = number of points A
    # ls_b.shape[1] = number of edges
    # ls_a.shape = (number of points A, number of points B, 2, 2)
    # ls_b.shape = (number of points B, 4, 2, 2)

    n_edges = ls_b.shape[1]
    n_pa = ls_a.shape[0]
    n_pb = ls_b.shape[0]
    # epsilon
    eps = kwargs.get("eps", 1e-9)

    va = (ls_a[:, :, 1] - ls_a[:, :, 0]).unsqueeze(2).expand(-1, -1, n_edges, -1)
    vb = (ls_b[:, :, 0] - ls_b[:, :, 1]).unsqueeze(0).expand(n_pa, -1, -1, -1)
    v3 = ls_b[:, :, 0].unsqueeze(0).expand(n_pa, -1, -1, -1) - ls_a[:, :, 0].view(
        n_pa, n_pb, 1, 2
    ).expand(-1, -1, n_edges, -1)
    # cramer's rule
    # det shape (number of ls_a, number of ls_b)
    det = va[:, :, :, 0] * vb[:, :, :, 1] - va[:, :, :, 1] * vb[:, :, :, 0]

    # t shape (number of ls_a, number of ls_b)
    # s shape (number of ls_a, number of ls_b)
    t = where(
        abs(det) > eps,
        (v3[:, :, :, 0] * vb[:, :, :, 1] - v3[:, :, :, 1] * vb[:, :, :, 0]) / det,
        -1,
    )
    s = where(
        abs(det) > eps,
        (va[:, :, :, 0] * v3[:, :, :, 1] - va[:, :, :, 1] * v3[:, :, :, 0]) / det,
        -1,
    )
    t = where((s <= 1) * (s >= 0), t, -1).clamp(0, 1)
    return t


def sfov_pixels_batch(
    fov_dict: Dict[str, Tensor],
    n_sub: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    """
    Get the pixel centers and the corners of the subfield of views in batch.
    """
    fov_corners = fov_corners_vertices_2d(fov_dict)
    npx_sfov = fov_dict["n pixels"] / tensor(n_sub)
    sfov_corners_batch = subdivision_vertices_rectangle(
        fov_corners, subdivision_grid_rectangle(n_sub)
    )
    sfov_pxs_grid = (
        stack(
            meshgrid(
                *[arange(int(npx_sfov[i])) for i in range(2)],
                indexing="ij",
            ),
            dim=-1,
        )
        .view(1, -1, 2)
        .expand(n_sub[0] * n_sub[1], -1, -1)
    )

    sfov_centers = (
        sfov_corners_batch.mean(dim=1)
        .view(-1, 1, 2)
        .expand(-1, int(npx_sfov[0] * npx_sfov[1]), -1)
    )
    sfov_sizes = (
        (npx_sfov * fov_dict["mm per pixel"])
        .view(-1, 1, 2)
        .expand(-1, int(npx_sfov[0] * npx_sfov[1]), -1)
    )
    sfov_pxs_batch = (
        (sfov_pxs_grid + 0.5) * fov_dict["mm per pixel"]
        - sfov_sizes * 0.5
        + sfov_centers
    )

    return sfov_pxs_batch, sfov_corners_batch


def sfov_properties(
    fov_dict: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Calculate the global indices of the pixels in the subfield of views
    and the coordinates of the pixels in the subfield of views.

    Parameters
    ----------
    fov_n_subs : Sequence[int]
        The number of subdivisions in the field of view.
    """

    fov_corners = fov_corners_vertices_2d(fov_dict)

    sfov_npx = (fov_dict["n pixels"] / fov_dict["n subdivisions"]).to(torch_int32)
    sfov_corners_batch = subdivision_vertices_rectangle(
        fov_corners, subdivision_grid_rectangle(fov_dict["n subdivisions"])
    )
    sfov_indices = (
        stack(
            meshgrid(
                *[arange(int(fov_dict["n subdivisions"][i])) for i in [0, 1]],
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        * sfov_npx
    )

    sfov_pxs_ids = (
        stack(
            meshgrid(*[arange(int(sfov_npx[i])) for i in [0, 1]], indexing="ij"),
            dim=-1,
        )
        .view(1, -1, 2)
        .expand(int(fov_dict["n subdivisions"].prod()), -1, 2)
    ) + sfov_indices.unsqueeze(1).expand(-1, int(sfov_npx.prod()), -1)

    sfov_pxs_coords = (
        (sfov_pxs_ids.float() + 0.5) * fov_dict["mm per pixel"]
        - fov_dict["size in mm"] * 0.5
        + fov_dict["center coordinates in mm"]
    )

    return sfov_pxs_ids, sfov_pxs_coords, sfov_corners_batch


def reduced_edges_2d_local(
    sfov_idx: int,
    crystal_idx: int,
    sfov_corners_batch: Tensor,
    plate_objects_vertices: Tensor,
    plate_objects_edges: Tensor,
    crystal_objects_vertices: Tensor,
    crystal_objects_edges: Tensor,
    device: device = device("cpu"),
) -> Tuple[Tensor, Tensor]:
    local_hull = convex_hull_2d(
        sort_points_for_hull_2d(
            cat(
                (
                    sfov_corners_batch[sfov_idx],
                    crystal_objects_vertices[crystal_idx],
                ),
                dim=0,
            )
        )
    )
    reduced_plate_objects_ids, reduced_crystal_objects_ids = (
        reduced_scanner_objects_ids_local(
            local_hull,
            plate_objects_vertices,
            crystal_objects_vertices,
            plate_objects_edges,
            crystal_objects_edges,
            device=device,
        )
    )

    return (
        plate_objects_edges[reduced_plate_objects_ids],
        crystal_objects_edges[reduced_crystal_objects_ids],
    )



def ppdf_2d_local(
    sfov_idx: int,
    crystal_idx: int,
    sfov_pixels_batch: Tensor,
    crystal_objects_vertices: Tensor,
    reduced_plate_edges: Tensor,
    reduced_crystal_edges: Tensor,
    subdivision_grid: Tensor,
    mu_dict: Tensor,
    device: device,
) -> Tensor:
    """
    Calculate the ppdf of a section of the the entire FOV. 2D version.

    Parameters
    ----------
    fov_idx : int
        The index of the local field of view.

    crystal_idx : int
        The index of the crystal object of which the PPDF is calculated.

    sfov_pixels_batch : Tensor
        The pixel centers of the subfield of views.
        shape: (n_subdivisions, n_pixels, 2)

    sfov_corners_batch : Tensor
        The corners of the subfield of views.
        shape: (n_subdivisions, 4, 2)

    plate_objects_vertices : Tensor
        The vertices of the plate objects.
        shape: (n_plate_objects, 4, 2)

    crystal_objects_vertices : Tensor
        The vertices of the crystal objects.
        shape: (n_crystal_objects, 4, 2)

    plate_objects_edges : Tensor
        The edges of the plate objects.
        shape: (n_plate_objects, 4, 2, 2)

    crystal_objects_edges : Tensor
        The edges of the crystal objects.
        shape: (n_crystal_objects, 4, 2, 2)

    subdivision_grid : Tensor
        The grid used for subdivision.
        shape: (n_subdivisions, 4, 2)

    Returns
        -------
        Tensor
            The computed ppdf values.
        shape: (n_fov, n_plate_objects, n_crystal_objects, n_subdivisions)
    """

    sub_crystals_vertices = subdivision_vertices_rectangle(
        crystal_objects_vertices[crystal_idx], subdivision_grid
    )

    sub_crystals_edges = polygon_edges_from_vertices_2d_batch(sub_crystals_vertices)

    pa_batch = sfov_pixels_batch[sfov_idx]
    pb_batch = sub_crystals_vertices.mean(dim=1)

    rays = rays_2d_batch(pa_batch, pb_batch)
    rays_plates_transmission_t = line_segments_t(
        rays.view(-1, 2, 2), reduced_plate_edges.view(-1, 2, 2)
    )

    rays_crystal_transmission_t = line_segments_t(
        rays.view(-1, 2, 2), reduced_crystal_edges.view(-1, 2, 2)
    )

    rays_sub_crystal_t = rays_edges_t_subdivisions(rays, sub_crystals_edges)

    intersection_length_plates = rays_intersection_lengths(
        rays, rays_plates_transmission_t
    ).view(rays.shape[0], rays.shape[1], -1)
    intersection_length_crystals = rays_intersection_lengths(
        rays, rays_crystal_transmission_t
    ).view(rays.shape[0], rays.shape[1], -1)

    intersection_length_subdivisions = rays_intersection_lengths(
        rays, rays_sub_crystal_t
    ).view(rays.shape[0], rays.shape[1])

    subdivision_rads_span = polygon_to_points_angular_span_2d_batch(
        sub_crystals_vertices, pa_batch
    )

    sum_plate_exponent = (intersection_length_plates * mu_dict[0]).sum(dim=2)
    sum_crystal_exponent = (intersection_length_crystals * mu_dict[1]).sum(dim=2)

    subdivision_exponent = intersection_length_subdivisions * mu_dict[1]
    angular_term = subdivision_rads_span / (2 * pi)
    return (
        (-sum_plate_exponent - sum_crystal_exponent).exp()
        * (subdivision_exponent.exp() - 1)
        * angular_term
    ).sum(dim=1)
