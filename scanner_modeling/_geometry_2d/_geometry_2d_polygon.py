from torch import Tensor, abs, arange, argwhere, cat, device
from torch import empty as empty_tensor
from torch import float64, isnan, logical_or, sign, stack, where


def polygon_edges_from_vertices_2d(polygon: Tensor) -> Tensor:
    """
    Get edges of a polygon
    """

    return stack((polygon, polygon.roll(-1, dims=0)), dim=1)


def polygon_edges_from_vertices_2d_batch(polygon_batch: Tensor) -> Tensor:
    """
    Get edges of polygons in batch
    """

    # polygon_batch shape (n_batch, n_vertices, 2)
    return stack((polygon_batch, polygon_batch.roll(-1, dims=1)), dim=2)


def line_segments_intersecting_ids_batch(
    ls_batch_a: Tensor,
    ls_batch_b: Tensor,
    **kwargs,
):
    """
    Check if first batch of line segments and second batch of line segments intersect
    """

    # ls_a_batch shape (batch number of line segments a, 2, 2)
    # ls_b_batch shape (batch number of line segments b, 2, 2)

    n_ls_a = ls_batch_a.shape[0]
    n_ls_b = ls_batch_b.shape[0]

    # epsilon
    eps = kwargs.get("eps", 1e-9)

    va = (
        (ls_batch_a[:, 1] - ls_batch_a[:, 0])
        .unsqueeze(1)
        .expand(-1, n_ls_b, -1)
    )
    vb = (
        (ls_batch_b[:, 0] - ls_batch_b[:, 1])
        .unsqueeze(0)
        .expand(n_ls_a, -1, -1)
    )
    v3 = ls_batch_b[:, 0].unsqueeze(0).expand(n_ls_a, -1, -1) - ls_batch_a[
        :, 0
    ].view(n_ls_a, 1, 2).expand(-1, n_ls_b, -1)
    # cramer's rule
    # det shape (number of ls_a, number of ls_b)
    det = va[:, :, 0] * vb[:, :, 1] - va[:, :, 1] * vb[:, :, 0]

    # t shape (number of ls_a, number of ls_b)
    # s shape (number of ls_a, number of ls_b)
    t = where(
        abs(det) > eps,
        (v3[:, :, 0] * vb[:, :, 1] - v3[:, :, 1] * vb[:, :, 0]) / det,
        float("nan"),
    )
    s = where(
        abs(det) > eps,
        (va[:, :, 0] * v3[:, :, 1] - va[:, :, 1] * v3[:, :, 0]) / det,
        float("nan"),
    )
    t = where((s <= 1) * (s >= 0) * (t < 1) * (t > 0), t, float("nan"))
    # print("t", t.size())
    return (argwhere(~isnan(t))[:, 1]).unique()


def polygons_hull_intersecting_2d_polygons_ids(
    polygon_edges_batch, hull, device=device("cpu")
):

    # polygon_edges_batch shape (n_polygons, n_edges, 2, 2)

    hull_edges = polygon_edges_from_vertices_2d(hull)
    intersecting_edges_ids = line_segments_intersecting_ids_batch(
        hull_edges,
        polygon_edges_batch.view(-1, 2, 2),
    )
    polygon_edges_batch_polygon_ids = arange(
        polygon_edges_batch.shape[0], device=device
    ).repeat_interleave(polygon_edges_batch.shape[1])
    intersecting_polygon_ids = polygon_edges_batch_polygon_ids[
        intersecting_edges_ids
    ].unique()
    return intersecting_polygon_ids


def polygons_hull_enclosed_2d_polygons_ids(
    polygon_vertices_batch: Tensor, hull, device=device("cpu")
) -> Tensor:
    # vertices_batch shape (n_batch, n_vertices, 2)
    # hull shape (n_hull, 2)
    n_batch = polygon_vertices_batch.shape[0]
    n_vertex = polygon_vertices_batch.shape[1]
    n_hull = hull.shape[0]
    p0_tensor = (
        hull.unsqueeze(0).unsqueeze(0).expand(n_batch, n_vertex, n_hull, 2)
    )
    p1_tensor = (
        hull.roll(1, dims=0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(n_batch, n_vertex, n_hull, 2)
    )
    p2_tensor = polygon_vertices_batch.unsqueeze(2).expand(
        n_batch, n_vertex, n_hull, 2
    )
    v1 = p1_tensor - p0_tensor
    v2 = p2_tensor - p0_tensor
    cross_signs = sign(
        v2[:, :, :, 0] * v1[:, :, :, 1] - v2[:, :, :, 1] * v1[:, :, :, 0]
    ).view(n_batch, n_vertex, n_hull)

    return arange(n_batch, device=device)[
        logical_or(
            (cross_signs >= 0).all(dim=2), (cross_signs <= 0).all(dim=2)
        ).all(1)
    ]


def reduced_scanner_objects_ids_local(
    hull_2d: Tensor,
    plate_segments_vertices: Tensor,
    detector_units_vertices: Tensor,
    plate_segments_edges: Tensor,
    detector_units_edges: Tensor,
    device: device = device("cpu"),
) -> tuple[Tensor, Tensor]:
    reduced_plate_segments_ids = (
        cat(
            (
                polygons_hull_enclosed_2d_polygons_ids(
                    plate_segments_vertices, hull_2d, device=device
                ),
                polygons_hull_intersecting_2d_polygons_ids(
                    plate_segments_edges, hull_2d, device=device
                ),
            )
        )
        .unique()
        .sort()
        .values
    )
    reduced_detector_units_ids = (
        cat(
            (
                polygons_hull_enclosed_2d_polygons_ids(
                    detector_units_vertices, hull_2d, device=device
                ),
                polygons_hull_intersecting_2d_polygons_ids(
                    detector_units_edges, hull_2d, device=device
                ),
            )
        )
        .unique()
        .sort()
        .values
    )
    return reduced_plate_segments_ids, reduced_detector_units_ids
