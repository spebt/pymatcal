from torch import atan2, argsort, cat, unique as torch_unique, vstack, Tensor


def get_three_p_cross(points: Tensor) -> Tensor:
    return (
        points[1, 0] * (points[2, 1] - points[0, 1])
        + points[2, 0] * (points[0, 1] - points[1, 1])
        + points[0, 0] * (points[1, 1] - points[2, 1])
    )


def sort_points_by_xy_2d_batch(points_batch: Tensor, main_axis: int = 0) -> Tensor:
    # Get unique x values
    unique_x, unique_x_index = torch_unique(
        points_batch[:, :, 0], return_inverse=True, sorted=True
    )
    unique_y, unique_y_index = torch_unique(
        points_batch[:, :, 1], return_inverse=True, sorted=True
    )
    # Sort the indices based on x and y values
    indices_by_xy = argsort(unique_x_index * 100 + unique_y_index)
    return points_batch.gather(1, indices_by_xy.unsqueeze(-1).expand(-1, -1, 2))


def sort_points_by_rad_2d_batch(points_batch: Tensor) -> Tensor:
    # sort the vertices by angle to point ref_point
    n_points = points_batch.shape[1] - 1
    ref_points_batch = points_batch[:, 0].unsqueeze(1).expand(-1, n_points, -1)

    rads = atan2(
        points_batch[:, 1:, 1] - ref_points_batch[:, :, 1],
        points_batch[:, 1:, 0] - ref_points_batch[:, :, 0],
    )
    # rads = rads + (rads < 0) * (2 * torch.pi)
    order = argsort(rads, dim=1)
    output = cat(
        (
            points_batch[:, :1],
            points_batch[:, 1:, :].gather(
                dim=1, index=order.unsqueeze(-1).expand(-1, -1, 2)
            ),
        ),
        dim=1,
    )
    return output


def sort_points_by_rad_2d(points: Tensor) -> Tensor:
    # sort the vertices by angle to point ref_point
    n_points = points.shape[1] - 1
    ref_point = points[0]

    rads = atan2(
        points[1:, 1] - ref_point[1],
        points[1:, 0] - ref_point[0],
    )
    # rads = rads + (rads < 0) * (2 * torch.pi)
    output = cat(
        (
            points[0].unsqueeze(0),
            points[1:][argsort(rads, dim=0)],
        )
    )
    return output


def sort_points_by_xy(points: Tensor) -> Tensor:
    # Get unique x values
    _, unique_x_index = torch_unique(points[:, 0], return_inverse=True, sorted=True)
    _, unique_y_index = torch_unique(points[:, 1], return_inverse=True, sorted=True)
    # Sort the indices based on x and y values
    indices_by_xy = argsort(unique_x_index * 100 + unique_y_index)
    return points[indices_by_xy]


def sort_points_for_hull_2d(points: Tensor):
    # Sort the points by x and y
    sorted_points = sort_points_by_xy(points)
    # Sort the points by angle reference to the first point
    output = sort_points_by_rad_2d(sorted_points)
    return output


def sort_points_for_hull_batch_2d(points_batch: Tensor):
    """
    Sort the points in a batch of 2D points for convex hull computation.

    Parameters
    ----------
    points_batch : torch.Tensor, shape (n_batch, n_points, 2)
        The batch of points to be sorted.

    Returns
    -------
    output: torch.Tensor, shape (n_batch, n_points, 2)
    """

    # sort the points by x and y
    sorted_points_batch = sort_points_by_xy_2d_batch(points_batch)
    # sort the points by angle reference to the first point
    output = sort_points_by_rad_2d_batch(sorted_points_batch)
    return output


def convex_hull_2d(sorted_points: Tensor):
    """
    Compute the convex hull of a set of 2D points using the Monotone Chain algorithm.
    Parameters
    ----------
    sorted_points : torch.Tensor, shape (n_points, 2)
        The sorted points to compute the convex hull from.
    Returns
    -------
    convex_hull : torch.Tensor, shape (n_hull_points, 2)
        The points of the convex hull.
    """

    # get the convex hull
    convex_hull = sorted_points[:2]
    for i in range(2, sorted_points.shape[0]):
        convex_hull = vstack((convex_hull, sorted_points[i]))
        if convex_hull.shape[0] > 1 and get_three_p_cross(convex_hull[-3:]) <= 0:
            convex_hull = vstack([convex_hull[:-2], convex_hull[-1]])
    return convex_hull


def detector_units_fov_sorted_points_batch(
    detector_units_vertices: Tensor,
    fov_vertices: Tensor,
) -> Tensor:
    """
    Compute the convex hull of the FOV and the detector unit vertices. Batch version.

    Parameters
    ----------
    detector_unit_vertices : torch.Tensor, shape (n_batch, 4, 2)
        The vertices of the detector unit.
    fov_vertices : torch.Tensor, shape (1, n_fov_points, 2)
        The vertices of the FOV.

    Returns
    -------
    points : torch.Tensor, shape (n_batch, n_hull_points, 2)
        The points of the convex hull.
    """
    n_batch = detector_units_vertices.shape[0]

    points_batch = cat(
        (
            fov_vertices.expand(n_batch, -1, -1),
            detector_units_vertices,
        ),
        dim=1,
    )
    return sort_points_for_hull_batch_2d(points_batch)
