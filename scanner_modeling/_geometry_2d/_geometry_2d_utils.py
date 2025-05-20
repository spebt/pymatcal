from typing import Dict, Sequence

from torch import Tensor, arange, atan2, float32
from torch import float64 as torch_float64
from torch import max as torch_max
from torch import meshgrid, pi, stack, tensor, where
from torch.nn import functional as F


def local_max_1d(data: Tensor, size: int = 3):
    """
    Get the local max of a 1D tensor with a given factor.
    """
    data_padded = F.pad(
        data.view(1, 1, -1), (size // 2, size // 2), mode="constant", value=0
    ).squeeze()
    return torch_max(data_padded.unfold(0, size, 1), dim=1).values


def fov_tensor_dict(
    n_pixels: Sequence[int] = (512, 512),
    size_in_mm: Sequence[float] = (128, 128),
    center_coordinates: Sequence[float] = (0.0, 0.0),
    n_subdivisions: Sequence[int] = (1, 1),
) -> dict:
    """
    Create a dictionary with the FOV information.
    """
    fov_dict = {
        "n pixels": tensor(n_pixels),
        "size in mm": tensor(size_in_mm),
        "center coordinates in mm": tensor(center_coordinates),
    }
    fov_dict["mm per pixel"] = fov_dict["size in mm"] / fov_dict["n pixels"]
    fov_dict["n subdivisions"] = tensor(n_subdivisions)
    return fov_dict


def fov_corners_vertices_2d(fov_dict: Dict[str, Tensor]) -> Tensor:
    fov_corners = tensor(
        [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]
    )
    fov_corners = (
        fov_corners * fov_dict["size in mm"] * 0.5
        + fov_dict["center coordinates in mm"]
    )

    return fov_corners


def pixels_coordinates(
    fov_dict: Dict,
) -> Tensor:

    pixel_indices = stack(
        meshgrid(
            arange(0, int(fov_dict["n pixels"][0])),
            arange(0, int(fov_dict["n pixels"][1])),
            indexing="ij",
        ),
        dim=2,
    ).view(-1, 2)
    return (
        (pixel_indices.to(dtype=float32) + 0.5) * fov_dict["mm per pixel"]
        - fov_dict["size in mm"] * 0.5
        + fov_dict["center coordinates in mm"]
    )


def pixels_to_detector_unit_rads(
    pixel_coordinates: Tensor,
    detector_unit_center: Tensor,
) -> Tensor:

    pixel_rads = atan2(
        pixel_coordinates[:, 1] - detector_unit_center[1],
        pixel_coordinates[:, 0] - detector_unit_center[0],
    )
    pixel_rads = pixel_rads + 2 * pi * (pixel_rads < 0)
    if pixel_rads.max() - pixel_rads.min() > pi:
        pixel_rads[pixel_rads > pi] -= 2 * pi
    return pixel_rads


def points_to_refs_angle_2d_batch(
    points_batch: Tensor, ref_point_batch: Tensor
) -> Tensor:
    """
    Calculate the angle of the points in 2D.
    """
    n_points = points_batch.shape[0]
    n_refs = ref_point_batch.shape[0]
    points_batch = (
        points_batch.to(torch_float64)
        .view(1, n_points, 2)
        .expand(n_refs, -1, -1)
    )
    ref_point = ref_point_batch.view(n_refs, 1, 2).expand(-1, n_points, -1)
    return atan2(
        points_batch[:, :, 1] - ref_point[:, :, 1],
        points_batch[:, :, 0] - ref_point[:, :, 0],
    )


def polygon_to_points_angular_span_2d_batch(
    polygon_vertices_batch: Tensor,
    ref_points_batch: Tensor,
) -> Tensor:
    """
    Calculate the angular span of points within a polygon in 2D.
    """
    polygon_vertices_rads = points_to_refs_angle_2d_batch(
        polygon_vertices_batch.view(-1, 2), ref_points_batch
    ).view(
        ref_points_batch.shape[0],
        polygon_vertices_batch.shape[0],
        polygon_vertices_batch.shape[1],
    )

    polygon_rads_span_batch = (
        polygon_vertices_rads.max(dim=2).values
        - polygon_vertices_rads.min(dim=2).values
    )
    return where(
        polygon_rads_span_batch > pi,
        2 * pi - polygon_rads_span_batch,
        polygon_rads_span_batch,
    )
