from typing import Dict, Sequence

from torch import Tensor, arange, atan2, float32
from torch import max as torch_max
from torch import meshgrid, pi, stack, tensor
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
    mm_per_pixel: Sequence[float] = (0.25, 0.25),
    center_coordinates: Sequence[float] = (0.0, 0.0),
) -> dict:
    """
    Create a dictionary with the FOV information.
    """
    fov_dict = {
        "n pixels": tensor([512, 512]),
        "mm per pixel": tensor([0.25, 0.25]),
        "center coordinates in mm": tensor([0.0, 0.0]),
    }
    fov_dict["size in mm"] = fov_dict["n pixels"] * fov_dict["mm per pixel"]
    return fov_dict


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
        pixel_indices.to(dtype=float32) * fov_dict["mm per pixel"]
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
