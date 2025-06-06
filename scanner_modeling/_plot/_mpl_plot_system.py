from torch import Tensor, tensor
from matplotlib.axes import Axes
from matplotlib.collections import (
    PolyCollection,
)
from typing import Dict
from .._geometry_2d._utils import fov_corners_vertices_2d
# import numpy as np


def plot_polygons_from_vertices_mpl(vertices: Tensor, ax: Axes, **kwargs):
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p


def plot_fov_as_rectangle_mpl(fov_dict: dict, ax: Axes, **kwargs):
    fov_corners = fov_corners_vertices_2d(fov_dict)
    return plot_polygons_from_vertices_mpl(
        fov_corners.unsqueeze(0), ax, **kwargs
    )


def plot_scanner_from_vertices_2d_mpl(
    plate_polygon_tensor,
    crystal_polygon_tensor,
    ax: Axes,
    fov_dict: Dict[str, Tensor],
    plate_alpha: float = 0.5,
    crystal_alpha: float = 0.5,
    fov_alpha: float = 0.5,
):

    plate_polycoll = plot_polygons_from_vertices_mpl(
        plate_polygon_tensor, ax, color="C0", alpha=plate_alpha
    )
    crystal_polycoll = plot_polygons_from_vertices_mpl(
        crystal_polygon_tensor, ax, color="C1", alpha=crystal_alpha
    )
    fov_polycoll = plot_fov_as_rectangle_mpl(
        fov_dict, ax, fc="none", ec="C2", alpha=fov_alpha
    )
    ax.autoscale()
    return {
        "plate polygon collection": plate_polycoll,
        "crystal polygon collection": crystal_polycoll,
        "fov polygon collection": fov_polycoll,
    }
