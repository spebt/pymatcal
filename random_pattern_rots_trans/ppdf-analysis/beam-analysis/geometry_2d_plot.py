import torch
import matplotlib
from matplotlib.axes import Axes
from matplotlib.collections import (
    PolyCollection,
)
from matplotlib import colors as mpl_colors
from typing import Dict

# import numpy as np


def plot_polygons_from_vertices_mpl(vertices: torch.Tensor, ax: Axes, **kwargs):
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p


def get_fov_corners(fov_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    fov_size_in_mm = fov_dict["size in mm"]
    fov_corners = torch.tensor(
        [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]
    )
    fov_corners = (
        fov_corners * fov_size_in_mm * 0.5 + fov_dict["center coordinates in mm"]
    )

    return fov_corners


def plot_fov_as_rectangle_mpl(fov_dict: dict, ax: Axes, **kwargs):
    fov_corners = get_fov_corners(fov_dict)
    return plot_polygons_from_vertices_mpl(fov_corners.unsqueeze(0), ax, **kwargs)


def plot_scanner_from_vertices_2d_mpl(
    plate_polygon_tensor, xtal_polygon_tensor, ax, fov_dict
):

    plate_polycoll = plot_polygons_from_vertices_mpl(
        plate_polygon_tensor, ax, color="b", alpha=0.5
    )
    crystal_polycoll = plot_polygons_from_vertices_mpl(
        xtal_polygon_tensor, ax, color="orange", alpha=0.5
    )
    fov_polycoll = plot_fov_as_rectangle_mpl(fov_dict, ax, fc="none", ec="g", alpha=0.5)
    ax.autoscale()
    return {
        "plate polygon collection": plate_polycoll,
        "crystal polygon collection": crystal_polycoll,
        "fov polygon collection": fov_polycoll,
    }


def plot_new_ppdf_mpl(ppdf_data, ax, fov_dict, **kwargs):
    original_cmap = matplotlib.colormaps.get(kwargs.get("cmap", "hot_r"))
    start = 0.0
    end = 0.8  # End at 80% of the original colormap
    n_colors = 256  # Number of colors in the new colormap
    # Sample colors from the original colormap
    if original_cmap is not None:
        colors_from_map = original_cmap(torch.linspace(start, end, n_colors).tolist())

        # Create the new colormap
        new_cmap = mpl_colors.LinearSegmentedColormap.from_list(
            f"{original_cmap.name}[{start:.2f}-{end:.2f}]", colors_from_map
        )
        ppdf = ppdf_data.view(
            int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
        )

        im = ax.imshow(
            ppdf.T,
            origin="lower",
            extent=(-64, 64, -64, 64),
            aspect="equal",
            vmin=0,
            cmap=new_cmap,
            **kwargs,
        )
        cb = ax.figure.colorbar(im, ax=ax)
        cb.ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        return im
