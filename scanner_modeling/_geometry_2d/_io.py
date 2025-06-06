import os
from typing import Tuple,Dict

from ._polygon import polygon_edges_from_vertices_2d_batch
from torch import Tensor
from torch import load as torch_load


def load_scanner_layouts(
    dirname: str, filename: str, keyword: str = "layouts"
) -> Tuple[Dict, str]:

    full_path = os.path.join(dirname, filename)
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist.")
        raise FileNotFoundError(f"File {full_path} does not exist.")
    filename_unique_id = filename.split(".")[0].split("_")[-1]
    scanner_layouts_data = torch_load(full_path, weights_only=True)[keyword]
    return scanner_layouts_data, filename_unique_id


def load_scanner_layout_vertices(
    layout_idx: int, scanner_layouts_data: Dict
) -> Tuple[Tensor, Tensor]:
    # Load the scanner geometry
    plates_vertices: Tensor = scanner_layouts_data[
        f"position {layout_idx:03d}"
    ]["plate segments"]
    detector_units_vertices: Tensor = scanner_layouts_data[
        f"position {layout_idx:03d}"
    ]["detector units"]
    return plates_vertices, detector_units_vertices


def load_scanner_geometry_from_layout(
    layout_idx: int, scanner_layouts_data
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Load the scanner geometry from the layout data.
    Parameters:
    -----------

        layout_idx (int): The index of the layout to load.

        scanner_layouts_data: The scanner layouts data.

    Returns:
    --------
        Tuple[Tensor, Tensor, Tensor, Tensor]: The plate and crystal vertices and edges.

        - plate_objects_vertices
        - crystal_objects_vertices
        - plate_objects_edges
        - crystal_objects_edges
    """
    # Load the scanner vertices
    plate_objects_vertices, crystal_objects_vertices = (
        load_scanner_layout_vertices(layout_idx, scanner_layouts_data)
    )

    plate_objects_edges = polygon_edges_from_vertices_2d_batch(
        plate_objects_vertices
    )
    crystal_objects_edges = polygon_edges_from_vertices_2d_batch(
        crystal_objects_vertices
    )
    return (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    )
