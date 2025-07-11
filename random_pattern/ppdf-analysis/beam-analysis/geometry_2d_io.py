import os

from torch import Tensor
from torch import load as torch_load


def load_scanner_layouts(dirname: str, filename: str):

    import os

    full_path = os.path.join(dirname, filename)
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist.")
        raise FileNotFoundError(f"File {full_path} does not exist.")
    filename_unique_id = filename.split(".")[0].split("_")[-1]
    scanner_layouts_data = torch_load(full_path, weights_only=True)["layouts"]
    return scanner_layouts_data, filename_unique_id


def load_scanner_layout_geometries(layout_idx: int, scanner_layouts_data):
    # Load the scanner geometry
    plates_vertices: Tensor = scanner_layouts_data[f"position {layout_idx:03d}"][
        "plate segments"
    ]
    detector_units_vertices: Tensor = scanner_layouts_data[
        f"position {layout_idx:03d}"
    ]["detector units"]
    return plates_vertices, detector_units_vertices
