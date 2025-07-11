"""
Generate a randomized scanner layout with given parameters.


"""

from typing import Sequence, Dict, Union
from torch import (
    bmm,
    linspace,
    arange,
    stack,
    cat,
    cos,
    sin,
    meshgrid,
    tensor,
    Tensor,
    float32,
    pi,
)

OutDataDict = Dict[str, Union[str, Dict]]


def positions_parameters(
    n_rotations: int, n_shifts: Sequence[int], shift_step: Sequence[float]
):
    """
    Generate a set of angles and shifts for the scanner layout.
    """
    angles = linspace(0, 2 * pi, n_rotations + 1)[:-1]
    shifts_indices = stack(
        meshgrid(arange(n_shifts[0]), arange(n_shifts[1]), indexing="ij"),
        dim=-1,
    )
    shifts_indices[1::2, :, 1] = shifts_indices[1::2, :, 1].flip(1)
    shifts = shifts_indices * tensor(shift_step)
    shifts = shifts - (shifts.view(-1, 2)).to(dtype=float32).mean(0)
    return cat(
        [
            angles.repeat(n_shifts[0] * n_shifts[1]).view(-1, 1),
            shifts.view(-1, 2).repeat_interleave(n_rotations, dim=0),
        ],
        dim=1,
    )


def translation_matrix_2d_batch(translations: Tensor):
    n = translations.size(0)
    # Get the translation matrix in (2+1)-D

    translation_matrix = (
        tensor([[1, 0], [0, 1], [0, 0]], dtype=float32)
        .unsqueeze(0)
        .repeat(n, 1, 1)
    )  # (n, 3, 2)
    translation_matrix = cat(
        [
            translation_matrix,
            cat(
                [translations, tensor([[1]], dtype=float32).repeat(n, 1)], dim=1
            ).unsqueeze(2),
        ],
        dim=2,
    )  # (n, 3, 3)
    return translation_matrix


def rotation_matrix_2d_batch(angles: Tensor):

    rotation_matrix = stack(
        (
            cos(angles),
            -sin(angles),
            sin(angles),
            cos(angles),
        ),
        dim=1,
    ).view(-1, 2, 2)
    return rotation_matrix


def transform_to_positions_2d_batch(positions: Tensor, points_2d_batch: Tensor):
    # positions: (m, 3)
    # points_2d_batch, shape: (n, 2)
    m = positions.shape[0]
    n = points_2d_batch.shape[0]
    # Get the rotation matrix in 2-D
    rotation_matrix_batch = rotation_matrix_2d_batch(
        positions[:, 0]
    )  # (m, 2, 2)
    rotation_matrix_batch = rotation_matrix_batch.unsqueeze(1).expand(
        -1, n, -1, -1
    )  # (m, n, 2, 2)
    # Expand the points to (m, n, 2, 1)

    points = (
        points_2d_batch.unsqueeze(0).view(1, n, 2, 1).expand(m, n, 2, 1)
    )  # (m, n, 2, 1)
    # Perform the rotation
    rotated_points = bmm(
        rotation_matrix_batch.reshape(-1, 2, 2), points.reshape(-1, 2, 1)
    )  # (m*n, 2, 1)
    # print(f"Rotated points shape: {rotated_points.shape}")
    # print(
    #     tensor([[[1]]], dtype=float32)
    #     .repeat(rotated_points.shape[0], 1, 1)
    #     .shape
    # )
    # Pad the points with one on the third dimension
    rotated_points = cat(
        [
            rotated_points,
            tensor([[[1]]], dtype=float32).repeat(
                rotated_points.shape[0], 1, 1
            ),
        ],
        dim=1,
    ).view(
        -1, 3, 1
    )  # (m*n, 3, 1)
    # Get the translation matrix in (2+1)-D
    translation_matrix_batch = translation_matrix_2d_batch(
        positions[:, 1:]
    )  # (m, 3, 3)
    # Expand the translation matrix to (m, n, 3, 3)
    translation_matrix_batch = translation_matrix_batch.unsqueeze(1).expand(
        -1, n, -1, -1
    )  # (m, n, 3, 3)
    # Perform the translation
    return bmm(
        translation_matrix_batch.reshape(-1, 3, 3),
        rotated_points.reshape(-1, 3, 1),
    )[:, :2, 0].view(
        m, n, 2
    )  # (m, n, 2)


if __name__ == "__main__":
    import sys
    import os
    from torch import (
        load as torch_load,
        save as torch_save,
        tensor,
    )
    from helper import (
        rotate_vertices_2d_batch,
        generate_md5_from_tensors,
    )

    input_filename = sys.argv[1]
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"File {input_filename} does not exist.")

    input_filename_unique_id = input_filename.split(".")[0].split("_")[-1]
    # Load the scanner layout
    scanner_layout_data = torch_load(input_filename)
    # Extract the detector units and plate segments
    raw_detector_units = scanner_layout_data["detector units"]
    raw_plate_segments = scanner_layout_data["plate segments"]

    n_rotations = 24
    n_shifts = [5, 5] # number of steps translation in x and y
    shift_step = [1, 1] # in mm
    positions = positions_parameters(n_rotations, n_shifts, shift_step)
    n_positions = int(positions.shape[0])
    print(f"N positions: {n_positions}")
    # print(f"Positions:\n{positions}")

    transformed_detector_units = transform_to_positions_2d_batch(
        positions, raw_detector_units.view(-1, 2)
    ).reshape(
        n_positions,
        raw_detector_units.shape[0],
        raw_detector_units.shape[1],
        2,
    )
    transformed_plate_segments = transform_to_positions_2d_batch(
        positions, raw_plate_segments.view(-1, 2)
    ).reshape(
        n_positions,
        raw_plate_segments.shape[0],
        raw_plate_segments.shape[1],
        2,
    )

    # Generate a unique ID for the layouts
    unique_id = generate_md5_from_tensors(
        transformed_detector_units, transformed_plate_segments
    )

    # Save the transformed layouts
    out_data: OutDataDict = {"scanner MD5": input_filename_unique_id}
    layouts = {}
    for i in range(n_positions):
        layouts[f"position {i:03d}"] = {
            "position": positions[i],
            "detector units": transformed_detector_units[i],
            "plate segments": transformed_plate_segments[i],
        }
    out_data["layouts"] = layouts
    out_file_name = f"scanner_layouts_{unique_id}.tensor"
    print(f"Save to:\n  {out_file_name}")
    torch_save(
        out_data,
        out_file_name,
    )

    print("Data saved successfully.")
