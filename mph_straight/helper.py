import torch
from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes
from torch import Tensor, tensor, pi, cos, sin, arange, stack, asin
from math import cos as math_cos, sin as math_sin

def plot_polygons_from_vertices_2d_mpl(vertices: Tensor, ax: Axes, **kwargs):
    """Plots a collection of polygons on a matplotlib Axes object."""
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p

def rotate_and_repeat_4gon(input: Tensor, n: int, step: float) -> Tensor:
    """Rotates and repeats the vertices of quadrilaterals to form a full circle."""
    m_polygons = input.shape[0]
    rotations = arange(0, n, dtype=input.dtype, device=input.device) * step
    rotation_matrices = stack(
        (cos(rotations), -sin(rotations), sin(rotations), cos(rotations)),
        dim=1
    ).reshape(-1, 2, 2)

    tiled_input = input.repeat(n, 1, 1)
    reshaped_vertices = tiled_input.view(-1, 2).unsqueeze(-1)
    tiled_rotation_matrices = rotation_matrices.unsqueeze(1).repeat(
        1, m_polygons * 4, 1, 1
    ).view(-1, 2, 2)

    rotated_vertices = torch.bmm(
        tiled_rotation_matrices, reshaped_vertices
    ).view(-1, 4, 2)
    return rotated_vertices

def generate_transaxial_spect_geometry(
    pinhole_diameter_mm: float,
    n_pinholes: int,
    collimator_ring_radius_mm: float,
    collimator_thickness_mm: float,
    detector_ring_radius_mm: float,
    detector_thickness_mm: float,
    detector_crystal_width_mm: float,
) -> tuple[Tensor, Tensor]:
    """
    Generates a 2D SPECT geometry with physically accurate collimator segments.

    Returns:
        tuple[Tensor, Tensor]:
            - detector_units: Vertices for the detector ring.
            - collimator_segments: Vertices for the solid tapered collimator parts.
    """
    # --- Collimator Segments ---
    R_COLL_INNER = collimator_ring_radius_mm - collimator_thickness_mm / 2.0
    R_COLL_OUTER = collimator_ring_radius_mm + collimator_thickness_mm / 2.0
    angle_step_per_pinhole = 2 * pi / n_pinholes

    # FINAL CORRECTION: Calculate the physical opening angle based on the 3mm diameter,
    # not the 27-degree field-of-view angle.
    # The angle subtended by a chord (the pinhole diameter) at a given radius.
    opening_angle_rad = 2 * asin(torch.tensor((pinhole_diameter_mm / 2.0) / collimator_ring_radius_mm))

    # The rest of the logic for creating tapered segments remains correct.
    angle_left_wall  = (-angle_step_per_pinhole / 2.0) + (opening_angle_rad / 2.0)
    angle_right_wall = (+angle_step_per_pinhole / 2.0) - (opening_angle_rad / 2.0)

    single_segment_verts = tensor([[
        [R_COLL_INNER * cos(angle_left_wall), R_COLL_INNER * sin(angle_left_wall)],
        [R_COLL_OUTER * cos(angle_left_wall), R_COLL_OUTER * sin(angle_left_wall)],
        [R_COLL_OUTER * cos(angle_right_wall), R_COLL_OUTER * sin(angle_right_wall)],
        [R_COLL_INNER * cos(angle_right_wall), R_COLL_INNER * sin(angle_right_wall)],
    ]])

    collimator_segments = rotate_and_repeat_4gon(single_segment_verts, n_pinholes, angle_step_per_pinhole)

    # --- Detector Geometry ---
    R_DET_INNER = detector_ring_radius_mm
    R_DET_OUTER = detector_ring_radius_mm + detector_thickness_mm
    circumference_det = 2 * pi * (R_DET_INNER + detector_thickness_mm / 2.0)
    n_total_crystals = int(circumference_det / detector_crystal_width_mm)
    angle_step_per_crystal = 2 * pi / n_total_crystals
    angular_width_crystal = detector_crystal_width_mm / R_DET_INNER
    angle_start_crystal = -angular_width_crystal / 2.0
    angle_end_crystal = angular_width_crystal / 2.0

    crystal_vertices = tensor([[
        [R_DET_INNER * math_cos(angle_start_crystal), R_DET_INNER * math_sin(angle_start_crystal)],
        [R_DET_OUTER * math_cos(angle_start_crystal), R_DET_OUTER * math_sin(angle_start_crystal)],
        [R_DET_OUTER * math_cos(angle_end_crystal),   R_DET_OUTER * math_sin(angle_end_crystal)],
        [R_DET_INNER * math_cos(angle_end_crystal),   R_DET_INNER * math_sin(angle_end_crystal)],
    ]])

    detector_units = rotate_and_repeat_4gon(crystal_vertices, n_total_crystals, angle_step_per_crystal)

    return detector_units, collimator_segments