# helper.py (Corrected)
import torch
from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes
from torch import Tensor, tensor, pi, cos, sin, arange, stack, asin, tan, deg2rad
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
    pinhole_opening_angle_deg: float,
    n_pinholes: int,
    collimator_ring_radius_mm: float,
    collimator_thickness_mm: float,
    detector_ring_radius_mm: float,
    detector_thickness_mm: float,
    detector_crystal_width_mm: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Generates a 2D SPECT geometry with a correctly focused bi-conical 
    (double-tapered) pinhole across two concentric collimator rings.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - detector_units
            - inner_collimator_segments
            - outer_collimator_segments
    """
    # --- Define Radii for the Two Collimator Rings ---
    single_ring_thickness = collimator_thickness_mm / 2.0
    center_junction_radius = collimator_ring_radius_mm

    R_inner_coll_inner = center_junction_radius - single_ring_thickness
    R_inner_coll_outer = center_junction_radius

    R_outer_coll_inner = center_junction_radius
    R_outer_coll_outer = center_junction_radius + single_ring_thickness

    # --- Common Geometric Calculations ---
    angle_step_per_pinhole = 2 * pi / n_pinholes
    opening_angle_rad = deg2rad(tensor(pinhole_opening_angle_deg, dtype=torch.float32))
    half_aperture_at_junction = pinhole_diameter_mm / 2.0
    
    taper_offset = single_ring_thickness * tan(opening_angle_rad / 2.0).item()
    half_width_at_wide_side = half_aperture_at_junction + taper_offset

    # Angular half-width of the pinhole void at each collimator surface
    angular_half_width_inner_wide = asin(tensor(half_width_at_wide_side / R_inner_coll_inner))
    angular_half_width_inner_narrow = asin(tensor(half_aperture_at_junction / R_inner_coll_outer))
    angular_half_width_outer_narrow = asin(tensor(half_aperture_at_junction / R_outer_coll_inner))
    angular_half_width_outer_wide = asin(tensor(half_width_at_wide_side / R_outer_coll_outer))

    # --- Generate Collimator Segments inside a Loop for Correct Focusing ---
    all_inner_segments = []
    all_outer_segments = []

    for i in range(n_pinholes):
        # Calculate the center angle for the CURRENT pinhole
        center_angle = i * angle_step_per_pinhole

        # --- 1. Generate Inner Collimator Segment for this pinhole ---
        # Calculate the absolute angles for the solid segment edges
        angle_seg_start_inner_ring = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_inner_wide
        angle_seg_end_inner_ring = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_inner_wide
        angle_seg_start_outer_ring = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_inner_narrow
        angle_seg_end_outer_ring = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_inner_narrow
        
        inner_verts = tensor([[
            [R_inner_coll_inner * cos(angle_seg_start_inner_ring), R_inner_coll_inner * sin(angle_seg_start_inner_ring)],
            [R_inner_coll_outer * cos(angle_seg_start_outer_ring), R_inner_coll_outer * sin(angle_seg_start_outer_ring)],
            [R_inner_coll_outer * cos(angle_seg_end_outer_ring),   R_inner_coll_outer * sin(angle_seg_end_outer_ring)],
            [R_inner_coll_inner * cos(angle_seg_end_inner_ring),   R_inner_coll_inner * sin(angle_seg_end_inner_ring)],
        ]])
        all_inner_segments.append(inner_verts)

        # --- 2. Generate Outer Collimator Segment for this pinhole ---
        # Calculate the absolute angles for the solid segment edges
        angle_seg_start_inner_ring_o = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_outer_narrow
        angle_seg_end_inner_ring_o = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_outer_narrow
        angle_seg_start_outer_ring_o = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_outer_wide
        angle_seg_end_outer_ring_o = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_outer_wide
        
        outer_verts = tensor([[
            [R_outer_coll_inner * cos(angle_seg_start_inner_ring_o), R_outer_coll_inner * sin(angle_seg_start_inner_ring_o)],
            [R_outer_coll_outer * cos(angle_seg_start_outer_ring_o), R_outer_coll_outer * sin(angle_seg_start_outer_ring_o)],
            [R_outer_coll_outer * cos(angle_seg_end_outer_ring_o),   R_outer_coll_outer * sin(angle_seg_end_outer_ring_o)],
            [R_outer_coll_inner * cos(angle_seg_end_inner_ring_o),   R_outer_coll_inner * sin(angle_seg_end_inner_ring_o)],
        ]])
        all_outer_segments.append(outer_verts)

    # Combine the lists of tensors into final output tensors
    inner_collimator_segments = torch.cat(all_inner_segments, dim=0)
    outer_colimator_segments = torch.cat(all_outer_segments, dim=0)

    # --- 3. Detector Geometry (This part was correct and remains unchanged) ---
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

    return detector_units, inner_collimator_segments, outer_colimator_segments