# helper.py (Updated for 3D compatibility / float64)
import torch
from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes
from torch import Tensor, tensor, pi, cos, sin, arange, stack, asin, tan, deg2rad

DTYPE = torch.float64  # match scanner_modeling 3D pipeline


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
        dim=1,
    ).reshape(-1, 2, 2)

    tiled_input = input.repeat(n, 1, 1)
    reshaped_vertices = tiled_input.view(-1, 2).unsqueeze(-1)
    tiled_rotation_matrices = rotation_matrices.unsqueeze(1).repeat(
        1, m_polygons * 4, 1, 1
    ).view(-1, 2, 2)

    rotated_vertices = torch.bmm(
        tiled_rotation_matrices,
        reshaped_vertices,
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
            - detector_units         (N_det, 4, 2)   float64
            - inner_collimator_segments (N_in, 4, 2) float64
            - outer_collimator_segments (N_out, 4, 2) float64

    Vertex order for the detector quad is:

        [ R_in * (cos a_start, sin a_start),
          R_out * (cos a_start, sin a_start),
          R_out * (cos a_end,   sin a_end),
          R_in * (cos a_end,   sin a_end) ]

    so vertices 0 and 3 are at the inner radius. After extrusion, this matches
    the detector front-face convention used in helper_3d and geometry_3d.
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
    opening_angle_rad = deg2rad(
        tensor(pinhole_opening_angle_deg, dtype=DTYPE)
    )
    half_aperture_at_junction = pinhole_diameter_mm / 2.0

    taper_offset = single_ring_thickness * tan(opening_angle_rad / 2.0).item()
    half_width_at_wide_side = half_aperture_at_junction + taper_offset

    # Angular half-width of the pinhole void at each collimator surface
    angular_half_width_inner_wide = asin(
        tensor(half_width_at_wide_side / R_inner_coll_inner, dtype=DTYPE)
    )
    angular_half_width_inner_narrow = asin(
        tensor(half_aperture_at_junction / R_inner_coll_outer, dtype=DTYPE)
    )
    angular_half_width_outer_narrow = asin(
        tensor(half_aperture_at_junction / R_outer_coll_inner, dtype=DTYPE)
    )
    angular_half_width_outer_wide = asin(
        tensor(half_width_at_wide_side / R_outer_coll_outer, dtype=DTYPE)
    )

    # --- Generate Collimator Segments inside a Loop for Correct Focusing ---
    all_inner_segments = []
    all_outer_segments = []

    for i in range(n_pinholes):
        # Calculate the center angle for the CURRENT pinhole
        center_angle = tensor(i, dtype=DTYPE) * angle_step_per_pinhole

        # --- 1. Generate Inner Collimator Segment for this pinhole ---
        angle_seg_start_inner_ring = (
            center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_inner_wide
        )
        angle_seg_end_inner_ring = (
            center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_inner_wide
        )
        angle_seg_start_outer_ring = (
            center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_inner_narrow
        )
        angle_seg_end_outer_ring = (
            center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_inner_narrow
        )

        inner_verts = tensor(
            [[
                [R_inner_coll_inner * cos(angle_seg_start_inner_ring),
                 R_inner_coll_inner * sin(angle_seg_start_inner_ring)],
                [R_inner_coll_outer * cos(angle_seg_start_outer_ring),
                 R_inner_coll_outer * sin(angle_seg_start_outer_ring)],
                [R_inner_coll_outer * cos(angle_seg_end_outer_ring),
                 R_inner_coll_outer * sin(angle_seg_end_outer_ring)],
                [R_inner_coll_inner * cos(angle_seg_end_inner_ring),
                 R_inner_coll_inner * sin(angle_seg_end_inner_ring)],
            ]],
            dtype=DTYPE,
        )
        all_inner_segments.append(inner_verts)

        # --- 2. Generate Outer Collimator Segment for this pinhole ---
        angle_seg_start_inner_ring_o = (
            center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_outer_narrow
        )
        angle_seg_end_inner_ring_o = (
            center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_outer_narrow
        )
        angle_seg_start_outer_ring_o = (
            center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_outer_wide
        )
        angle_seg_end_outer_ring_o = (
            center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_outer_wide
        )

        outer_verts = tensor(
            [[
                [R_outer_coll_inner * cos(angle_seg_start_inner_ring_o),
                 R_outer_coll_inner * sin(angle_seg_start_inner_ring_o)],
                [R_outer_coll_outer * cos(angle_seg_start_outer_ring_o),
                 R_outer_coll_outer * sin(angle_seg_start_outer_ring_o)],
                [R_outer_coll_outer * cos(angle_seg_end_outer_ring_o),
                 R_outer_coll_outer * sin(angle_seg_end_outer_ring_o)],
                [R_outer_coll_inner * cos(angle_seg_end_inner_ring_o),
                 R_outer_coll_inner * sin(angle_seg_end_inner_ring_o)],
            ]],
            dtype=DTYPE,
        )
        all_outer_segments.append(outer_verts)

    # Combine the lists of tensors into final output tensors
    inner_collimator_segments = torch.cat(all_inner_segments, dim=0)
    outer_collimator_segments = torch.cat(all_outer_segments, dim=0)

    # --- 3. Detector Geometry ---
    R_DET_INNER = detector_ring_radius_mm
    R_DET_OUTER = detector_ring_radius_mm + detector_thickness_mm

    # 1. Use the mid-radius for the most accurate circumference and width calculations.
    mid_radius_det = R_DET_INNER + detector_thickness_mm / 2.0
    circumference_det = 2 * pi * mid_radius_det

    # 2. Determine the number of crystals that fit without overlapping.
    n_total_crystals = int(circumference_det / detector_crystal_width_mm)

    # 3. Fixed angular width of a single crystal based on its physical size.
    angular_width_crystal = tensor(detector_crystal_width_mm / mid_radius_det, dtype=DTYPE)

    # 4. Total angular space occupied by all crystals.
    total_crystal_angular_span = n_total_crystals * angular_width_crystal

    # 5. Leftover angular space (total gap) and evenly distributed gap.
    total_angular_gap = tensor(2 * pi, dtype=DTYPE) - total_crystal_angular_span
    angular_gap_between_crystals = total_angular_gap / n_total_crystals

    # 6. Step size = crystal width + gap.
    angle_step_per_crystal = angular_width_crystal + angular_gap_between_crystals

    print(
        f"Placing {n_total_crystals} crystals with an evenly distributed angular gap of "
        f"{torch.rad2deg(angular_gap_between_crystals):.4f} degrees between each."
    )

    # 7. First crystal based on its intrinsic angular width.
    angle_start_crystal = -angular_width_crystal / 2.0
    angle_end_crystal = angular_width_crystal / 2.0

    # Vertex order: inner-start, outer-start, outer-end, inner-end
    crystal_vertices = tensor(
        [[
            [R_DET_INNER * cos(angle_start_crystal),
             R_DET_INNER * sin(angle_start_crystal)],
            [R_DET_OUTER * cos(angle_start_crystal),
             R_DET_OUTER * sin(angle_start_crystal)],
            [R_DET_OUTER * cos(angle_end_crystal),
             R_DET_OUTER * sin(angle_end_crystal)],
            [R_DET_INNER * cos(angle_end_crystal),
             R_DET_INNER * sin(angle_end_crystal)],
        ]],
        dtype=DTYPE,
    )

    detector_units = rotate_and_repeat_4gon(
        crystal_vertices,
        n_total_crystals,
        angle_step_per_crystal,
    )

    return detector_units, inner_collimator_segments, outer_collimator_segments


def translate_vertices_2d(vertices: Tensor, dx: float, dy: float) -> Tensor:
    """
    Translate a batch of polygons (N_polys, 4, 2) by (dx, dy) in mm.
    No inplace modification; returns a new tensor.
    """
    shift = torch.tensor([dx, dy], dtype=vertices.dtype, device=vertices.device)
    return vertices + shift


def make_transaxial_positions(
    n_pos: int,
    a_mm: float,
    b_mm: float,
    outward_shift_mm: float = 0.0,
) -> Tensor:
    """
    Evenly spaced points on an ellipse of semi-axes (a_mm, b_mm) with optional uniform
    outward radial shift to enlarge the convex hull (per Chen et al.):
      16->0 mm, 8->2 mm, 6->5 mm, 4->15 mm.
    Returns: (n_pos, 2) tensor of (x_mm, y_mm), float64.
    """
    thetas = torch.linspace(0, 2 * pi, steps=n_pos + 1, dtype=DTYPE)[:-1]
    xs = a_mm * torch.cos(thetas)
    ys = b_mm * torch.sin(thetas)
    pts = torch.stack([xs, ys], dim=1)  # (n_pos, 2)

    if outward_shift_mm != 0.0:
        radii = torch.linalg.norm(pts, dim=1, keepdims=True).clamp(min=1e-6)
        dirs = pts / radii
        pts = pts + outward_shift_mm * dirs

    return pts
