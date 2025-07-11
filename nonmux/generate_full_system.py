# generate_full_system.py

import torch
import math
from torch import Tensor, tensor, pi, cos, sin, arange, stack, deg2rad, save as torch_save
from helper import generate_md5_from_tensors # Assuming original helper.py is present for hashing
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes


# ==============================================================================
# PART 1: HELPER FUNCTIONS (Copied from both projects for a self-contained script)
# ==============================================================================

def plot_polygons_from_vertices_2d_mpl(vertices: Tensor, ax: Axes, **kwargs):
    """Plots a collection of polygons on a matplotlib Axes object."""
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p

def generate_detector_layout(detector_cfg: dict) -> tuple[Tensor, Tensor]:
    """Generates the 12-panel flat detector array from our previous work (v8)."""
    print("--- Generating Detector Panels (Hexagonal Layout) ---")
    
    # Unpack config
    BASE_RADIUS_MAIN = detector_cfg['base_radius_main']
    CRYSTAL_PITCH = detector_cfg['crystal_pitch']
    SAFETY_MARGIN = detector_cfg['safety_margin']
    N_CRYSTALS_PER_BLOCK = detector_cfg['n_crystals_per_block']
    N_LAYERS = detector_cfg['n_layers']
    LAYER_THICKNESS = detector_cfg['layer_thickness']
    CRYSTAL_SIZE = detector_cfg['crystal_size']
    population_scheme = tensor(detector_cfg['population_scheme'], dtype=torch.float32)

    # Non-collision logic
    main_panel_width_mm = (4 * N_CRYSTALS_PER_BLOCK) * CRYSTAL_PITCH
    half_width = main_panel_width_mm / 2.0
    r_corner_max = math.sqrt(BASE_RADIUS_MAIN**2 + half_width**2)
    BASE_RADIUS_GAP = r_corner_max + SAFETY_MARGIN

    panel_definitions = []
    for i in range(6): panel_definitions.append({'blocks': 4, 'angle_deg': i * 60.0, 'base_radius': BASE_RADIUS_MAIN})
    for i in range(6): panel_definitions.append({'blocks': 1, 'angle_deg': (i * 60.0) + 30.0, 'base_radius': BASE_RADIUS_GAP})

    all_detector_vertices = []
    all_plate_segments = []

    for panel_def in panel_definitions:
        tangential_n_slots = panel_def['blocks'] * N_CRYSTALS_PER_BLOCK
        panel_center_angle_rad = math.radians(panel_def['angle_deg'])
        base_radius = panel_def['base_radius']
        panel_local_vertices = []
        for k in range(N_LAYERS):
            n_active = int(torch.round(population_scheme[k] * (tangential_n_slots / 32.0)))
            if n_active > 0:
                indices = torch.unique(torch.linspace(0, tangential_n_slots - 1, n_active).round().long())
            else: continue
            local_y = (indices - (tangential_n_slots - 1) / 2.0) * CRYSTAL_PITCH
            local_x = torch.full_like(local_y.float(), base_radius + (k * LAYER_THICKNESS))
            half_size = CRYSTAL_SIZE / 2.0
            corner_offsets = tensor([[-half_size, -half_size], [half_size, -half_size], [half_size, half_size], [-half_size, half_size]])
            local_centers = torch.stack([local_x, local_y], dim=1)
            panel_local_vertices.append(local_centers.unsqueeze(1) + corner_offsets)
        if not panel_local_vertices: continue
        panel_local_vertices = torch.cat(panel_local_vertices, dim=0)
        c, s = math.cos(panel_center_angle_rad), math.sin(panel_center_angle_rad)
        rot_matrix = tensor([[c, -s], [s, c]])
        all_detector_vertices.append(torch.matmul(panel_local_vertices.view(-1, 2), rot_matrix).view(-1, 4, 2))
        panel_tangential_width = tangential_n_slots * CRYSTAL_PITCH
        panel_radial_depth = N_LAYERS * LAYER_THICKNESS
        corners_local = tensor([[base_radius, -panel_tangential_width / 2.0], [base_radius + panel_radial_depth, -panel_tangential_width / 2.0], [base_radius + panel_radial_depth, panel_tangential_width / 2.0], [base_radius, panel_tangential_width / 2.0]])
        all_plate_segments.append(torch.matmul(corners_local, rot_matrix))
    return torch.cat(all_detector_vertices, dim=0), torch.stack(all_plate_segments)

def generate_ring_collimator_plates(collimator_cfg: dict) -> Tensor:
    """Generates the solid plates of the bi-conical ring collimator."""
    print("\n--- Generating Ring Collimator ---")
    
    # Unpack config
    pinhole_diameter = collimator_cfg['pinhole_diameter_mm']
    opening_angle_deg = collimator_cfg['pinhole_opening_angle_deg']
    n_pinholes = collimator_cfg['n_pinholes']
    ring_radius = collimator_cfg['collimator_ring_radius_mm']
    
    # Assume symmetric thickness for the two rings
    collimator_thickness = 2.0 # A reasonable default, can be added to cfg
    single_ring_thickness = collimator_thickness / 2.0
    
    # Inner ring radii
    R_inner_coll_inner = ring_radius - single_ring_thickness
    R_inner_coll_outer = ring_radius
    # Outer ring radii
    R_outer_coll_inner = ring_radius
    R_outer_coll_outer = ring_radius + single_ring_thickness

    # Geometric calculations for pinhole shape
    angle_step_per_pinhole = 2 * pi / n_pinholes
    opening_angle_rad = deg2rad(tensor(opening_angle_deg, dtype=torch.float32))
    half_aperture_at_junction = pinhole_diameter / 2.0
    taper_offset = single_ring_thickness * torch.tan(opening_angle_rad / 2.0)
    half_width_at_wide_side = half_aperture_at_junction + taper_offset

    # --- START OF CORRECTION ---
    # Angular half-width of the PINHOLE VOID at each surface
    # We must cast the float result to a tensor before passing to torch.asin()
    angular_half_width_inner_wide = torch.asin(tensor(half_width_at_wide_side / R_inner_coll_inner))
    angular_half_width_inner_narrow = torch.asin(tensor(half_aperture_at_junction / R_inner_coll_outer))
    angular_half_width_outer_narrow = torch.asin(tensor(half_aperture_at_junction / R_outer_coll_inner))
    angular_half_width_outer_wide = torch.asin(tensor(half_width_at_wide_side / R_outer_coll_outer))
    # --- END OF CORRECTION ---

    all_inner_segments, all_outer_segments = [], []
    for i in range(n_pinholes):
        current_angle = i * angle_step_per_pinhole
        next_angle = (i + 1) * angle_step_per_pinhole
        # Define the SOLID segment that lives BETWEEN two pinhole voids
        # 1. Inner Collimator Segment
        v1_a = current_angle + angular_half_width_inner_narrow
        v2_a = current_angle + angular_half_width_inner_wide
        v3_a = next_angle - angular_half_width_inner_wide
        v4_a = next_angle - angular_half_width_inner_narrow
        v1 = [R_inner_coll_outer * cos(v1_a), R_inner_coll_outer * sin(v1_a)]
        v2 = [R_inner_coll_inner * cos(v2_a), R_inner_coll_inner * sin(v2_a)]
        v3 = [R_inner_coll_inner * cos(v3_a), R_inner_coll_inner * sin(v3_a)]
        v4 = [R_inner_coll_outer * cos(v4_a), R_inner_coll_outer * sin(v4_a)]
        all_inner_segments.append(tensor([v1, v2, v3, v4]))
        # 2. Outer Collimator Segment
        v1_ao = current_angle + angular_half_width_outer_narrow
        v2_ao = current_angle + angular_half_width_outer_wide
        v3_ao = next_angle - angular_half_width_outer_wide
        v4_ao = next_angle - angular_half_width_outer_narrow
        v1_o = [R_outer_coll_inner * cos(v1_ao), R_outer_coll_inner * sin(v1_ao)]
        v2_o = [R_outer_coll_outer * cos(v2_ao), R_outer_coll_outer * sin(v2_ao)]
        v3_o = [R_outer_coll_outer * cos(v3_ao), R_outer_coll_outer * sin(v3_ao)]
        v4_o = [R_outer_coll_inner * cos(v4_ao), R_outer_coll_inner * sin(v4_ao)]
        all_outer_segments.append(tensor([v1_o, v2_o, v3_o, v4_o]))

    # Weld the two ring segments into single hexagonal plates
    inner_plates = torch.stack(all_inner_segments)
    outer_plates = torch.stack(all_outer_segments)
    
    hexagonal_plates = []
    for i in range(n_pinholes):
        # This simplified welding logic assumes the vertex ordering from the calculation above
        all_6_verts = torch.cat([inner_plates[i, [1, 2]], outer_plates[i, [1, 2]], inner_plates[i, [0, 3]]], dim=0)
        center = torch.mean(all_6_verts, dim=0)
        angles = torch.atan2(all_6_verts[:, 1] - center[1], all_6_verts[:, 0] - center[0])
        hexagonal_plates.append(all_6_verts[torch.argsort(angles)])
    
    return torch.stack(hexagonal_plates)

# ==============================================================================
# PART 2: MAIN SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- A. Define Configuration for DETECTORS ---
    detector_cfg = {
        'base_radius_main': 133.0,
        'crystal_pitch': 3.36,
        'safety_margin': 2.0,
        'n_crystals_per_block': 8,
        'n_layers': 8,
        'layer_thickness': 3.36,
        'crystal_size': 2.4,
        'population_scheme': [12, 15, 18, 21, 24, 27, 30, 32],
    }

    # --- B. Define Configuration for COLLIMATOR ---
    # Using parameters from your professor's instructions
    collimator_cfg = {
        'collimator_ring_radius_mm': 93.0,
        'pinhole_diameter_mm': 2.0,
        'pinhole_opening_angle_deg': 70.0,
        'aperture_opening_ratio': 0.5, # The key parameter to tune
    }

    # Calculate the number of pinholes based on the ratio
    n_pinholes_float = (collimator_cfg['aperture_opening_ratio'] * 2 * math.pi * collimator_cfg['collimator_ring_radius_mm']) / collimator_cfg['pinhole_diameter_mm']
    collimator_cfg['n_pinholes'] = int(round(n_pinholes_float))
    
    print("--- System Configuration ---")
    print(f"Desired Aperture Ratio: {collimator_cfg['aperture_opening_ratio']:.2%}")
    print(f"Calculated Number of Pinholes: {collimator_cfg['n_pinholes']}")


    # --- C. Generate all system components ---
    detector_units, detector_plates = generate_detector_layout(detector_cfg)
    collimator_plates = generate_ring_collimator_plates(collimator_cfg)

    # --- D. Save the final combined layout ---
    unique_id = generate_md5_from_tensors(detector_units, detector_plates, collimator_plates)
    out_file_name = f"full_system_{unique_id}.tensor"
    print(f"\nSaving complete system configuration to:\n  {out_file_name}")

    torch_save({
        "detector units": detector_units,
        "detector plates": detector_plates,
        "collimator plates": collimator_plates,
    }, out_file_name)
    print("System saved successfully.")

    # --- E. Visualize the final system ---
    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    plot_polygons_from_vertices_2d_mpl(detector_units, ax, facecolor='#1f77b4', edgecolor='#1f77b4', linewidth=0.2, label='Detector Units')
    #plot_polygons_from_vertices_2d_mpl(detector_plates, ax, facecolor='none', edgecolor='#17becf', linewidth=1.5, label='Detector Plates')
    plot_polygons_from_vertices_2d_mpl(collimator_plates, ax, facecolor='gray', edgecolor='black', label='Collimator Plates')

    ax.set_aspect('equal', 'box')
    plot_limit = detector_cfg['base_radius_main'] * 1.5
    ax.set_xlim([-plot_limit, plot_limit])
    ax.set_ylim([-plot_limit, plot_limit])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Full System Layout: Detectors + Ring Collimator")
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    viz_file_name = f"full_system_{unique_id}.png"
    plt.savefig(viz_file_name)
    print(f"Visualization saved to {viz_file_name}")