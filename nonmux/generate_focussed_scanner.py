# generate_focused_scanner.py (v8 - Final Flat Panel)
import torch
import math
from helper import generate_md5_from_tensors

# --- 1. Define Core Scanner Parameters ---
N_LAYERS = 8
CRYSTAL_PITCH = 3.36
LAYER_THICKNESS = 3.36
BASE_RADIUS_MAIN = 133.0
CRYSTAL_SIZE = 2.4
N_CRYSTALS_PER_BLOCK = 8
SAFETY_MARGIN = 2.0 

# --- 2. Calculate Non-Collision Radii ---
main_panel_width_blocks = 4
main_panel_width_mm = (main_panel_width_blocks * N_CRYSTALS_PER_BLOCK) * CRYSTAL_PITCH
half_width = main_panel_width_mm / 2.0
r_corner_max = math.sqrt(BASE_RADIUS_MAIN**2 + half_width**2)
BASE_RADIUS_GAP = r_corner_max + SAFETY_MARGIN

print(f"Main panel radius: {BASE_RADIUS_MAIN:.2f} mm")
print(f"Gap panel radius (with safety margin): {BASE_RADIUS_GAP:.2f} mm")

# --- 3. Define Panel Configuration ---
panel_definitions = []
for i in range(6):
    panel_definitions.append({'blocks': 4, 'angle_deg': i * 60.0, 'base_radius': BASE_RADIUS_MAIN})
for i in range(6):
    panel_definitions.append({'blocks': 1, 'angle_deg': (i * 60.0) + 30.0, 'base_radius': BASE_RADIUS_GAP})

# --- 4. Define Population Scheme ---
population_scheme = torch.tensor([12, 15, 18, 21, 24, 27, 30, 32], dtype=torch.float32)

# --- 5. Main Generation Logic (Final Flat Panel Geometry) ---
all_detector_vertices = []
all_plate_segments = []

print("\nGenerating final scanner layout with FLAT PANEL geometry...")
for i, panel_def in enumerate(panel_definitions):
    tangential_n_slots = panel_def['blocks'] * N_CRYSTALS_PER_BLOCK
    panel_center_angle_rad = math.radians(panel_def['angle_deg'])
    base_radius = panel_def['base_radius']

    panel_local_vertices = []
    
    # *** FINAL LOGIC: Build a flat panel first, then rotate it. ***
    
    # 1. Generate all crystal centers in the panel's LOCAL, UN-ROTATED frame
    for k in range(N_LAYERS):
        base_n_active = population_scheme[k]
        n_active = int(torch.round(base_n_active * (tangential_n_slots / 32.0)))
        if n_active > 0:
            indices_float = torch.linspace(0, tangential_n_slots - 1, n_active)
            tangential_indices = torch.unique(indices_float.round().long())
        else:
            continue
            
        local_y = (tangential_indices - (tangential_n_slots - 1) / 2.0) * CRYSTAL_PITCH
        local_x = torch.full_like(local_y.float(), base_radius + (k * LAYER_THICKNESS))
        
        # 2. For each center, create its 4 local vertices (still un-rotated)
        half_size = CRYSTAL_SIZE / 2.0
        corner_offsets = torch.tensor([[-half_size, -half_size], [half_size, -half_size], [half_size, half_size], [-half_size, half_size]])
        
        local_centers = torch.stack([local_x, local_y], dim=1)
        crystal_local_vertices = local_centers.unsqueeze(1) + corner_offsets
        panel_local_vertices.append(crystal_local_vertices)

    if not panel_local_vertices: continue
    panel_local_vertices = torch.cat(panel_local_vertices, dim=0)

    # 3. Rotate the ENTIRE set of local vertices for the panel at once
    c, s = math.cos(panel_center_angle_rad), math.sin(panel_center_angle_rad)
    rot_matrix = torch.tensor([[c, -s], [s, c]])
    
    # Reshape for batch multiplication: (N*4, 2) -> rotate -> (N, 4, 2)
    panel_global_vertices = torch.matmul(panel_local_vertices.view(-1, 2), rot_matrix).view(-1, 4, 2)
    all_detector_vertices.append(panel_global_vertices)

    # 4. Create and rotate the plate segment outline
    panel_tangential_width = tangential_n_slots * CRYSTAL_PITCH
    panel_radial_depth = N_LAYERS * LAYER_THICKNESS
    v_min, v_max = base_radius, base_radius + panel_radial_depth
    u_min, u_max = -panel_tangential_width / 2.0, +panel_tangential_width / 2.0
    corners_local = torch.tensor([[v_min, u_min], [v_max, u_min], [v_max, u_max], [v_min, u_max]])
    corners_global = torch.matmul(corners_local, rot_matrix)
    all_plate_segments.append(corners_global)

# --- 6. Finalize Tensors and Save ---
final_detector_units = torch.cat(all_detector_vertices, dim=0)
final_plate_segments = torch.stack(all_plate_segments)
unique_id = generate_md5_from_tensors(final_detector_units, final_plate_segments)
out_file_name = f"scanner_flat_panel_final_{unique_id}.tensor"
print(f"\nSaving final FLAT PANEL scanner configuration to:\n  {out_file_name}")
torch.save({"detector units": final_detector_units, "plate segments": final_plate_segments}, out_file_name)
print("\nLayout generated. Please visualize the new file.")