import torch
import matplotlib.pyplot as plt
import os
import sys

# --- Imports from pymatcal ---
# Ensure your project root is in PYTHONPATH or sys.path
# sys.path.append("/path/to/your/project/root")

from scanner_modeling.geometry_2d import (
    load_scanner_layouts,               # [cite: 19]
    load_scanner_geometry_from_layout,  # 
    fov_tensor_dict,                    # [cite: 39]
)
from scanner_modeling._raytracer_2d._local_functions import (
    rays_2d_batch,      # 
    line_segments_t,    # 
)
from scanner_modeling._plot._mpl_plot_system import (
    plot_scanner_from_vertices_2d_mpl,  # 
)

# --- Configuration ---
VOXEL_POSITION = torch.tensor([0.0, 0.0])   # Source (0,0)
DETECTOR_ID_TO_TEST = 141                   # Target detector index
LAYOUT_PATH = "../data/scanner_layouts/mph_hourglass_single_position_base_2mm_20pinholes.tensor"
LAYOUT_IDX = 0                              # Position index (usually 000)
ZOOM_IN = False
ZOOM_BOX_SIZE = 40.0

def run_debug():
    # 1. Load Geometry using Library Functions
    if not os.path.exists(LAYOUT_PATH):
        raise FileNotFoundError(f"Layout file not found: {LAYOUT_PATH}")
    
    dirname = os.path.dirname(LAYOUT_PATH)
    filename = os.path.basename(LAYOUT_PATH)
    
    # Load raw data [cite: 19]
    layouts_data, _ = load_scanner_layouts(dirname, filename)
    
    # Process into vertices and edges 
    # Returns: plates_v, crystals_v, plates_e, crystals_e
    plates_verts, det_verts, plates_edges, det_edges = \
        load_scanner_geometry_from_layout(LAYOUT_IDX, layouts_data)

    print(f"Loaded geometry: {plates_verts.shape[0]} aperture segments, "
          f"{det_verts.shape[0]} detectors.")

    # 2. Define Ray
    # Target: Centroid of the detector crystal
    # Matches logic in ppdf_2d_local where pb_batch is mean of vertices [cite: 93]
    target_crystal_verts = det_verts[DETECTOR_ID_TO_TEST]
    target_point = target_crystal_verts.mean(dim=0)
    
    # Create Ray Tensor 
    # Input shapes: (1, 2) for both source and target
    # Output shape: (1, 1, 2, 2) -> (n_start, n_end, 2_points, 2_coords)
    ray_batch = rays_2d_batch(VOXEL_POSITION.unsqueeze(0), target_point.unsqueeze(0))
    
    # Flatten ray to (1, 2, 2) for intersection check
    ray_flat = ray_batch.view(-1, 2, 2)

    # 3. intersection Logic
    # Flatten plate edges to (N_total_edges, 2, 2) to check against all at once
    # plates_edges comes as (N_plates, 4, 2, 2)
    all_plate_edges_flat = plates_edges.view(-1, 2, 2)

    # Calculate 't' parameters 
    # Returns t values where 0 <= t <= 1 for hits, -1 for misses 
    t_values = line_segments_t(ray_flat, all_plate_edges_flat)
    
    # Filter for valid hits (t != -1)
    valid_mask = (t_values >= 0) & (t_values <= 1)
    valid_t = t_values[valid_mask]
    
    # Calculate intersection coordinates: P = Start + t * (End - Start)
    ray_start = ray_flat[0, 0]
    ray_vec = ray_flat[0, 1] - ray_start
    intersection_points = ray_start + valid_t.unsqueeze(1) * ray_vec

    print(f"Ray from {VOXEL_POSITION.tolist()} to {target_point.tolist()}")
    print(f"Found {len(intersection_points)} intersections with aperture.")

    # 4. Visualization 
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create default FOV dict for plotting context [cite: 39]
    fov = fov_tensor_dict(size_in_mm=(128, 128)) 
    
    # Use library plotter
    plot_scanner_from_vertices_2d_mpl(
        plates_verts, 
        det_verts, 
        ax, 
        fov, 
        plate_alpha=0.6, 
        crystal_alpha=0.2
    )

    # Overlay Debug Info
    # Ray line
    ax.plot([ray_start[0], ray_start[0] + ray_vec[0]], 
            [ray_start[1], ray_start[1] + ray_vec[1]], 
            'm-', label='Ray')
    
    # Source & Target
    ax.plot(VOXEL_POSITION[0], VOXEL_POSITION[1], 'y*', markersize=12, label='Source')
    ax.plot(target_point[0], target_point[1], 'go', markersize=8, label='Target')
    
    # Intersections
    if len(intersection_points) > 0:
        ax.plot(intersection_points[:, 0], intersection_points[:, 1], 
                'rx', markersize=10, mew=2, label='Intersection')

    # Zoom Logic
    if ZOOM_IN:
        mid_x = (VOXEL_POSITION[0] + target_point[0]) / 2
        mid_y = (VOXEL_POSITION[1] + target_point[1]) / 2
        ax.set_xlim(mid_x - ZOOM_BOX_SIZE/2, mid_x + ZOOM_BOX_SIZE/2)
        ax.set_ylim(mid_y - ZOOM_BOX_SIZE/2, mid_y + ZOOM_BOX_SIZE/2)

    ax.legend(loc='upper right')
    ax.set_title(f"Debug: Ray to Detector {DETECTOR_ID_TO_TEST}")
    plt.tight_layout()
    plt.savefig(f"debug_ray_det{DETECTOR_ID_TO_TEST}.png")
    print(f"Plot saved to debug_ray_det{DETECTOR_ID_TO_TEST}.png")

if __name__ == "__main__":
    run_debug()