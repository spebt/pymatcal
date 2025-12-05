#!/usr/bin/env python
"""
debug_scgc_raytrace_3d.py

Advanced debugging tool that:
1. Loads the scanner geometry from a .tensor file.
2. Converts Hexahedra -> OBBs.
3. Casts a ray from a Voxel to a specific Detector.
4. CALCULATES intersections with collimator plates (Inner & Outer rings).
5. Visualizes the scene with color-coded hits (Red=Entry, Green=Exit).

Usage:
    python debug_scgc_raytrace_3d.py --layout 0 --det 1200 --layout_file ../data/scanner_layouts/mph_hourglass_single_position_base_3d_v2.tensor
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Import Scanner Modeling Logic ---
from scanner_modeling._config import DTYPE
from scanner_modeling.geometry_2d import load_scanner_layouts
from scanner_modeling.geometry_3d import load_scanner_geometry_3d_from_layout
# Import the actual ray tracing kernels
from scanner_modeling._geometry_3d._polyhedron import world_to_local_ray
from scanner_modeling._raytracer_3d._intersection_obb_3d import ray_obb_intersection_local

# ----------------------------------------------------------------------
# 1. Visualization Helpers (Hexahedra Rendering)
# ----------------------------------------------------------------------

def get_hex_faces(hex_tensor):
    """Convert (N, 8, 3) hexes to face list for Matplotlib."""
    if hex_tensor.numel() == 0:
        return []
    # Standard 6 faces of a hexahedron (0-3 bottom, 4-7 top)
    face_indices = np.array([
        [0, 1, 2, 3], [4, 5, 6, 7], # z-, z+
        [0, 1, 5, 4], [3, 2, 6, 7], # y-, y+
        [0, 4, 7, 3], [1, 2, 6, 5]  # x-, x+
    ])
    hex_np = hex_tensor.detach().cpu().numpy()
    all_faces = []
    for h in hex_np:
        all_faces.extend(h[face_indices])
    return all_faces

def plot_hexahedra(ax, hex_tensor, color, alpha=0.2, edge_color=None, label=None):
    """Adds a collection of hexahedra to the axes."""
    if hex_tensor.numel() == 0:
        return
    faces = get_hex_faces(hex_tensor)
    mesh = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=edge_color, label=label)
    ax.add_collection3d(mesh)

# ----------------------------------------------------------------------
# 2. Geometry Conversion (Hex -> OBB)
# ----------------------------------------------------------------------

def hexes_to_obb_block(hexes: torch.Tensor, material_index_val: int):
    """Convert hexahedra (N, 8, 3) into an OBB dictionary."""
    device = hexes.device
    N = hexes.shape[0]
    
    if N == 0:
        return {
            "centers": torch.empty((0, 3), dtype=DTYPE, device=device),
            "rotations": torch.empty((0, 3, 3), dtype=DTYPE, device=device),
            "half_sizes": torch.empty((0, 3), dtype=DTYPE, device=device),
        }

    hexes = hexes.to(dtype=DTYPE, device=device)
    centers = hexes.mean(dim=1)

    # Local axes from vertex 0
    v0, v1, v3, v4 = hexes[:, 0], hexes[:, 1], hexes[:, 3], hexes[:, 4]
    e_rad, e_tan, e_ax = v1 - v0, v3 - v0, v4 - v0

    # Orthonormalize
    u_rad = e_rad / torch.clamp(torch.linalg.norm(e_rad, dim=1, keepdim=True), min=1e-9)
    u_ax  = e_ax  / torch.clamp(torch.linalg.norm(e_ax,  dim=1, keepdim=True), min=1e-9)
    u_tan = torch.cross(u_ax, u_rad, dim=1)
    u_tan = u_tan / torch.clamp(torch.linalg.norm(u_tan, dim=1, keepdim=True), min=1e-9)
    
    rotations = torch.stack([u_rad, u_tan, u_ax], dim=-1)

    # Half sizes
    hs_x = (e_rad * u_rad).sum(dim=1).abs() * 0.5
    hs_y = (e_tan * u_tan).sum(dim=1).abs() * 0.5
    hs_z = (e_ax  * u_ax ).sum(dim=1).abs() * 0.5
    half_sizes = torch.stack([hs_x, hs_y, hs_z], dim=1)

    return {"centers": centers, "rotations": rotations, "half_sizes": half_sizes}

# ----------------------------------------------------------------------
# 3. Intersection Logic
# ----------------------------------------------------------------------

def check_intersections(ray_o, ray_d, obb_dict):
    """
    Check intersection of a single ray against a batch of OBBs.
    ray_o, ray_d: (3,)
    obb_dict: Output of hexes_to_obb_block
    """
    # Reshape ray for broadcasting: (1, 3)
    o_world = ray_o.view(1, 3)
    d_world = ray_d.view(1, 3)
    
    centers = obb_dict["centers"]     # (N, 3)
    rotations = obb_dict["rotations"] # (N, 3, 3)
    half_sizes = obb_dict["half_sizes"]

    # 1. Transform World Ray -> Local Ray for each object
    # Returns (1, N, 3)
    o_local, d_local = world_to_local_ray(o_world, d_world, centers, rotations)
    
    # Remove the ray batch dim since we have 1 ray: (N, 3)
    o_local = o_local.squeeze(0)
    d_local = d_local.squeeze(0)

    # 2. Run Intersection Kernel
    t_enter, t_exit, hit_mask = ray_obb_intersection_local(o_local, d_local, half_sizes)
    
    # 3. Collect Results
    # Filter for hits
    hit_indices = torch.nonzero(hit_mask).squeeze()
    if hit_indices.ndim == 0 and hit_indices.numel() == 1:
        hit_indices = hit_indices.unsqueeze(0)
        
    results = []
    if hit_indices.numel() > 0:
        for idx in hit_indices:
            idx = int(idx)
            t0 = t_enter[idx]
            t1 = t_exit[idx]
            
            # Calculate world hit points
            # P = O + t*D
            p_enter = ray_o + t0 * ray_d
            p_exit  = ray_o + t1 * ray_d
            
            results.append({
                "id": idx,
                "t_enter": t0.item(),
                "p_enter": p_enter.cpu().numpy(),
                "p_exit": p_exit.cpu().numpy()
            })
            
    # Sort by t_enter (distance from source)
    results.sort(key=lambda x: x["t_enter"])
    return results

# ----------------------------------------------------------------------
# 4. Main Script
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Debug Ray-Collimator Intersections")
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--det", type=int, required=True, help="Detector Index")
    parser.add_argument("--layout_file", type=str, required=True)
    parser.add_argument("--voxel", type=str, default="0,0,0", help="x,y,z mm")
    args = parser.parse_args()

    # --- Load Geometry ---
    if not os.path.exists(args.layout_file):
        raise FileNotFoundError(f"Missing file: {args.layout_file}")
    
    dir_name = os.path.dirname(args.layout_file)
    file_name = os.path.basename(args.layout_file)
    scanner_layouts, _ = load_scanner_layouts(dir_name, file_name)
    layout = scanner_layouts[f"position {args.layout:03d}"]

    # Get Tensors
    det_hex = layout["detector units 3d"].to(DTYPE)
    plate_hex = layout.get("plate segments 3d", torch.empty((0,8,3))).to(DTYPE)
    
    N_det = det_hex.shape[0]
    if args.det >= N_det:
        print(f"Error: Det index {args.det} out of bounds (0-{N_det-1})")
        return

    # --- Identify Inner vs Outer Plates ---
    # We assume 'plate segments 3d' contains both rings.
    # Heuristic: Calculate radial distance of centroids.
    if plate_hex.shape[0] > 0:
        plate_centers = plate_hex.mean(dim=1)
        radii = torch.norm(plate_centers[:, :2], dim=1)
        
        # Simple thresholding to split rings
        # Assuming bimodal distribution
        min_r, max_r = radii.min(), radii.max()
        mid_r = (min_r + max_r) / 2.0
        
        inner_mask = radii < mid_r
        outer_mask = radii >= mid_r
        
        in_hex = plate_hex[inner_mask]
        out_hex = plate_hex[outer_mask]
        
        # Map local indices back to global IDs for reporting
        inner_ids = torch.nonzero(inner_mask).squeeze()
        outer_ids = torch.nonzero(outer_mask).squeeze()
    else:
        in_hex = torch.empty((0,8,3))
        out_hex = torch.empty((0,8,3))

    # --- Convert to OBBs ---
    det_obbs = hexes_to_obb_block(det_hex, 1)
    in_obbs  = hexes_to_obb_block(in_hex, 0)
    out_obbs = hexes_to_obb_block(out_hex, 0)

    # --- Define Ray ---
    target_center = det_obbs["centers"][args.det]
    vx, vy, vz = map(float, args.voxel.split(","))
    P0 = torch.tensor([vx, vy, vz], dtype=DTYPE) # Voxel
    P1 = target_center                           # Detector Face Center
    D_vec = P1 - P0
    # Normalize D? The intersection logic works with non-normalized D (returns t param),
    # but calculating physical points P = O + t*D works regardless.
    # However, to check validity, t must be between 0 and 1 if D is full segment length.
    # But usually ray tracers treat D as direction. Let's stick to D = P1 - P0 (Segment vector).
    # Then t is [0, 1].

    # --- Perform Intersection Tests ---
    print(f"\n--- RAY DIAGNOSIS for Detector {args.det} ---")
    print(f"Ray Origin (Voxel): {P0.tolist()}")
    print(f"Ray Target (Det):   {P1.tolist()}")

    hits_in = check_intersections(P0, D_vec, in_obbs)
    hits_out = check_intersections(P0, D_vec, out_obbs)

    # --- Report Inner Ring ---
    if hits_in:
        print(f"INNER RING: [BLOCKED] Hit {len(hits_in)} plates.")
        for h in hits_in:
            g_id = inner_ids[h['id']].item() if inner_ids.ndim > 0 else inner_ids.item()
            print(f"   - Plate Global ID {g_id}: t={h['t_enter']:.4f}, Loc={np.round(h['p_enter'], 1)}")
    else:
        print("INNER RING: [PASSED] Ray clear.")

    # --- Report Outer Ring ---
    if hits_out:
        print(f"OUTER RING: [BLOCKED] Hit {len(hits_out)} plates.")
        for h in hits_out:
            g_id = outer_ids[h['id']].item() if outer_ids.ndim > 0 else outer_ids.item()
            print(f"   - Plate Global ID {g_id}: t={h['t_enter']:.4f}, Loc={np.round(h['p_enter'], 1)}")
    else:
        print("OUTER RING: [PASSED] Ray clear.")

    # --- Conclusion ---
    blocked = (len(hits_in) > 0) or (len(hits_out) > 0)
    if not blocked:
        print("\nRESULT: SUCCESS! Ray passed through pinholes.")
    else:
        print("\nRESULT: BLOCKED.")

    # ----------------------------------------------------------------------
    # 5. Plotting (Color Scheme: Gray/LightGray/SkyBlue/Magenta)
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Inner Collimator (Gray)
    if in_hex.numel() > 0:
        plot_hexahedra(ax, in_hex, color="gray", edge_color="black", alpha=0.35)

    # Plot Outer Collimator (LightGray)
    if out_hex.numel() > 0:
        plot_hexahedra(ax, out_hex, color="lightgray", edge_color="black", alpha=0.30)

    # Plot Detectors (SkyBlue)
    # Render all detectors? It's slow. Let's render target + neighbours + sample ring
    # Plot Target (SkyBlue, opaque)
    target_hex = det_hex[args.det:args.det+1]
    plot_hexahedra(ax, target_hex, color="skyblue", edge_color="navy", alpha=0.8)

    # Plot Background Ring (Points)
    # Exclude target
    indices = np.arange(N_det)
    mask = indices != args.det
    bg_centers = det_obbs["centers"][mask].cpu().numpy()
    ax.scatter(bg_centers[:,0], bg_centers[:,1], bg_centers[:,2], c="skyblue", s=2, alpha=0.2)

    # Plot Ray (Magenta)
    p0n, p1n = P0.numpy(), P1.numpy()
    ax.plot([p0n[0], p1n[0]], [p0n[1], p1n[1]], [p0n[2], p1n[2]], "m-", lw=3.0, label="Ray")

    # Plot Intersections
    # Red X = Entry, Green X = Exit
    def plot_hits(hit_list):
        for h in hit_list:
            ent = h['p_enter']
            ext = h['p_exit']
            ax.scatter([ent[0]], [ent[1]], [ent[2]], c='red', marker='x', s=100, linewidth=2, zorder=10)
            ax.scatter([ext[0]], [ext[1]], [ext[2]], c='green', marker='x', s=100, linewidth=2, zorder=10)
    
    plot_hits(hits_in)
    plot_hits(hits_out)

    # Plot Voxel & Target Center
    ax.scatter([p0n[0]], [p0n[1]], [p0n[2]], c="gold", s=100, marker="*", label="Voxel", zorder=10)
    ax.scatter([p1n[0]], [p1n[1]], [p1n[2]], c="lime", s=50, marker="o", label="Target", zorder=10)

    # Formatting
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"3D Ray-Collimator Intersection Debug\nDetector {args.det}")
    
    # Auto-scale
    all_pts = []
    if plate_hex.numel() > 0: all_pts.append(plate_hex.view(-1,3))
    all_pts.append(target_hex.view(-1,3))
    all_pts.append(P0.view(1,3))
    
    cat_pts = torch.cat(all_pts, dim=0).cpu().numpy()
    mins = cat_pts.min(axis=0)
    maxs = cat_pts.max(axis=0)
    center = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2 * 1.1
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.det}_viz.png")
    # plt.show()

if __name__ == "__main__":
    main()