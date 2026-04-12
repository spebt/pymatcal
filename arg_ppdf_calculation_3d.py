import argparse
import os
import h5py

import torch
import matplotlib.pyplot as plt
import time

from scanner_modeling._config import DTYPE

# 3D FOV + 3D geometry helpers
from scanner_modeling.geometry_3d import (
    fov_tensor_dict_3d,
    voxels_coordinates_3d,
    load_scanner_geometry_3d_from_layout,
    build_convex_union_convex_poly_block,
)

# 2D layout loader
from scanner_modeling.geometry_2d import load_scanner_layouts

# Phase 5 system-matrix builder
from scanner_modeling.raytracer_3d import compute_system_matrix_for_detector

# Use the hexahedra helper to get quads (faces)
from helper_3d import hexahedron_quads


def hexes_to_obb_block(
    hexes: torch.Tensor,
    material_index_val: int,
    # We remove scalar detector/voxel indices args here to handle them flexibly in main
):
    """
    Convert hexahedra (N, 8, 3) into an OBB block.
    Returns only the geometric OBB parts. Metadata is assigned in main.
    """
    device = hexes.device
    dtype = DTYPE

    N = hexes.shape[0]
    if N == 0:
        # Return empty dict (omitted for brevity, same as before)
        pass 

    hexes = hexes.to(dtype=dtype, device=device)

    # Center of each hex
    centers = hexes.mean(dim=1)  # (N, 3)

    # Define local axes via three edges from vertex 0
    v0 = hexes[:, 0, :]
    v1 = hexes[:, 1, :]
    v3 = hexes[:, 3, :]
    v4 = hexes[:, 4, :]

    e_rad = v1 - v0   # radial
    e_tan = v3 - v0   # tangential
    e_ax = v4 - v0    # axial

    # --- Enforce Orthonormality ---
    u_rad = e_rad / torch.clamp(torch.linalg.norm(e_rad, dim=1, keepdim=True), min=1e-9)
    u_ax  = e_ax / torch.clamp(torch.linalg.norm(e_ax, dim=1, keepdim=True), min=1e-9)
    u_tan = torch.cross(u_ax, u_rad, dim=1) # Cross product ensures orthogonality
    u_tan = u_tan / torch.clamp(torch.linalg.norm(u_tan, dim=1, keepdim=True), min=1e-9)
    
    rotations = torch.stack([u_rad, u_tan, u_ax], dim=-1)  # (N, 3, 3)

    # Projected Half Sizes
    half_size_x = (e_rad * u_rad).sum(dim=1).abs() * 0.5
    half_size_y = (e_tan * u_tan).sum(dim=1).abs() * 0.5 
    half_size_z = (e_ax * u_ax).sum(dim=1).abs() * 0.5
    
    half_sizes = torch.stack([half_size_x, half_size_y, half_size_z], dim=1)

    # Material metadata
    material_index = torch.full(
        (N,), int(material_index_val), dtype=torch.int64, device=device
    )

    return {
        "centers": centers,
        "rotations": rotations,
        "half_sizes": half_sizes,
        "material_index": material_index,
        # We purposefully omit detector indices here to assign them in main
    }

def main():
    parser = argparse.ArgumentParser(description="Compute System Matrix for ONE Detector")
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    # CHANGED: This now selects the specific crystal ID (e.g. 1200)
    parser.add_argument("--target-det", type=int, required=True, 
                        help="Index of the single detector to simulate (0..1299)")
    
    parser.add_argument("--layout-file", type=str, default="../data/scanner_layouts/sct_single_position_base_3d.tensor")
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()
    
    # ---------------------------------------------------------
    # 1) Load Layout
    # ---------------------------------------------------------
    layout_dir = os.path.dirname(args.layout_file)
    layout_filename = os.path.basename(args.layout_file)
    scanner_layouts, uid = load_scanner_layouts(layout_dir, layout_filename)

    # Also load the raw tensor file to read top-level metadata (mu values,
    # hole prisms) that live outside the "layouts" sub-dict.
    raw_layout = torch.load(args.layout_file, weights_only=False)
    applied_cfg  = raw_layout.get("applied_config", {})
    hole_prisms  = raw_layout.get("collimator hole prisms", None)

    # Per-material attenuation (mm^-1).  Fall back to MPH defaults if absent.
    mu_plate = float(applied_cfg.get("mu_plate_mm_inv",    3.5))
    mu_det   = float(applied_cfg.get("mu_detector_mm_inv", 0.475))

    key = f"position {args.layout:03d}"
    layout_entry = scanner_layouts[key]
    det_hex = layout_entry["detector units 3d"].to(DTYPE)   # (N_det, 8, 3)
    plate_hex = layout_entry["plate segments 3d"].to(DTYPE) # (N_plate, 8, 3)

    N_det_total = det_hex.shape[0]
    device = det_hex.device
    print(f"Total Detectors in Ring: {N_det_total}")

    if args.target_det >= N_det_total:
        raise ValueError(f"Target detector {args.target_det} out of range (Max {N_det_total-1})")

    # ---------------------------------------------------------
    # 2) Assign IDs: Each Crystal is a Unique Detector
    # ---------------------------------------------------------
    # Detectors: Index 0 to 1299
    det_block = hexes_to_obb_block(det_hex, material_index_val=1)
    det_block["detector_index"] = torch.arange(N_det_total, dtype=torch.int64, device=device)
    det_block["detector_voxel_index"] = torch.zeros(N_det_total, dtype=torch.int64, device=device) # 1 voxel per det

    # Plates: Not detectors
    plate_block = hexes_to_obb_block(plate_hex, material_index_val=0)
    N_plate = plate_hex.shape[0]
    plate_block["detector_index"] = torch.full((N_plate,), -1, dtype=torch.int64, device=device)
    plate_block["detector_voxel_index"] = torch.full((N_plate,), -1, dtype=torch.int64, device=device)

    # Combine for the Physics Engine (Objects Dict)
    # The engine needs ALL objects to calculate attenuation correctly
    def concat_blocks(a, b): return torch.cat([a, b], dim=0)

    obb_layout = {
        "centers": concat_blocks(det_block["centers"], plate_block["centers"]),
        "rotations": concat_blocks(det_block["rotations"], plate_block["rotations"]),
        "half_sizes": concat_blocks(det_block["half_sizes"], plate_block["half_sizes"]),
        "material_index": concat_blocks(det_block["material_index"], plate_block["material_index"]),
        "detector_index": concat_blocks(det_block["detector_index"], plate_block["detector_index"]),
        "detector_voxel_index": concat_blocks(det_block["detector_voxel_index"], plate_block["detector_voxel_index"]),
    }

    # ------------------------------------------------------------------
    # Option A hole mechanism: load octagonal prism descriptors from the
    # tensor file and register them as CONVEX_POLY objects with
    # mu = -mu_plate.  The negative mu causes index_add_ in ppdf_3d_local
    # to subtract the hole path length from the plate contribution, giving
    # full transmission (exp(0)=1) through a hole and partial transmission
    # at grazing incidence.
    #
    # If the layout has no hole prisms (e.g. MPH scanner), this block is
    # skipped entirely and behaviour is unchanged.
    # ------------------------------------------------------------------
    MATERIAL_PLATE    = 0
    MATERIAL_DETECTOR = 1
    MATERIAL_HOLE_VOID = 2   # only used when hole_prisms is present

    holes_poly_block = None
    if hole_prisms is not None:
        M_holes = hole_prisms["centers"].shape[0]
        print(f"Loading {M_holes} hole prisms (Option A, negative-mu subtractive geometry)...")

        holes_poly_block = build_convex_union_convex_poly_block(
            centers             = hole_prisms["centers"].to(device=device),
            rotations           = hole_prisms["rotations"].to(device=device),
            plane_normals_local = hole_prisms["plane_normals_local"].to(device=device),
            plane_offsets_local = hole_prisms["plane_offsets_local"].to(device=device),
            material_index      = MATERIAL_HOLE_VOID,
            detector_index      = -1,
            detector_voxel_index = -1,
            num_planes          = hole_prisms["num_planes"].to(device=device),
        )
        # Add tight local AABBs so the broad phase can cull the 1218 small
        # prisms efficiently (without this, every ray would check all holes).
        if "poly_local_aabb_min" in hole_prisms:
            holes_poly_block["poly_local_aabb_min"] = hole_prisms["poly_local_aabb_min"].to(device=device)
            holes_poly_block["poly_local_aabb_max"] = hole_prisms["poly_local_aabb_max"].to(device=device)

    entry = {"obb": obb_layout}
    if holes_poly_block is not None:
        entry["convex_poly"] = holes_poly_block

    scanner_layouts_for_objects = {key: entry}
    objects = load_scanner_geometry_3d_from_layout(args.layout, scanner_layouts_for_objects)

    # ---------------------------------------------------------
    # 3) Select ONLY the Target Detector for Rays
    # ---------------------------------------------------------
    # We only want to compute PPDFs for the SINGLE target detector
    target_idx = args.target_det
    
    # Extract center for just this one detector
    target_center = det_block["centers"][target_idx].unsqueeze(0) # (1, 3)
    
    # Extract Face Geometry for Solid Angle (Just for target)
    # Include all faces EXCEPT the back face (farthest from FOV).
    quads = hexahedron_quads(det_hex[target_idx:target_idx+1]) # (1, 6, 4, 3)
    face_centers = quads.mean(dim=2)
    # Use full 3-D distance from the FOV centre (origin) to identify the back face.
    # The back face is whichever face centre is geometrically farthest from the FOV.
    # This works for both ring scanners (outer radial face) and flat panels (+Y face)
    # without relying on a 2-D XY projection, which could mis-identify the back face
    # for detectors with large X or Z offsets relative to their Y depth.
    dist3d = torch.linalg.norm(face_centers, dim=-1)        # (1, 6)
    back_face_idx = int(dist3d.argmax(dim=1)[0].item())

    keep = [f for f in range(6) if f != back_face_idx]  # 5 face indices
    target_voxel_faces = torch.empty((1, 5, 2, 3, 3), dtype=DTYPE, device=device)
    for slot, face_idx in enumerate(keep):
        q = quads[0, face_idx]  # (4, 3)
        target_voxel_faces[0, slot, 0] = torch.stack([q[0], q[1], q[2]], dim=0)
        target_voxel_faces[0, slot, 1] = torch.stack([q[0], q[2], q[3]], dim=0)

    # ---------------------------------------------------------
    # 4) Define FOV and Materials
    # ---------------------------------------------------------
    fov_dict = fov_tensor_dict_3d(
        n_voxels=(128, 128, 128), # Adjust grid size here
        size_in_mm=(32.0, 32.0, 32.0),
        center_coordinates=(0.0, 0.0, 0.0),
        n_subdivisions=(1, 1, 1),
    )
    Nx, Ny, Nz = [int(v.item()) for v in fov_dict["n voxels"]]
    
    # Map Materials
    # Index 0 = plate, 1 = detector, 2 = hole void (negative, subtractive).
    # mu values come from applied_config so the same script works for both
    # MPH (defaults: 3.5 / 0.475) and SCT (4.0 / 0.35 / -4.0) layouts.
    if hole_prisms is not None:
        mu_table = torch.tensor([mu_plate, mu_det, -mu_plate], dtype=DTYPE, device=device)
    else:
        mu_table = torch.tensor([mu_plate, mu_det], dtype=DTYPE, device=device)
    mu_objects_expanded = mu_table[objects['material_index']]
    mu_detector = torch.tensor(mu_det, dtype=DTYPE, device=device)

    # ---------------------------------------------------------
    # 5) Compute System Matrix (Target Only)
    # ---------------------------------------------------------
    print(f"Computing System Matrix for Detector {target_idx}...")
    start_time = time.time()
    triples = compute_system_matrix_for_detector(
        det_index=target_idx,
        det_voxel_centers=target_center, # Only 1 center passed here!
        voxel_faces=target_voxel_faces,  # Only 1 face set passed here!
        fov_dict=fov_dict,
        objects=objects,                 # BUT full object list passed for attenuation
        mu_objects=mu_objects_expanded,
        mu_detector=float(mu_detector.item()),
        max_rays_per_chunk=50000,
        device=device,
    )
    end_time = time.time()
    print(f"{end_time - start_time}s time taken.")
    # ---------------------------------------------------------
    # 6) Save
    # ---------------------------------------------------------
    # Matrix size: 1 (row) x N_FOV (cols)
    # Since we are only doing 1 detector, we can save a compact vector or 2D slice
    print(f"Computation complete. Found {len(triples)} non-zero entries.")
    
    # Reconstruct dense volume for plotting/saving
    fov_n_vox = Nx * Ny * Nz
    ppdf_volume = torch.zeros(fov_n_vox, dtype=DTYPE, device=device)

    if triples:
        cols_t = torch.tensor([c for _, c, _ in triples], dtype=torch.long, device=device)
        vals_t = torch.tensor([v for _, _, v in triples], dtype=DTYPE, device=device)
        ppdf_volume.scatter_(0, cols_t, vals_t)

    ppdf_volume = ppdf_volume.view(Nx, Ny, Nz)
    output_path = args.output
    if os.path.isdir(output_path):
        # If the user provides a directory, create a default filename
        filename = f"ppdf_det_{args.target_det}_layout_{args.layout}.h5"
        output_path = os.path.join(output_path, filename)
        
    with h5py.File(output_path, "w") as f:
        f.create_dataset("ppdf_volume", data=ppdf_volume.cpu().numpy(), compression="gzip")
        f.attrs["target_detector"] = target_idx
# ---------------------------------------------------------
    # 7) Plot in Millimeters (mm)
    # ---------------------------------------------------------
    if not args.no_plot:
        z_mid_idx = Nz // 2
        slice_xy = ppdf_volume[:, :, z_mid_idx].cpu().numpy()

        # Calculate physical extent for axes
        # extent = [left, right, bottom, top]
        # Since center is (0,0), range is [-size/2, size/2]
        
        fov_size_x = fov_dict["size in mm"][0].item()
        fov_size_y = fov_dict["size in mm"][1].item()
        
        extent = [
            -fov_size_x / 2.0,  # Left (mm)
             fov_size_x / 2.0,  # Right (mm)
            -fov_size_y / 2.0,  # Bottom (mm)
             fov_size_y / 2.0   # Top (mm)
        ]

        plt.figure(figsize=(7, 6))
        
        # origin="lower" puts (0,0) index at bottom-left.
        # With extent set, matplotlib maps indices to physical coordinates.
        im = plt.imshow(
            slice_xy.T, # Transpose if needed to match x-axis horizontal
            origin="lower", 
            interpolation="nearest",
            extent=extent
        )
        
        plt.colorbar(im, label="Probability Density")
        plt.title(f"Sensitivity: Det {target_idx}, Z={0.0} mm") # Assuming Z center is 0
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")
        
        # Optional: Add grid for reference
        plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)

        plot_file = args.output.replace(".hdf5", ".png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        print(f"Plot (in mm) saved to {plot_file}")
        # plt.show()

if __name__ == "__main__":
    main()