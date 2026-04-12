#!/usr/bin/env python
# arg_ppdf_calculation_3d_batch.py

import argparse
import os
import time
import h5py
import numpy as np
import torch

from scanner_modeling._config import DTYPE

# 3D FOV + 3D geometry helpers
from scanner_modeling.geometry_3d import (
    fov_tensor_dict_3d,
    load_scanner_geometry_3d_from_layout,
    build_convex_union_convex_poly_block,
)

# 2D layout loader
from scanner_modeling.geometry_2d import load_scanner_layouts

# Phase 5 system-matrix builder
from scanner_modeling.raytracer_3d import compute_system_matrix_for_detector

# Hexahedra helper to get quads (faces)
from helper_3d import hexahedron_quads


def hexes_to_obb_block(hexes: torch.Tensor, material_index_val: int):
    """
    Convert hexahedra (N, 8, 3) into an OBB block, with orthonormalized axes.
    Returns only geometric OBB parts; metadata (detector_index, etc.) added outside.
    """
    device = hexes.device
    dtype = DTYPE

    N = hexes.shape[0]
    if N == 0:
        return {
            "centers": torch.empty((0, 3), dtype=dtype, device=device),
            "rotations": torch.empty((0, 3, 3), dtype=dtype, device=device),
            "half_sizes": torch.empty((0, 3), dtype=dtype, device=device),
            "material_index": torch.empty((0,), dtype=torch.int64, device=device),
        }

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
    u_ax  = e_ax  / torch.clamp(torch.linalg.norm(e_ax,  dim=1, keepdim=True), min=1e-9)
    u_tan = torch.cross(u_ax, u_rad, dim=1)
    u_tan = u_tan / torch.clamp(torch.linalg.norm(u_tan, dim=1, keepdim=True), min=1e-9)

    rotations = torch.stack([u_rad, u_tan, u_ax], dim=-1)  # (N, 3, 3)

    # Projected half-sizes
    half_size_x = (e_rad * u_rad).sum(dim=1).abs() * 0.5
    half_size_y = (e_tan * u_tan).sum(dim=1).abs() * 0.5
    half_size_z = (e_ax  * u_ax ).sum(dim=1).abs() * 0.5

    half_sizes = torch.stack([half_size_x, half_size_y, half_size_z], dim=1)

    material_index = torch.full(
        (N,), int(material_index_val), dtype=torch.int64, device=device
    )

    return {
        "centers": centers,
        "rotations": rotations,
        "half_sizes": half_sizes,
        "material_index": material_index,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute PPDF system matrix for a range of detectors (sequential).")
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--start", type=int, required=True, help="First detector index (inclusive)")
    parser.add_argument("--count", type=int, required=True, help="Number of detectors to process")
    parser.add_argument("--out_shards", type=str, default="output_shards")

    # FOV settings (same as your batch script CLI)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--sx", type=float, default=64.0)
    parser.add_argument("--sy", type=float, default=64.0)
    parser.add_argument("--sz", type=float, default=2.0)

    parser.add_argument("--tiles", type=str, default="4,4,4")

    parser.add_argument("--layout_file", type=str, required=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Max number of rays per chunk for compute_system_matrix_for_detector",
    )

    args = parser.parse_args()

    # -----------------------------
    # 1) Load layout and geometry
    # -----------------------------
    layout_dir = os.path.dirname(args.layout_file)
    layout_filename = os.path.basename(args.layout_file)
    scanner_layouts, _ = load_scanner_layouts(layout_dir, layout_filename)

    # Also load raw tensor for top-level metadata (mu values, hole prisms).
    raw_layout = torch.load(args.layout_file, weights_only=False)
    applied_cfg = raw_layout.get("applied_config", {})
    hole_prisms  = raw_layout.get("collimator hole prisms", None)

    # Per-material attenuation (mm^-1). Fall back to MPH defaults if absent.
    mu_plate = float(applied_cfg.get("mu_plate_mm_inv",    3.5))
    mu_det   = float(applied_cfg.get("mu_detector_mm_inv", 0.475))

    key = f"position {args.layout:03d}"
    layout_entry = scanner_layouts[key]

    det_hex = layout_entry["detector units 3d"].to(DTYPE)   # (N_det, 8, 3)
    plate_hex = layout_entry["plate segments 3d"].to(DTYPE) # (N_plate, 8, 3)

    N_det_total = det_hex.shape[0]
    device = det_hex.device
    print(f"Total detectors in ring: {N_det_total}")

    # Range of detectors for this shard
    start_det = args.start
    end_det = min(args.start + args.count, N_det_total)
    if start_det >= N_det_total:
        raise ValueError(f"start={start_det} is >= total detectors {N_det_total}")

    print(f"Processing detectors from {start_det} to {end_det - 1} (inclusive)")

    # -----------------------------
    # 2) Build OBB blocks (once)
    # -----------------------------
    det_block = hexes_to_obb_block(det_hex, material_index_val=1)
    det_block["detector_index"] = torch.arange(N_det_total, dtype=torch.int64, device=device)
    det_block["detector_voxel_index"] = torch.zeros(N_det_total, dtype=torch.int64, device=device)

    plate_block = hexes_to_obb_block(plate_hex, material_index_val=0)
    N_plate = plate_block["centers"].shape[0]
    plate_block["detector_index"] = torch.full((N_plate,), -1, dtype=torch.int64, device=device)
    plate_block["detector_voxel_index"] = torch.full((N_plate,), -1, dtype=torch.int64, device=device)

    def concat_blocks(a, b):
        return torch.cat([a, b], dim=0)

    obb_layout = {
        "centers": concat_blocks(det_block["centers"],         plate_block["centers"]),
        "rotations": concat_blocks(det_block["rotations"],     plate_block["rotations"]),
        "half_sizes": concat_blocks(det_block["half_sizes"],   plate_block["half_sizes"]),
        "material_index": concat_blocks(det_block["material_index"], plate_block["material_index"]),
        "detector_index": concat_blocks(det_block["detector_index"], plate_block["detector_index"]),
        "detector_voxel_index": concat_blocks(
            det_block["detector_voxel_index"], plate_block["detector_voxel_index"]
        ),
    }

    # Option A hole prisms: convex poly objects with mu = -mu_plate.
    MATERIAL_PLATE     = 0
    MATERIAL_DETECTOR  = 1
    MATERIAL_HOLE_VOID = 2

    holes_poly_block = None
    if hole_prisms is not None:
        M_holes = hole_prisms["centers"].shape[0]
        print(f"Loading {M_holes} hole prisms (negative-mu subtractive geometry)...")
        holes_poly_block = build_convex_union_convex_poly_block(
            centers              = hole_prisms["centers"].to(device=device),
            rotations            = hole_prisms["rotations"].to(device=device),
            plane_normals_local  = hole_prisms["plane_normals_local"].to(device=device),
            plane_offsets_local  = hole_prisms["plane_offsets_local"].to(device=device),
            material_index       = MATERIAL_HOLE_VOID,
            detector_index       = -1,
            detector_voxel_index = -1,
            num_planes           = hole_prisms["num_planes"].to(device=device),
        )
        if "poly_local_aabb_min" in hole_prisms:
            holes_poly_block["poly_local_aabb_min"] = hole_prisms["poly_local_aabb_min"].to(device=device)
            holes_poly_block["poly_local_aabb_max"] = hole_prisms["poly_local_aabb_max"].to(device=device)

    entry = {"obb": obb_layout}
    if holes_poly_block is not None:
        entry["convex_poly"] = holes_poly_block

    scanner_layouts_for_objects = {key: entry}
    objects = load_scanner_geometry_3d_from_layout(args.layout, scanner_layouts_for_objects)

    # -----------------------------
    # 3) Precompute voxel faces for all detectors (once)
    #    Include all faces EXCEPT the back face (farthest from FOV).
    # -----------------------------
    quads = hexahedron_quads(det_hex)  # (N_det_total, 6, 4, 3)

    face_centers = quads.mean(dim=2)               # (N_det_total, 6, 3)
    # Full 3-D distance from the FOV centre (origin) — correct for ring scanners
    # (outer radial face) and flat panels (+Y face) alike.  The previous [:2]
    # projection worked by coincidence for the SCT panel but fails in general when
    # detector X or Z offsets approach the Y depth.
    dist3d = torch.linalg.norm(face_centers, dim=-1)          # (N_det_total, 6)
    back_face_idx = dist3d.argmax(dim=1)           # (N_det_total,) — farthest from FOV

    # Build all_voxel_faces: (N_det_total, 5, 2, 3, 3) — 5 faces, 2 triangles each
    all_voxel_faces = torch.empty(
        (N_det_total, 5, 2, 3, 3), dtype=DTYPE, device=device
    )
    for det_i in range(N_det_total):
        back_idx = int(back_face_idx[det_i].item())
        keep = [f for f in range(6) if f != back_idx]  # 5 face indices
        for slot, face_idx in enumerate(keep):
            q = quads[det_i, face_idx]  # (4, 3)
            all_voxel_faces[det_i, slot, 0] = torch.stack([q[0], q[1], q[2]], dim=0)
            all_voxel_faces[det_i, slot, 1] = torch.stack([q[0], q[2], q[3]], dim=0)

    # -----------------------------
    # 4) FOV + material mapping
    # -----------------------------
    tiles_tuple = tuple(map(int, args.tiles.split(",")))

    fov_dict = fov_tensor_dict_3d(
        n_voxels=(args.nx, args.ny, args.nz),
        size_in_mm=(args.sx, args.sy, args.sz),
        center_coordinates=(0.0, 0.0, 0.0),
        n_subdivisions=tiles_tuple,
    )
    Nx, Ny, Nz = [int(v.item()) for v in fov_dict["n voxels"]]
    total_voxels = Nx * Ny * Nz

    # Map materials: index 0=plate, 1=detector, 2=hole void (negative, subtractive).
    if hole_prisms is not None:
        mu_table = torch.tensor([mu_plate, mu_det, -mu_plate], dtype=DTYPE, device=device)
    else:
        mu_table = torch.tensor([mu_plate, mu_det], dtype=DTYPE, device=device)
    mu_objects_expanded = mu_table[objects["material_index"]]
    mu_detector = torch.tensor(mu_det, dtype=DTYPE, device=device)

    # -----------------------------
    # 5) Loop over detectors (sequential)
    # -----------------------------
    results = []
    t0 = time.time()

    for det_idx in range(start_det, end_det):
        print(f"\n=== Detector {det_idx} ===")
        t_det0 = time.time()

        target_center = det_block["centers"][det_idx : det_idx + 1]  # (1, 3)
        target_faces = all_voxel_faces[det_idx : det_idx + 1]        # (1, 1, 2, 3, 3)

        triples = compute_system_matrix_for_detector(
            det_index=det_idx,
            det_voxel_centers=target_center,
            voxel_faces=target_faces,
            fov_dict=fov_dict,
            objects=objects,
            mu_objects=mu_objects_expanded,
            mu_detector=float(mu_detector.item()),
            max_rays_per_chunk=args.chunk_size,
            device=device,
        )

        # Dense row for this detector (vectorized scatter)
        dense_row = torch.zeros(total_voxels, dtype=torch.float32, device=device)
        if triples:
            cols_t = torch.tensor([c for _, c, _ in triples], dtype=torch.long, device=device)
            vals_t = torch.tensor([v for _, _, v in triples], dtype=torch.float32, device=device)
            dense_row.scatter_(0, cols_t, vals_t)

        results.append((det_idx, dense_row.cpu().numpy()))

        t_det1 = time.time()
        print(f"Detector {det_idx}: {t_det1 - t_det0:.2f} s")

    t1 = time.time()
    print(f"\nFinished detectors {start_det}..{end_det-1} in {t1 - t0:.2f} s total")

    # Sort by detector index just in case
    results.sort(key=lambda x: x[0])

    # -----------------------------
    # 6) Save shard to HDF5
    # -----------------------------
    os.makedirs(args.out_shards, exist_ok=True)
    shard_name = f"ppdf_shard_{start_det:05d}_{end_det-1:05d}.h5"
    out_path = os.path.join(args.out_shards, shard_name)

    print(f"Writing shard to: {out_path}")

    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset(
            "ppdfs",
            shape=(len(results), total_voxels),
            dtype="f4",
            compression="gzip",
            chunks=(1, total_voxels),
        )

        dset.attrs["layout_idx"] = args.layout
        dset.attrs["start_idx"] = start_det
        dset.attrs["end_idx"] = end_det - 1
        dset.attrs["Nx"] = Nx
        dset.attrs["Ny"] = Ny
        dset.attrs["Nz"] = Nz
        dset.attrs["size_mm"] = np.array(
            [args.sx, args.sy, args.sz], dtype=np.float32
        )

        for i, (det_idx, dense_row) in enumerate(results):
            dset[i] = dense_row

    print("Done.")


if __name__ == "__main__":
    main()
