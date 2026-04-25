import torch
from torch import Tensor
from typing import Dict, Tuple, List, Optional
from ..geometry_3d import voxels_coordinates_3d, object_to_world_aabb, solid_angle_detector_voxel
from ._local_functions_3d import rays_3d_batch
from ._intersection_aabb_3d import build_ray_object_candidate_lists
from ._ppdf_3d import ppdf_3d_local

from .._config import DTYPE


def compute_system_matrix_for_detector(
    det_index: int,
    det_voxel_centers: Tensor,
    voxel_faces: Tensor,
    fov_dict: Dict,
    objects: Dict[str, Tensor],
    mu_objects: Tensor,
    mu_detector: float,
    max_rays_per_chunk: int = 10_000,
    device="cpu",
    eps: float = 1e-9,
    # --- OPTIMIZATION ARGS ---
    aabb_min: Optional[Tensor] = None,
    aabb_max: Optional[Tensor] = None,
):
    """
    Phase-5 blockwise assembly of A[i, j] for a single detector i.
    Integrates Broad-Phase culling for performance.

    Optimization: Pass aabb_min/max if they are already computed to avoid
    re-calculating them for every detector.
    """

    # -- (0) Safety guard — batched multi-detector evaluation OOMs (Issue #5)
    assert det_voxel_centers.shape[0] == 1, (
        "det_voxel_centers must contain exactly 1 center. "
        "Passing the full detector ring (N_q > 1) materialises a dense "
        "(N_fov × N_q, 2, 3) ray tensor that will OOM. "
        "Call compute_system_matrix_for_detector once per detector."
    )

    # -- (1) Build FOV voxel centers
    fov_centers = voxels_coordinates_3d(fov_dict).to(device=device, dtype=DTYPE)
    N_fov = fov_centers.shape[0]
    N_q = det_voxel_centers.shape[0]

    # -- (2) Precompute AABBs for objects (if not provided)
    if aabb_min is None or aabb_max is None:
        aabb_min, aabb_max = object_to_world_aabb(objects)

    aabb_min = aabb_min.to(device=device)
    aabb_max = aabb_max.to(device=device)

    # -- (3) Move objects dict and mu_objects to device ONCE before the loop.
    #        Doing this inside ppdf_3d_local (called per chunk) causes a
    #        CPU→GPU PCIe transfer every chunk iteration (Issue #3).
    objects = {k: v.to(device=device) if isinstance(v, Tensor) else v
               for k, v in objects.items()}
    mu_objects = mu_objects.to(device=device, dtype=DTYPE)

    mu_detector_t = torch.tensor(mu_detector, dtype=DTYPE, device=device)

    # Collect sparse triplet tensors across all chunks
    all_rows = []
    all_cols = []
    all_vals = []

    # -- (3) Loop over FOV blocks to stay memory-safe
    block_size = max_rays_per_chunk
    for start in range(0, N_fov, block_size):
        end = min(start + block_size, N_fov)

        fov_block = fov_centers[start:end]

        # Build block rays once (shared by broad phase and PPDF)
        rays = rays_3d_batch(fov_block, det_voxel_centers)
        o = rays[..., 0, :].reshape(-1, 3)  # (B*Q, 3)
        e = rays[..., 1, :].reshape(-1, 3)
        d = e - o

        # Broad-phase: returns flat (ray_idx, obj_idx) pairs directly (Issue #2)
        obj_idx, ray_idx, _ = build_ray_object_candidate_lists(
            o, d, aabb_min, aabb_max, max_rays_per_chunk=max_rays_per_chunk
        )

        # Solid angle for this block (all 5 faces; back face excluded at build time)
        solid_angle = solid_angle_detector_voxel(fov_block, voxel_faces, faces="all")

        # Narrow-phase PPDF — pre-computed rays passed to avoid recomputation
        f_block = ppdf_3d_local(
            fov_block,
            det_voxel_centers,
            objects,
            mu_objects,
            mu_detector_t,
            solid_angle,
            det_index,
            sparse_ray_indices=ray_idx,
            sparse_obj_indices=obj_idx,
            o_world=o,
            d_world=d,
        )

        # --- Vectorized sparse triplet extraction ---
        # f_block shape: (N_q, block_size)
        nz_mask = f_block.abs() > eps
        nz_indices = nz_mask.nonzero(as_tuple=False)  # (N_nz, 2): [q_idx, bi_idx]

        if nz_indices.shape[0] > 0:
            q_idx = nz_indices[:, 0]
            bi_idx = nz_indices[:, 1]

            rows = det_index * N_q + q_idx
            cols = start + bi_idx
            vals = f_block[q_idx, bi_idx]

            all_rows.append(rows)
            all_cols.append(cols)
            all_vals.append(vals)

    # Assemble final triplet list
    if all_rows:
        rows_t = torch.cat(all_rows).cpu().tolist()
        cols_t = torch.cat(all_cols).cpu().tolist()
        vals_t = torch.cat(all_vals).cpu().tolist()
        triples = list(zip(rows_t, cols_t, vals_t))
    else:
        triples = []

    return triples