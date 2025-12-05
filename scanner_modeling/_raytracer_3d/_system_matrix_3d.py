import torch
from torch import Tensor
from typing import Dict, Tuple, List, Optional
from ..geometry_3d import voxels_coordinates_3d, object_to_world_aabb
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
    # --- NEW OPTIMIZATION ARGS ---
    aabb_min: Optional[Tensor] = None,
    aabb_max: Optional[Tensor] = None,
):
    """
    Phase-5 blockwise assembly of A[i, j] for a single detector i.
    Integrates Broad-Phase culling for performance.
    
    Optimization: Pass aabb_min/max if they are already computed to avoid 
    re-calculating them for every detector.
    """

    # -- (1) Build FOV voxel centers
    fov_centers = voxels_coordinates_3d(fov_dict).to(device=device, dtype=DTYPE)
    N_fov = fov_centers.shape[0]
    N_q = det_voxel_centers.shape[0]

    # -- (2) Precompute AABBs for objects (If not provided)
    if aabb_min is None or aabb_max is None:
        aabb_min, aabb_max = object_to_world_aabb(objects)
    
    # Ensure they are on the right device
    aabb_min = aabb_min.to(device=device)
    aabb_max = aabb_max.to(device=device)

    triples = []

    # -- (3) Loop over FOV blocks to stay memory-safe
    block_size = max_rays_per_chunk
    for start in range(0, N_fov, block_size):
        end = min(start + block_size, N_fov)

        fov_block = fov_centers[start:end]

        # Build block rays
        # rays_3d_batch gives shape (B, Q, 2, 3)
        rays = rays_3d_batch(fov_block, det_voxel_centers)
        o = rays[..., 0, :].reshape(-1, 3)  # (B*Q, 3)
        e = rays[..., 1, :].reshape(-1, 3)
        d = e - o

        # Broad-phase: candidate object lists
        # Returns: obj_idx (flat), ray_offsets (CSR style)
        obj_idx, ray_offsets, _ = build_ray_object_candidate_lists(
            o, d, aabb_min, aabb_max, max_rays_per_chunk=max_rays_per_chunk
        )

        # --- CONVERT CSR TO COORDINATE FORMAT (PAIRS) ---
        # We need explicit ray indices to pass to the sparse narrow phase
        counts_per_ray = torch.diff(ray_offsets)
        
        # Expand: [0, 0, 1, 2, 2, 2...] corresponding to obj_idx
        ray_indices_expanded = torch.repeat_interleave(
            torch.arange(counts_per_ray.shape[0], device=device),
            counts_per_ray
        )
        
        # Narrow-phase PPDF for this block
        from ..geometry_3d import solid_angle_detector_voxel
        solid_angle = solid_angle_detector_voxel(fov_block, voxel_faces)

        f_block = ppdf_3d_local(
            fov_block,
            det_voxel_centers,
            objects,
            mu_objects,
            torch.tensor(mu_detector, dtype=DTYPE, device=device),
            solid_angle,
            det_index,
            # Pass the sparse candidate list
            sparse_ray_indices=ray_indices_expanded,
            sparse_obj_indices=obj_idx
        )

        # --- Write as sparse triplets ---
        for q in range(N_q):
            row = det_index * N_q + q # measurement index

            for bi in range(f_block.shape[1]):
                col = start + bi  # global voxel index
                val = f_block[q, bi].item()
                if abs(val) > eps: # Simple thresholding
                    triples.append((row, col, val))

    return triples