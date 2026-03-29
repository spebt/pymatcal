import torch
from torch import Tensor
from .._config import DTYPE
from typing import Tuple

# Broad-phase uses the global DTYPE (float32) for the slab test.
# When DTYPE was float64 this was overridden to float32 since the
# output is boolean. Now that DTYPE is float32, this matches.
_BROAD_DTYPE = DTYPE


def ray_aabb_intersect(
    o: Tensor,          # (..., 3) ray origins in world coordinates
    d: Tensor,          # (..., 3) ray directions in world coordinates
    aabb_min: Tensor,   # (N_obj, 3) per-object mins in world coordinates
    aabb_max: Tensor,   # (N_obj, 3) per-object maxs in world coordinates
) -> Tensor:
    """
    Broad-phase ray–AABB intersection test using the axis-aligned slab method.

    Parameters
    ----------
    o : Tensor
        Ray origins in world coordinates. Shape:
            - (N_rays, 3)            or
            - (N_vox, N_det_vox, 3)  or, more generally, (*R, 3)

    d : Tensor
        Ray directions in world coordinates, same shape as `o`.

    aabb_min, aabb_max : Tensor
        World-space AABB bounds for each object, shape (N_obj, 3).

    Returns
    -------
    hit_mask : Tensor
        Boolean tensor of shape (*R, N_obj), where hit_mask[... , k] is True
        if the corresponding ray intersects the AABB of object k
        (for t >= 0 along d).

    Notes
    -----
    - This is *axis-aligned* (AABB) and works in world coordinates.
    - It is intentionally cheap and vectorized, for broad-phase culling
      before narrow-phase OBB / convex-poly intersection.
    - Uses float32 internally since the output is boolean — no precision
      benefit from float64, and float32 halves memory / improves throughput.
    - Internally uses the standard slab method:

          inv_d = 1 / d_safe
          t1 = (aabb_min - o) * inv_d
          t2 = (aabb_max - o) * inv_d
          t_min = min(t1, t2)
          t_max = max(t1, t2)
          t_enter = max(t_min_x, t_min_y, t_min_z)
          t_exit  = min(t_max_x, t_max_y, t_max_z)

      and reports a hit when t_exit >= max(t_enter, 0).
    """
    if o.shape != d.shape or o.shape[-1] != 3:
        raise ValueError(
            f"o and d must have the same shape (*, 3); "
            f"got {o.shape} and {d.shape}"
        )

    if aabb_min.ndim != 2 or aabb_min.shape[1] != 3:
        raise ValueError(
            f"aabb_min must have shape (N_obj, 3); got {aabb_min.shape}"
        )
    if aabb_max.ndim != 2 or aabb_max.shape[1] != 3:
        raise ValueError(
            f"aabb_max must have shape (N_obj, 3); got {aabb_max.shape}"
        )

    device = o.device
    bp = _BROAD_DTYPE

    # Cast to float32 for broad phase (boolean output — no precision loss)
    o = o.to(device=device, dtype=bp)
    d = d.to(device=device, dtype=bp)
    aabb_min = aabb_min.to(device=device, dtype=bp)
    aabb_max = aabb_max.to(device=device, dtype=bp)

    # Avoid division-by-zero for nearly-parallel rays
    eps = torch.finfo(bp).eps
    d_safe = torch.where(d.abs() < eps, torch.full_like(d, eps), d)
    inv_d = 1.0 / d_safe  # (*R, 3)

    # Expand for broadcasting over objects:
    #   o[..., None, :]       : (*R, 1, 3)
    #   inv_d[..., None, :]   : (*R, 1, 3)
    #   aabb_min[None, ...,:] : (1, N_obj, 3)
    # Result shapes: (*R, N_obj, 3)
    o_exp = o.unsqueeze(-2)
    inv_d_exp = inv_d.unsqueeze(-2)
    aabb_min_exp = aabb_min.view(1, -1, 3)
    aabb_max_exp = aabb_max.view(1, -1, 3)

    t1 = (aabb_min_exp - o_exp) * inv_d_exp
    t2 = (aabb_max_exp - o_exp) * inv_d_exp

    t_min = torch.minimum(t1, t2)  # per-axis entry: (*R, N_obj, 3)
    t_max = torch.maximum(t1, t2)  # per-axis exit:  (*R, N_obj, 3)

    # Ray enters after the latest per-axis entry, exits before earliest exit
    t_enter = t_min.max(dim=-1).values   # (*R, N_obj)
    t_exit = t_max.min(dim=-1).values    # (*R, N_obj)

    zero = torch.zeros((), dtype=bp, device=device)
    hit_mask = t_exit >= torch.maximum(t_enter, zero)

    return hit_mask

def build_ray_object_candidate_lists(
    o: Tensor,
    d: Tensor,
    aabb_min: Tensor,
    aabb_max: Tensor,
    max_rays_per_chunk: int = 10_000,
) -> Tuple[Tensor, Tensor, Tuple[int, ...]]:
    """
    Build per-ray candidate object lists using the ray–AABB broad phase.

    Runs a single vectorized slab test on all rays at once (no internal
    chunking — the caller is responsible for chunking if needed).

    Parameters
    ----------
    o : Tensor
        Ray origins in world coordinates, shape (*R, 3), e.g.:
          - (N_rays, 3), or
          - (N_vox, N_det_vox, 3).

    d : Tensor
        Ray directions in world coordinates, same shape as `o`.

    aabb_min, aabb_max : Tensor
        World-space AABB bounds for each object, shape (N_obj, 3).

    max_rays_per_chunk : int, optional
        Kept for API compatibility but no longer used internally.
        The caller (compute_system_matrix_for_detector) already chunks
        FOV blocks, so internal re-chunking is redundant.

    Returns
    -------
    obj_indices : Tensor
        1D tensor of length N_pairs (int64) listing the object index
        for each (ray, object) pair that passed the AABB test.

    ray_offsets : Tensor
        1D tensor (int64) of length N_rays_flat + 1, where
            ray_offsets[r]   is the start index in `obj_indices`
            ray_offsets[r+1] is the end index (exclusive)
        for the r-th flattened ray (0 ≤ r < N_rays_flat).

    ray_shape : tuple of int
        The original ray batch shape (*R,), so you can map flattened
        ray indices back to multi-dimensional indices if needed.

    Notes
    -----
    - The flattened ray index r ∈ [0, N_rays_flat) corresponds to
      the usual row-major indexing of o.view(-1, 3).
    - If there are no rays or no objects, the function returns:
        obj_indices = empty (0,)
        ray_offsets = zeros(N_rays_flat + 1)
    """
    if o.shape != d.shape or o.shape[-1] != 3:
        raise ValueError(
            f"o and d must have the same shape (*, 3); got {o.shape} and {d.shape}"
        )

    if aabb_min.ndim != 2 or aabb_min.shape[1] != 3:
        raise ValueError(
            f"aabb_min must have shape (N_obj, 3); got {aabb_min.shape}"
        )
    if aabb_max.ndim != 2 or aabb_max.shape[1] != 3:
        raise ValueError(
            f"aabb_max must have shape (N_obj, 3); got {aabb_max.shape}"
        )

    device = o.device
    ray_shape = o.shape[:-1]
    # Number of rays = product of all batch dims
    n_rays = 1
    for dim in ray_shape:
        n_rays *= int(dim)

    n_objs = int(aabb_min.shape[0])

    # Handle trivial cases early
    if n_rays == 0 or n_objs == 0:
        obj_indices = torch.empty((0,), dtype=torch.int64, device=device)
        ray_offsets = torch.zeros((n_rays + 1,), dtype=torch.int64, device=device)
        return obj_indices, ray_offsets, ray_shape

    # Flatten rays to (N_rays, 3)
    o_flat = o.reshape(n_rays, 3)
    d_flat = d.reshape(n_rays, 3)

    # Single vectorized broad-phase test (no internal chunking)
    hit_mask = ray_aabb_intersect(o_flat, d_flat, aabb_min, aabb_max)
    # hit_mask: (N_rays, N_obj)

    # Extract (ray_idx, obj_idx) pairs from boolean mask
    # nonzero returns row-major order, so pairs are already sorted by ray index
    hits = hit_mask.nonzero(as_tuple=False)  # (N_pairs, 2)

    if hits.shape[0] == 0:
        obj_indices = torch.empty((0,), dtype=torch.int64, device=device)
        ray_offsets = torch.zeros((n_rays + 1,), dtype=torch.int64, device=device)
        return obj_indices, ray_offsets, ray_shape

    ray_indices_flat = hits[:, 0]  # already sorted (row-major from nonzero)
    obj_indices = hits[:, 1]

    # Count hits per ray, then build prefix sums to get offsets
    ray_counts = torch.bincount(
        ray_indices_flat,
        minlength=n_rays,
    )  # (N_rays,)

    ray_offsets = torch.empty(
        (n_rays + 1,),
        dtype=torch.int64,
        device=device,
    )
    ray_offsets[0] = 0
    ray_offsets[1:] = ray_counts.cumsum(0)

    return obj_indices, ray_offsets, ray_shape
