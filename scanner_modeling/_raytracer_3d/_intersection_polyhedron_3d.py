from __future__ import annotations

# scanner_modeling/_raytracer_3d/_intersection_polyhedron_3d.py
import torch
from torch import Tensor
from .._config import DTYPE


def ray_convex_polyhedron_intersection_local(
    o_local: Tensor,        # (N_rays, N_poly, 3) in local coords
    d_local: Tensor,        # (N_rays, N_poly, 3)
    plane_normals: Tensor,  # (N_poly, P_max, 3)
    plane_offsets: Tensor,  # (N_poly, P_max)
    plane_mask: Tensor | None = None,  # (N_poly, P_max) bool, True = active plane
):
    """
    Exact ray–convex-polyhedron intersection in *local* coordinates.

    Planes are of the form: n·x <= d
    Ray is: r(t) = o + t d,  t >= 0

    Shapes
    ------
    o_local      : (N_rays, N_poly, 3)
    d_local      : (N_rays, N_poly, 3)
    plane_normals: (N_poly, P_max, 3)
    plane_offsets: (N_poly, P_max)
    plane_mask   : (N_poly, P_max) boolean, True for valid planes.
                   If None, all planes are treated as active.

    Returns
    -------
    t_enter : (N_rays, N_poly)
        Maximum lower bound on t across all active planes.
    t_exit  : (N_rays, N_poly)
        Minimum upper bound on t across all active planes.
    hit_mask : (N_rays, N_poly) bool
        True where the ray intersects the convex polyhedron with t_exit >= max(t_enter, 0).

    Notes
    -----
    - This function returns *parametric* t values. The physical path length
      inside the polyhedron is:

          L = clamp(t_exit - t_enter, min=0) * ||d_local||,

      where ||d_local|| is the Euclidean norm along the last dimension.
    """
    # --- Basic checks / casting ---
    if o_local.shape != d_local.shape or o_local.shape[-1] != 3:
        raise ValueError(
            f"o_local and d_local must have shape (N_rays, N_poly, 3); "
            f"got {o_local.shape} and {d_local.shape}"
        )

    if plane_normals.ndim != 3 or plane_normals.shape[-1] != 3:
        raise ValueError(
            f"plane_normals must have shape (N_poly, P_max, 3); got {plane_normals.shape}"
        )
    if plane_offsets.ndim != 2:
        raise ValueError(
            f"plane_offsets must have shape (N_poly, P_max); got {plane_offsets.shape}"
        )

    device = o_local.device

    o_local = o_local.to(device=device, dtype=DTYPE)
    d_local = d_local.to(device=device, dtype=DTYPE)
    plane_normals = plane_normals.to(device=device, dtype=DTYPE)
    plane_offsets = plane_offsets.to(device=device, dtype=DTYPE)

    N_rays, N_poly, _ = o_local.shape
    N_poly_planes, P_max = plane_offsets.shape
    if N_poly_planes != N_poly:
        raise ValueError(
            f"plane_normals/plane_offsets N_poly={N_poly_planes} "
            f"must match o_local N_poly={N_poly}"
        )

    # --- Broadcast per-plane data over rays ---
    # o: (N_rays, N_poly, 1, 3)
    # d: (N_rays, N_poly, 1, 3)
    o = o_local.unsqueeze(-2)
    d = d_local.unsqueeze(-2)

    # plane_normals: (1, N_poly, P_max, 3)
    n = plane_normals.unsqueeze(0)
    # plane_offsets: (1, N_poly, P_max)
    d_plane = plane_offsets.unsqueeze(0)

    # plane_mask: (1, N_poly, P_max)
    if plane_mask is None:
        active = torch.ones_like(plane_offsets, dtype=torch.bool, device=device)
    else:
        if plane_mask.shape != plane_offsets.shape:
            raise ValueError(
                f"plane_mask must have shape (N_poly, P_max); got {plane_mask.shape}"
            )
        active = plane_mask.to(device=device, dtype=torch.bool)
    active = active.unsqueeze(0)  # (1, N_poly, P_max)

    # --- Dot products: nd = n·d, no = n·o ---
    # nd, no, d_plane all broadcast to (N_rays, N_poly, P_max)
    nd = (n * d).sum(dim=-1)        # (N_rays, N_poly, P_max)
    no = (n * o).sum(dim=-1)        # (N_rays, N_poly, P_max)
    d_plane = d_plane.expand_as(nd) # (N_rays, N_poly, P_max)
    active = active.expand_as(nd)   # (N_rays, N_poly, P_max)

    eps = torch.finfo(DTYPE).eps

    # --- Parallel planes: |n·d| < eps ---
    parallel = active & (nd.abs() < eps)

    # If parallel and n·o > d, the ray never satisfies this plane => misses poly
    violates_parallel = parallel & (no > d_plane)
    violates_any = violates_parallel.any(dim=-1)  # (N_rays, N_poly)

    # (If parallel & n·o <= d, plane is "always satisfied" for that ray;
    # we simply don't impose any t bound from that plane.)

    # --- Non-parallel planes ---
    nonparallel = active & ~parallel

    # t_plane = (d_plane - no) / (n·d) for non-parallel planes
    numer = d_plane - no
    safe_nd = torch.where(nonparallel, nd, torch.ones_like(nd))
    t_plane = numer / safe_nd

    # We now classify planes as entry (lower bound on t) or exit (upper bound on t)
    # For n·d > 0:     t <= t_plane (exit plane)
    # For n·d < 0:     t >= t_plane (entry plane)
    exit_mask = nonparallel & (nd >  eps)
    entry_mask = nonparallel & (nd < -eps)

    # --- Aggregate t_enter (max lower bound) and t_exit (min upper bound) ---
    # Initialize with -inf / +inf per (ray, poly)
    t_enter = torch.full(
        nd.shape[:-1], -torch.inf, dtype=DTYPE, device=device
    )  # (N_rays, N_poly)
    t_exit = torch.full(
        nd.shape[:-1], torch.inf, dtype=DTYPE, device=device
    )

    if entry_mask.any():
        # For entries, ignore everything except entry planes
        t_enter_candidates = torch.where(
            entry_mask, t_plane, torch.full_like(t_plane, -torch.inf)
        )
        t_enter = torch.maximum(
            t_enter,
            t_enter_candidates.max(dim=-1).values,  # max over planes
        )

    if exit_mask.any():
        t_exit_candidates = torch.where(
            exit_mask, t_plane, torch.full_like(t_plane, torch.inf)
        )
        t_exit = torch.minimum(
            t_exit,
            t_exit_candidates.min(dim=-1).values,  # min over planes
        )

    # --- Final hit mask ---
    zero = torch.zeros((), dtype=DTYPE, device=device)
    hit_mask = t_exit >= torch.maximum(t_enter, zero)

    # Enforce the parallel-violation rule: any violating plane => no hit
    hit_mask = hit_mask & (~violates_any)

    return t_enter, t_exit, hit_mask
