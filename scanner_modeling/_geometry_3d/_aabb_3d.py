# pymatcal/scanner_modeling/_geometry_3d/_aabb_3d.py

from typing import Dict, Tuple

import torch
from torch import Tensor

from .._config import DTYPE


def object_to_world_aabb(objects: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Compute per-object world-space AABBs for a unified 3D object dictionary.

    Parameters
    ----------
    objects : Dict[str, Tensor]
        Unified object dictionary from `load_scanner_geometry_3d_from_layout`,
        with the usual keys:
          - "centers"        : (N_obj, 3)
          - "rotations"      : (N_obj, 3, 3)
          - "obb_object_ids" : (N_obb,)
          - "half_sizes"     : (N_obb, 3)
          - "poly_object_ids": (N_poly,)
          - (optional) "poly_local_aabb_min": (N_poly, 3)
          - (optional) "poly_local_aabb_max": (N_poly, 3)

        NOTE: For convex polyhedra, we support two regimes:

        1. If "poly_local_aabb_min/max" are present, we compute a *tight*
           world-space AABB by transforming the 8 corners of the local AABB.
        2. If they are absent, we fall back to an "infinite" AABB for those
           objects (no broad-phase culling for them, but correctness is preserved).

    Returns
    -------
    aabb_min, aabb_max : (N_obj, 3), (N_obj, 3)
        World-space AABB per object k:
            aabb_min[k] = (xmin, ymin, zmin)
            aabb_max[k] = (xmax, ymax, zmax)

        Distances are in the global DISTANCE_UNIT (mm).
    """
    centers = objects["centers"].to(dtype=DTYPE)
    rotations = objects["rotations"].to(dtype=DTYPE)

    N_obj = int(centers.shape[0])
    device = centers.device

    inf = torch.tensor(torch.inf, dtype=DTYPE, device=device)
    aabb_min = torch.full((N_obj, 3), inf, dtype=DTYPE, device=device)
    aabb_max = torch.full((N_obj, 3), -inf, dtype=DTYPE, device=device)

    # ---------- 1) OBBs: |R|·half_sizes trick ----------
    obb_ids = objects.get("obb_object_ids", None)
    if obb_ids is not None and obb_ids.numel() > 0:
        half_sizes = objects["half_sizes"].to(dtype=DTYPE, device=device)  # (N_obb, 3)

        R_obb = rotations[obb_ids]        # (N_obb, 3, 3)
        c_obb = centers[obb_ids]          # (N_obb, 3)

        # extents = |R| · half_sizes, per object
        R_abs = R_obb.abs()               # (N_obb, 3, 3)
        extents = torch.einsum("bij,bj->bi", R_abs, half_sizes)  # (N_obb, 3)

        aabb_min[obb_ids] = c_obb - extents
        aabb_max[obb_ids] = c_obb + extents

    # ---------- 2) Convex polyhedra ----------
    poly_ids = objects.get("poly_object_ids", None)
    if poly_ids is not None and poly_ids.numel() > 0:
        has_local_aabb = (
            "poly_local_aabb_min" in objects
            and "poly_local_aabb_max" in objects
        )

        if has_local_aabb:
            local_min = objects["poly_local_aabb_min"].to(dtype=DTYPE, device=device)
            local_max = objects["poly_local_aabb_max"].to(dtype=DTYPE, device=device)
            if local_min.shape != local_max.shape:
                raise ValueError(
                    f"poly_local_aabb_min/max must have the same shape; "
                    f"got {local_min.shape} and {local_max.shape}"
                )

            R_poly = rotations[poly_ids]   # (N_poly, 3, 3)
            c_poly = centers[poly_ids]     # (N_poly, 3)
            N_poly = local_min.shape[0]

            if N_poly != poly_ids.numel():
                raise ValueError(
                    f"poly_local_aabb_* N_poly={N_poly} must match "
                    f"len(poly_object_ids)={poly_ids.numel()}"
                )

            # Build the 8 corners of the local AABB for each polyhedron
            # corner_bits: (8, 3) ∈ {0,1}^3
            corner_bits = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=DTYPE,
                device=device,
            )  # (8, 3)

            # (N_poly, 8, 3): interpolate between min and max per axis
            corners_local = (
                local_min.unsqueeze(1) * (1.0 - corner_bits.unsqueeze(0))
                + local_max.unsqueeze(1) * corner_bits.unsqueeze(0)
            )

            # Transform to world: x_world = R · x_local + c
            # R_poly: (N_poly, 3, 3); corners_local: (N_poly, 8, 3)
            corners_world = torch.einsum(
                "bij,bnj->bni", R_poly, corners_local
            ) + c_poly.unsqueeze(1)  # (N_poly, 8, 3)

            poly_min = corners_world.min(dim=1).values  # (N_poly, 3)
            poly_max = corners_world.max(dim=1).values  # (N_poly, 3)

            aabb_min[poly_ids] = poly_min
            aabb_max[poly_ids] = poly_max
        else:
            # No local AABB metadata: fall back to “no culling” for polys.
            # This keeps broad-phase *correct* but not optimal: every ray
            # will consider these objects in narrow phase.
            aabb_min[poly_ids] = -inf
            aabb_max[poly_ids] = +inf

    return aabb_min, aabb_max
