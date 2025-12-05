# pymatcal/scanner_modeling/_geometry_3d/_polyhedron.py
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor, cat, stack
from .._config import DTYPE


@dataclass
class ConvexPolyhedronBatch:
    """
    Generic convex polyhedra in local coordinates, batched.

    plane_normals_local: (N_obj, P_max, 3)
    plane_offsets_local: (N_obj, P_max)  with plane equation n·x <= d
    num_planes:          (N_obj,)  integer tensor or None (use all planes)
    
    Note: 
    num_planes: if provided, indicates how many leading planes are valid
    for each object; trailing entries are ignored.
    """
    plane_normals_local: Tensor  # (N_obj, P_max, 3)
    plane_offsets_local: Tensor  # (N_obj, P_max)
    num_planes: Optional[Tensor] = None  # (N_obj,)


def obb_planes_from_center_extents(
    half_sizes: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Construct plane normals and offsets for axis-aligned OBBs in local frame.

    Input
    -----
    half_sizes: Tensor, shape (N_obj, 3)
        e[i] = (ex, ey, ez) half-lengths along local x, y, z.

    Returns
    -------
    plane_normals_local: (N_obj, 6, 3)
    plane_offsets_local: (N_obj, 6)

    Note
    -------
    half_sizes are in mm in local coordinates and this is exactly the representation used by the OBB slab intersection
    """
    ex = half_sizes[:, 0:1]
    ey = half_sizes[:, 1:2]
    ez = half_sizes[:, 2:3]

    base_normals = half_sizes.new_tensor(
        [
            [ 1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0, -1.0],
        ],
        dtype=DTYPE,
    )  # (6, 3)

    plane_normals_local = base_normals.unsqueeze(0).expand(half_sizes.shape[0], -1, -1)
    plane_offsets_local = cat([ex, ex, ey, ey, ez, ez], dim=1)  # (N, 6)
    return plane_normals_local.to(DTYPE), plane_offsets_local.to(DTYPE)


# ---------- 3D coordinate transforms: world <-> local ----------

def local_to_world_points(
    x_local: Tensor,                 # (..., 3)
    center_world: Tensor,           # (..., 3)
    R_world_from_local: Tensor,     # (..., 3, 3)
) -> Tensor:
    """
    Local → World: p_world = R p_local + c

    Supports arbitrary leading batch dims via broadcasting.
    """
    x_local = x_local.to(DTYPE)
    center_world = center_world.to(DTYPE)
    R_world_from_local = R_world_from_local.to(DTYPE)

    # (..., 3, 3) @ (..., 3, 1) -> (..., 3, 1)
    return (R_world_from_local @ x_local.unsqueeze(-1)).squeeze(-1) + center_world


def world_to_local_points(
    x_world: Tensor,                 # (..., 3)
    center_world: Tensor,           # (..., 3)
    R_world_from_local: Tensor,     # (..., 3, 3)
) -> Tensor:
    """
    World → Local: p_local = Rᵀ (p_world - c)

    Supports arbitrary leading batch dims via broadcasting.
    """
    x_world = x_world.to(DTYPE)
    center_world = center_world.to(DTYPE)
    R_world_from_local = R_world_from_local.to(DTYPE)

    x_rel = x_world - center_world
    R_local_from_world = R_world_from_local.transpose(-1, -2)
    return (R_local_from_world @ x_rel.unsqueeze(-1)).squeeze(-1)


def local_to_world_dirs(
    d_local: Tensor,                # (..., 3)
    R_world_from_local: Tensor,     # (..., 3, 3)
) -> Tensor:
    """
    Local → World for direction vectors (no translation).

    d_world = R d_local
    """
    d_local = d_local.to(DTYPE)
    R_world_from_local = R_world_from_local.to(DTYPE)
    return (R_world_from_local @ d_local.unsqueeze(-1)).squeeze(-1)


def world_to_local_dirs(
    d_world: Tensor,                # (..., 3)
    R_world_from_local: Tensor,     # (..., 3, 3)
) -> Tensor:
    """
    World → Local for direction vectors (no translation).

    d_local = Rᵀ d_world
    """
    d_world = d_world.to(DTYPE)
    R_world_from_local = R_world_from_local.to(DTYPE)
    R_local_from_world = R_world_from_local.transpose(-1, -2)
    return (R_local_from_world @ d_world.unsqueeze(-1)).squeeze(-1)


def world_to_local_rays(
    rays_world: Tensor,             # (..., 2, 3)
    center_world: Tensor,           # (..., 3)
    R_world_from_local: Tensor,     # (..., 3, 3)
) -> Tensor:
    """
    Transform batched rays from world to local coordinates.

    rays_world: (..., 2, 3), where
        [..., 0, :] = origin
        [..., 1, :] = endpoint (or second point on ray)

    Returns:
        rays_local: (..., 2, 3) in local coordinates.
    """
    device = rays_world.device
    o_world = rays_world[..., 0, :]
    e_world = rays_world[..., 1, :]

    o_local = world_to_local_points(o_world, center_world, R_world_from_local).to(device=device, dtype=DTYPE)
    e_local = world_to_local_points(e_world, center_world, R_world_from_local).to(device=device, dtype=DTYPE)

    return stack((o_local, e_local), dim=-2)


def world_to_local_ray(
    ray_origins: Tensor,       # (N_rays, 3)
    ray_dirs: Tensor,          # (N_rays, 3)
    obj_centers: Tensor,       # (N_objs, 3)
    obj_rotations: Tensor,     # (N_objs, 3, 3)  local -> world
) -> tuple[Tensor, Tensor]:
    """
    Transform rays from world space into each object's local coordinate frame.

    For each ray i and object j:
        o_local[i, j] = R_j^T * (o_world[i] - c_j)
        d_local[i, j] = R_j^T * d_world[i]

    where:
        - c_j is the object center in world coordinates
        - R_j is the 3x3 rotation matrix taking local -> world

    Args
    ----
    ray_origins:
        (N_rays, 3) ray origins in world coordinates.
    ray_dirs:
        (N_rays, 3) ray directions in world coordinates.
    obj_centers:
        (N_objs, 3) object centers in world coordinates.
    obj_rotations:
        (N_objs, 3, 3) rotation matrices mapping local -> world.

    Returns
    -------
    ray_origins_local:
        (N_rays, N_objs, 3)
    ray_dirs_local:
        (N_rays, N_objs, 3)
    """
    if ray_origins.ndim != 2 or ray_origins.shape[-1] != 3:
        raise ValueError(f"ray_origins must have shape (N_rays, 3), got {ray_origins.shape}")
    if ray_dirs.ndim != 2 or ray_dirs.shape[-1] != 3:
        raise ValueError(f"ray_dirs must have shape (N_rays, 3), got {ray_dirs.shape}")
    if obj_centers.ndim != 2 or obj_centers.shape[-1] != 3:
        raise ValueError(f"obj_centers must have shape (N_objs, 3), got {obj_centers.shape}")
    if obj_rotations.ndim != 3 or obj_rotations.shape[-2:] != (3, 3):
        raise ValueError(
            f"obj_rotations must have shape (N_objs, 3, 3), got {obj_rotations.shape}"
        )

    device = ray_origins.device

    ray_origins = ray_origins.to(device=device, dtype=DTYPE)
    ray_dirs    = ray_dirs.to(device=device, dtype=DTYPE)
    obj_centers = obj_centers.to(device=device, dtype=DTYPE)
    obj_rotations = obj_rotations.to(device=device, dtype=DTYPE)

    N_rays = ray_origins.shape[0]
    N_objs = obj_centers.shape[0]

    # Broadcast to (N_rays, N_objs, 3)
    o_world = ray_origins[:, None, :]          # (N_rays, 1, 3)
    c_world = obj_centers[None, :, :]          # (1, N_objs, 3)
    d_world = ray_dirs[:, None, :]             # (N_rays, 1, 3)

    rel_world = o_world - c_world              # (N_rays, N_objs, 3)

    R = obj_rotations                          # (N_objs, 3, 3) local -> world
    R_T = R.transpose(-1, -2)                  # (N_objs, 3, 3) world -> local

    d_world_expanded = d_world.expand(N_rays, N_objs, 3)  # (N_rays, N_objs, 3)

    # IMPORTANT: "okj" not "ojk" → applies R_T correctly as Rᵀ @ v
    ray_origins_local = torch.einsum("okj,roj->rok", R_T, rel_world)
    ray_dirs_local    = torch.einsum("okj,roj->rok", R_T, d_world_expanded)

    return ray_origins_local, ray_dirs_local

# Backwards-compatible alias for earlier usage (if any)
def transform_points_local_to_world(
    x_local: Tensor,
    center_world: Tensor,
    R_world_from_local: Tensor,
) -> Tensor:
    """
    Alias for local_to_world_points to maintain compatibility.
    """
    return local_to_world_points(x_local, center_world, R_world_from_local)
