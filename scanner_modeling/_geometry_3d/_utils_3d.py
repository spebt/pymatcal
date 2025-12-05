from typing import Dict, Sequence

import torch
from torch import Tensor, arange, meshgrid, stack, tensor
from .._config import DTYPE



def fov_tensor_dict_3d(
    n_voxels: Sequence[int] = (128, 128, 128),
    size_in_mm: Sequence[float] = (256.0, 256.0, 256.0),
    center_coordinates: Sequence[float] = (0.0, 0.0, 0.0),
    n_subdivisions: Sequence[int] = (1, 1, 1),
) -> Dict[str, Tensor]:
    """
    3D analogue of fov_tensor_dict (2D).

    All tensors are Float64 by default, distances in mm.

    Parameters
    ----------
    n_voxels : (Nx, Ny, Nz)
    size_in_mm : (Lx, Ly, Lz), in mm
    center_coordinates : (cx, cy, cz), in mm
    n_subdivisions: (Sx, Sy, Sz) for later SFOV-like splitting

    Notes
    -----
    - "n voxels" and "n subdivisions" are stored as Float64 for convenience
      but conceptually represent integer counts.
    - Only "size in mm" and derived fields ("mm per voxel") are physical
      distances in the chosen DISTANCE_UNIT.

    """
    fov_dict = {
        "n voxels": tensor(n_voxels, dtype=DTYPE),
        "size in mm": tensor(size_in_mm, dtype=DTYPE),
        "center coordinates in mm": tensor(center_coordinates, dtype=DTYPE),
    }
    fov_dict["mm per voxel"] = fov_dict["size in mm"] / fov_dict["n voxels"]
    fov_dict["n subdivisions"] = tensor(n_subdivisions, dtype=DTYPE)
    return fov_dict


def fov_corners_vertices_3d(fov_dict: Dict[str, Tensor]) -> Tensor:
    """
    Return the 8 corner vertices of the 3D FOV box in world coordinates.

    Shape: (8, 3)
    Vertices are ordered in a consistent pattern, but only convex hull
    and OBB kernels will care about the full set.
    """
    # corners in [-1, 1]^3
    base = tensor(
        [
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1],
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1],
        ],
        dtype=DTYPE,
    )
    return base * fov_dict["size in mm"] * 0.5 + fov_dict["center coordinates in mm"]


def voxels_coordinates_3d(fov_dict: Dict[str, Tensor]) -> Tensor:
    """
    3D analogue of pixels_coordinates.

    Returns:
        Tensor of shape (N_total, 3) where
        N_total = Nx * Ny * Nz = product of 'n voxels'.

    Voxel centers are in world coordinates (mm).
    """
    nx, ny, nz = [int(v.item()) for v in fov_dict["n voxels"]]
    ix = arange(0, nx)
    iy = arange(0, ny)
    iz = arange(0, nz)

    grid = stack(
        meshgrid(ix, iy, iz, indexing="ij"),
        dim=-1,  # (Nx, Ny, Nz, 3)
    ).view(-1, 3)

    # Convert to Float64 and to mm
    grid = grid.to(dtype=DTYPE)
    return (
        (grid + 0.5) * fov_dict["mm per voxel"]
        - fov_dict["size in mm"] * 0.5
        + fov_dict["center coordinates in mm"]
    )

def rotation_from_azimuth_tilt(
    azimuth,
    tilt,
    roll=0.0,
) -> Tensor:
    """
    Construct batched 3×3 rotation matrices from scanner angles.

    All angles are in **radians**.

    Parameters
    ----------
    azimuth : float or Tensor
        In-plane ring angle, rotation about the global +z axis.
        Shape: broadcastable with `tilt` and `roll`.

    tilt : float or Tensor
        Out-of-plane tilt (elevation), rotation about the (global) y axis.
        Shape: broadcastable with `azimuth` and `roll`.

    roll : float or Tensor, optional
        Spin around the detector normal, implemented as an additional
        rotation about the local z axis. Shape: broadcastable.

    Returns
    -------
    R_world_from_local : Tensor
        Rotation matrices of shape (..., 3, 3) with dtype=DTYPE.
        Interpreted as: x_world = R_world_from_local @ x_local.

    Notes
    -----
    The rotation is composed as:

        R_world_from_local = Rz(azimuth) @ Ry(tilt) @ Rz(roll)

    with the standard right-handed convention:

        Rz(a) = [[ cos a, -sin a, 0],
                 [ sin a,  cos a, 0],
                 [     0,      0, 1]]

        Ry(t) = [[ cos t, 0, sin t],
                 [     0, 1,     0],
                 [-sin t, 0, cos t]]

    This function is fully vectorized: `azimuth`, `tilt`, `roll` may be
    scalars or tensors with arbitrary broadcastable shapes.
    """
    # Infer device from any tensor argument; default to CPU if all are scalars.
    device = None
    for arg in (azimuth, tilt, roll):
        if isinstance(arg, Tensor):
            device = arg.device
            break

    az = torch.as_tensor(azimuth, dtype=DTYPE, device=device)
    tl = torch.as_tensor(tilt, dtype=DTYPE, device=device)
    rl = torch.as_tensor(roll, dtype=DTYPE, device=device)

    # Broadcast all angle tensors to a common shape
    az, tl, rl = torch.broadcast_tensors(az, tl, rl)

    # Trig terms
    cos_az, sin_az = torch.cos(az), torch.sin(az)
    cos_tl, sin_tl = torch.cos(tl), torch.sin(tl)
    cos_rl, sin_rl = torch.cos(rl), torch.sin(rl)

    zeros = torch.zeros_like(az)
    ones = torch.ones_like(az)

    # Rz(azimuth)
    Rz_az = torch.stack(
        [
            torch.stack([cos_az, -sin_az, zeros], dim=-1),
            torch.stack([sin_az,  cos_az, zeros], dim=-1),
            torch.stack([zeros,   zeros,  ones],  dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)

    # Ry(tilt)
    Ry_tl = torch.stack(
        [
            torch.stack([cos_tl, zeros,  sin_tl], dim=-1),
            torch.stack([zeros,  ones,   zeros],  dim=-1),
            torch.stack([-sin_tl, zeros, cos_tl], dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)

    # Rz(roll)
    Rz_rl = torch.stack(
        [
            torch.stack([cos_rl, -sin_rl, zeros], dim=-1),
            torch.stack([sin_rl,  cos_rl, zeros], dim=-1),
            torch.stack([zeros,   zeros,  ones],  dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)

    # Compose: R = Rz(azimuth) @ Ry(tilt) @ Rz(roll)
    R_world_from_local = torch.einsum(
        "...ij,...jk,...kl->...il", Rz_az, Ry_tl, Rz_rl
    )

    return R_world_from_local

def apply_transform(
    points: Tensor,
    translation: Tensor,
    rotation: Tensor,
    inverse: bool = False,
) -> Tensor:
    """
    Generic batched 3D transform utility.

    Parameters
    ----------
    points : Tensor (..., 3)
        Points in local or world space. The last dimension is xyz.
        We interpret the shape as (*B, N, 3), where *B are batch dims
        (possibly empty) and N is the number of points per batch.

    translation : Tensor (..., 3)
        Translation vector 'c'. Must be broadcastable to shape (*B, 3),
        i.e. either shape (3,) or (*B, 3).

    rotation : Tensor (..., 3, 3)
        Rotation matrices. Must be broadcastable to shape (*B, 3, 3),
        i.e. either shape (3, 3) or (*B, 3, 3).

    inverse : bool
        - False: world ← local
            p_world = R ⋅ p_local + c
        - True: local ← world
            p_local = Rᵀ ⋅ (p_world - c)

    Returns
    -------
    transformed : Tensor (..., 3)
        Points transformed with broadcasting along batch dimensions.
    """
    device = points.device

    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(
            f"'points' must have shape (..., 3), got {tuple(points.shape)}"
        )

    points = points.to(dtype=DTYPE, device=device)

    # Interpret points as (*B, N, 3)
    B_shape = points.shape[:-2]  # batch dims (possibly empty)
    N = points.shape[-2]

    # --- Normalize rotation to shape (*B, 3, 3) ---
    rotation = rotation.to(dtype=DTYPE, device=device)

    if rotation.ndim == 2:
        if rotation.shape != (3, 3):
            raise ValueError(
                f"When 'rotation' is 2D, it must be (3,3), got {tuple(rotation.shape)}"
            )
        rotation = rotation.expand(B_shape + (3, 3))
    elif rotation.ndim >= 3:
        if rotation.shape[-2:] != (3, 3):
            raise ValueError(
                f"'rotation' last two dims must be (3,3), got {tuple(rotation.shape[-2:])}"
            )
        if rotation.shape[:-2] != B_shape:
            raise ValueError(
                f"'rotation' batch dims {rotation.shape[:-2]} "
                f"do not match points batch dims {B_shape}"
            )
    else:
        raise ValueError(
            f"'rotation' must have ndim>=2, got {rotation.ndim}"
        )

    # --- Normalize translation to shape (*B, 3) ---
    translation = translation.to(dtype=DTYPE, device=device)

    if translation.ndim == 1:
        if translation.shape[-1] != 3:
            raise ValueError(
                f"When 'translation' is 1D, it must be (3,), got {tuple(translation.shape)}"
            )
        translation = translation.expand(B_shape + (3,))
    elif translation.ndim >= 2:
        if translation.shape[-1] != 3:
            raise ValueError(
                f"'translation' last dim must be 3, got {tuple(translation.shape)}"
            )
        if translation.shape[:-1] != B_shape:
            raise ValueError(
                f"'translation' batch dims {translation.shape[:-1]} "
                f"do not match points batch dims {B_shape}"
            )
    else:
        raise ValueError(
            f"'translation' must have ndim>=1, got {translation.ndim}"
        )

    # --- Flatten batch dims for einsum: (*B, N, 3) -> (B_flat, N, 3) ---
    p = points.reshape(-1, N, 3)
    R = rotation.reshape(-1, 3, 3)
    c = translation.reshape(-1, 3).unsqueeze(-2)  # (B_flat, 1, 3)

    if not inverse:
        # world = R @ local + c
        p_trans = torch.einsum("bij,bnj->bni", R, p) + c
    else:
        # local = Rᵀ @ (world - c)
        p_centered = p - c
        R_T = R.transpose(-1, -2)
        p_trans = torch.einsum("bij,bnj->bni", R_T, p_centered)

    # Reshape back to original batch shape
    return p_trans.reshape(B_shape + (N, 3))

def reorder_detector_voxel_faces_front_first(
    voxel_faces: Tensor,
    fov_center_world: Tensor,
) -> Tensor:
    """
    Reorder detector voxel faces so that face index 0 is the “front” face
    (entrance) closest to the FOV center.

    This is intended for MATRICES-style ring geometries where the FOV is
    approximately centered and detectors face inward.

    Parameters
    ----------
    voxel_faces : Tensor
        Detector voxel faces, shape:
            (N_det_vox, N_faces, N_tri_per_face, 3, 3)

        The convention is:
            voxel_faces[q, f, t, k, :] =
                vertex k of triangle t on face f of voxel q
            with k ∈ {0, 1, 2}.

        Triangles per face are assumed to lie on a common plane, but we do
        *not* assume any particular winding order.

    fov_center_world : Tensor
        FOV center in world coordinates, in mm. Either:
            - shape (3,) — same center for all detector voxels, or
            - shape (N_det_vox, 3) — per-voxel “target” point.

    Returns
    -------
    voxel_faces_reordered : Tensor
        Same as `voxel_faces`, but with the faces dimension permuted so that
        face 0 is the face whose center is closest (in Euclidean distance)
        to the FOV center.

    Notes
    -----
    - This function does *not* compute solid angles; it only enforces a
      consistent face ordering convention: index 0 = entrance face.
    - Selection is purely geometric: we pick, for each voxel, the face whose
      centroid is nearest to the FOV center. This is robust to detector tilt
      and works for generic ring geometries.
    """
    if voxel_faces.ndim != 5 or voxel_faces.shape[-2:] != (3, 3):
        raise ValueError(
            "voxel_faces must have shape (N_det_vox, N_faces, "
            f"N_tri_per_face, 3, 3); got {tuple(voxel_faces.shape)}"
        )

    N_det_vox, N_faces, N_tri_per_face = voxel_faces.shape[:3]

    device = voxel_faces.device
    voxel_faces = voxel_faces.to(dtype=DTYPE, device=device)

    # --- Normalize fov_center_world shape ---
    fov_center_world = torch.as_tensor(
        fov_center_world, dtype=DTYPE, device=device
    )

    if fov_center_world.shape == (3,):
        # Shared FOV center for all detector voxels
        fov_center_world = fov_center_world.view(1, 1, 1, 3)
        per_voxel_target = False
    elif fov_center_world.shape == (N_det_vox, 3):
        # Per-voxel target (e.g. for more exotic geometries)
        fov_center_world = fov_center_world.view(N_det_vox, 1, 1, 3)
        per_voxel_target = True
    else:
        raise ValueError(
            "fov_center_world must have shape (3,) or (N_det_vox, 3); "
            f"got {tuple(fov_center_world.shape)}"
        )

    # --- Compute face centroids ---
    # Mean over triangles and vertices: (N_det_vox, N_faces, 3)
    face_centers = voxel_faces.mean(dim=(2, 3))

    if not per_voxel_target:
        diff = face_centers - fov_center_world.view(1, 1, 3)
    else:
        diff = face_centers - fov_center_world.view(N_det_vox, 1, 3)

    dist2 = (diff * diff).sum(dim=-1)          # (N_det_vox, N_faces)
    front_idx = dist2.argmin(dim=1)            # (N_det_vox,)

    # --- Reorder faces: put front face at index 0 ---
    # This is a one-time geometry preprocessing step, so a small Python
    # loop over voxels is acceptable and keeps the logic simple.
    reordered_list = []
    for q in range(N_det_vox):
        fi = int(front_idx[q])
        if N_faces == 1:
            order = [0]
        else:
            rest = [i for i in range(N_faces) if i != fi]
            order = [fi] + rest

        perm = torch.tensor(order, dtype=torch.long, device=device)
        reordered_list.append(voxel_faces[q, perm, ...])

    voxel_faces_reordered = torch.stack(reordered_list, dim=0)
    return voxel_faces_reordered
