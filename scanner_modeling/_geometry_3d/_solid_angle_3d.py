# pymatcal/scanner_modeling/_geometry_3d/_solid_angle_3d.py

from typing import Iterable, Sequence

import torch
from torch import Tensor

from .._config import DTYPE


def solid_angle_triangle(
    v0: Tensor,
    v1: Tensor,
    v2: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    Solid angle of a single triangle as seen from the origin, using
    the Oosterom & Strackee closed-form formula.

    Parameters
    ----------
    v0, v1, v2 : Tensor
        Vertex position vectors from the observation point to each
        triangle vertex. Shape (..., 3). All broadcastable to a
        common shape.

        Example:
            - Unbatched: (3,)
            - Batched:   (N, 3), or (N_det, N_tri, 3), etc.

    eps : float
        Numerical epsilon to avoid division by zero in degenerate
        configurations.

    Returns
    -------
    omega : Tensor
        Solid angle in steradians, shape equal to the broadcasted
        leading dimensions of v0, v1, v2. Non-negative magnitude;
        orientation of the triangle (winding) is not preserved.

    Notes
    -----
    Oosterom & Strackee formula:

        Ω = 2 * atan2( |a · (b × c)|,
                       ||a|| ||b|| ||c||
                       + (a·b) ||c||
                       + (a·c) ||b||
                       + (b·c) ||a|| )

    where a, b, c are vertex vectors from the observation point.
    """
    v0 = v0.to(dtype=DTYPE)
    v1 = v1.to(dtype=DTYPE)
    v2 = v2.to(dtype=DTYPE)

    # Broadcast to common shape
    a, b, c = torch.broadcast_tensors(v0, v1, v2)

    # Norms
    la = a.norm(dim=-1)
    lb = b.norm(dim=-1)
    lc = c.norm(dim=-1)

    # Dot products
    ab = (a * b).sum(dim=-1)
    ac = (a * c).sum(dim=-1)
    bc = (b * c).sum(dim=-1)

    # Triple product a · (b × c)
    cross_bc = torch.cross(b, c, dim=-1)
    triple = (a * cross_bc).sum(dim=-1)

    numer = triple.abs()

    denom = (
        la * lb * lc
        + ab * lc
        + ac * lb
        + bc * la
    )

    # Guard against degenerate cases
    eps_t = torch.tensor(eps, dtype=DTYPE, device=denom.device)
    denom_safe = torch.where(denom.abs() < eps_t, eps_t, denom)

    omega = 2.0 * torch.atan2(numer, denom_safe)

    # For nearly-degenerate triangles, force Ω → 0
    omega = torch.where(numer <= eps_t, torch.zeros_like(omega), omega)

    return omega


def solid_angle_triangles_batch(
    src_points: Tensor,      # (N_src, 3)
    tri_verts: Tensor,       # (N_det_vox, N_tri, 3, 3)
    eps: float = 1e-12,
) -> Tensor:
    """
    Solid angle from multiple source points to multiple triangle meshes.

    Parameters
    ----------
    src_points : Tensor, shape (N_src, 3)
        Observation points (e.g. FOV voxel centers) in world coords.

    tri_verts : Tensor, shape (N_det_vox, N_tri, 3, 3)
        Triangle vertices for each detector voxel in world coords.
        tri_verts[q, t, k, :] = vertex k of triangle t of detector
        voxel q, k ∈ {0,1,2}.

    eps : float
        Numerical epsilon passed to solid_angle_triangle.

    Returns
    -------
    omega : Tensor, shape (N_src, N_det_vox)
        omega[j, q] = Σ_t Ω(j → triangle t of voxel q)

    Notes
    -----
    - This is a geometric primitive that sums per-triangle solid
      angles for each (source, voxel) pair.
    - It assumes all triangles in tri_verts belong to the same
      physical surface (e.g. entrance face or union of faces).
    """
    if src_points.ndim != 2 or src_points.shape[-1] != 3:
        raise ValueError(
            f"src_points must have shape (N_src, 3); got {tuple(src_points.shape)}"
        )
    if tri_verts.ndim != 4 or tri_verts.shape[-2:] != (3, 3):
        raise ValueError(
            f"tri_verts must have shape (N_det_vox, N_tri, 3, 3); "
            f"got {tuple(tri_verts.shape)}"
        )

    src_points = src_points.to(dtype=DTYPE)
    tri_verts = tri_verts.to(dtype=DTYPE)

    N_src = src_points.shape[0]
    N_det_vox, N_tri = tri_verts.shape[:2]

    # Expand:
    #   src_points  -> (N_src, 1, 1, 1, 3)
    #   tri_verts   -> (1, N_det_vox, N_tri, 3, 3)
    src_exp = src_points.view(N_src, 1, 1, 1, 3)
    tris_exp = tri_verts.view(1, N_det_vox, N_tri, 3, 3)

    # Vectors from source to triangle vertices
    v = tris_exp - src_exp  # (N_src, N_det_vox, N_tri, 3, 3)

    v0 = v[..., 0, :]
    v1 = v[..., 1, :]
    v2 = v[..., 2, :]

    omega_tri = solid_angle_triangle(v0, v1, v2, eps=eps)  # (N_src, N_det_vox, N_tri)
    omega = omega_tri.sum(dim=-1)                          # (N_src, N_det_vox)

    return omega


def solid_angle_detector_voxel(
    fov_voxel_pos: Tensor,      # (N_fov, 3)
    voxel_faces: Tensor,        # (N_det_vox, N_faces, N_tri_per_face, 3, 3)
    faces: str | Sequence[int] = "front",
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute Ω_{iqj} for detector voxels as seen from FOV voxels, using
    Oosterom & Strackee on triangle meshes.

    Parameters
    ----------
    fov_voxel_pos : Tensor, shape (N_fov, 3)
        FOV voxel centers j in world coordinates (mm).

    voxel_faces : Tensor, shape (N_det_vox, N_faces, N_tri_per_face, 3, 3)
        Triangulated faces of each detector voxel (i, q) in world coords.
        The convention is:
            voxel_faces[q, f, t, k, :] =
                vertex k of triangle t on face f of voxel q.

    faces : {"front", "all"} or Sequence[int]
        Which faces to include:

        - "front" (default): only face index 0 is used; this is assumed
          to be the entrance face.
        - "all": all faces are included.
        - Sequence[int]: explicit list of face indices (e.g. [0, 1, 2, 3, 4]).

    eps : float
        Numerical epsilon for solid-angle primitives.

    Returns
    -------
    omega_iqj : Tensor, shape (N_det_vox, N_fov)
        omega_iqj[q, j] = Ω_{iqj} in steradians.

    Notes
    -----
    - The implementation is fully vectorized over FOV voxels, detector
      voxels, faces and triangles.
    - All coordinates must be in the global DISTANCE_UNIT (mm).
    """
    if voxel_faces.ndim != 5 or voxel_faces.shape[-2:] != (3, 3):
        raise ValueError(
            f"voxel_faces must have shape (N_det_vox, N_faces, N_tri_per_face, 3, 3); "
            f"got {tuple(voxel_faces.shape)}"
        )

    N_det_vox, N_faces, N_tri_per_face = voxel_faces.shape[:3]

    # Resolve which faces to include
    if isinstance(faces, str):
        if faces == "front":
            face_indices = [0]
        elif faces == "all":
            face_indices = list(range(N_faces))
        else:
            raise ValueError(
                f"faces='{faces}' not understood; use 'front', 'all', "
                f"or a sequence of face indices."
            )
    else:
        # Assume iterable of ints
        face_indices = list(faces)
        if len(face_indices) == 0:
            raise ValueError("faces sequence must be non-empty")

        if max(face_indices) >= N_faces or min(face_indices) < 0:
            raise ValueError(
                f"faces indices must be in [0, {N_faces-1}], got {face_indices}"
            )

    # Select and flatten the chosen faces → triangles per voxel
    faces_sel = voxel_faces[:, face_indices, :, :, :]  # (N_det_vox, N_sel, N_tri_per_face, 3, 3)
    N_sel = len(face_indices)

    tri_verts = faces_sel.reshape(
        N_det_vox,
        N_sel * N_tri_per_face,
        3,
        3,
    )  # (N_det_vox, N_tri_total, 3, 3)

    omega_src_det = solid_angle_triangles_batch(
        fov_voxel_pos,
        tri_verts,
        eps=eps,
    )  # (N_fov, N_det_vox)

    # Return as (N_det_vox, N_fov) to match PPDF convention f_{iqj}[q, j]
    return omega_src_det.transpose(0, 1)
