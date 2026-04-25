
from dataclasses import dataclass

import torch
from torch import Tensor
from math import pi
from typing import Dict, Tuple, Optional

from .._config import DTYPE
from ..geometry_3d import world_to_local_ray  # world → local rays, batched
from ..geometry_3d import OBJECT_TYPE_OBB, OBJECT_TYPE_CONVEX_POLY  # type codes

from ._local_functions_3d import rays_3d_batch  # FOV↔detector rays
from ._intersection_obb_3d import ray_obb_intersection_local
from ._intersection_polyhedron_3d import ray_convex_polyhedron_intersection_local

@dataclass
class RayPPDFTerms3D:
    """
    Container for the geometric / path-length terms of a ray (i, q, j)
    in the 3D PPDF derivation.

    All tensors are Float64 and distances in mm.

    Attributes
    ----------
    L_objects : Tensor
        Shape (..., K).
        L_objects excludes the detector voxel contributions and only external materials go into μ_k L_k

    S_voxel : Tensor
        Shape (...,).
        S_voxel[...] = S_{iqj}: traversal distance within detector voxel (i, q).

    solid_angle : Tensor
        Shape (...,).
        solid_angle[...] = Ω_{iqj}: solid angle of voxel (i,q) as seen from FOV voxel j.
        Units: steradians.

    Note: L_{iqji} (total traversal length in detector i) can be represented
    either as a separate tensor or as sum over a subset of L_objects, depending
    on how detector layers are modeled. For the core PPDF factor we only need
    Σ_k μ_k L_{iqjk} and μ_i S_{iqj}.
    """

    L_objects: Tensor          # (..., K)
    S_voxel: Tensor            # (...,)
    solid_angle: Tensor        # (...,)

    def to(self, device=None, dtype=None) -> "RayPPDFTerms3D":
        """Convenience move/cast."""
        if dtype is None:
            dtype = DTYPE
        if device is None:
            device = self.L_objects.device
        return RayPPDFTerms3D(
            L_objects=self.L_objects.to(device=device, dtype=dtype),
            S_voxel=self.S_voxel.to(device=device, dtype=dtype),
            solid_angle=self.solid_angle.to(device=device, dtype=dtype),
        )


@dataclass
class SparsePathLengths:
    """Sparse representation of ray-object path lengths (only actual hits)."""
    ray_indices: Tensor    # (N_hits,) which ray
    obj_indices: Tensor    # (N_hits,) which object (global index)
    lengths: Tensor        # (N_hits,) path length L for each hit pair
    N_rays: int
    N_obj: int


def _compute_sparse_path_lengths(
    ray_origins_world: Tensor,          # (N_rays, 3)
    ray_dirs_world: Tensor,             # (N_rays, 3)
    objects: Dict[str, Tensor],
    sparse_ray_indices: Tensor,         # (N_pairs,) from broad phase
    sparse_obj_indices: Tensor,         # (N_pairs,) from broad phase
) -> SparsePathLengths:
    """
    Compute exact per-ray, per-object path lengths for sparse candidate pairs.
    Returns only pairs with nonzero path length (true narrow-phase hits).
    No dense (N_rays, N_obj) allocation.
    """
    ray_origins_world = ray_origins_world.to(dtype=DTYPE)
    ray_dirs_world = ray_dirs_world.to(dtype=DTYPE)

    centers = objects["centers"]
    rotations = objects["rotations"]

    N_rays = int(ray_origins_world.shape[0])
    N_obj = int(centers.shape[0])
    device = ray_origins_world.device

    if N_obj == 0 or sparse_ray_indices.numel() == 0:
        return SparsePathLengths(
            ray_indices=torch.empty(0, dtype=torch.int64, device=device),
            obj_indices=torch.empty(0, dtype=torch.int64, device=device),
            lengths=torch.empty(0, dtype=DTYPE, device=device),
            N_rays=N_rays, N_obj=N_obj,
        )

    N_pairs = sparse_ray_indices.shape[0]

    # 1. Gather world data for candidate pairs only
    o_w = ray_origins_world[sparse_ray_indices]   # (N_pairs, 3)
    d_w = ray_dirs_world[sparse_ray_indices]       # (N_pairs, 3)

    # Rotation is norm-preserving: ||R^T d_w|| = ||d_w||.
    # Compute once here and reuse for both OBB and poly branches (Issue #4).
    d_w_norm = d_w.norm(dim=-1)                    # (N_pairs,)

    c_obj = centers[sparse_obj_indices]            # (N_pairs, 3)
    R_obj = rotations[sparse_obj_indices]          # (N_pairs, 3, 3)

    # 2. Transform to local space
    rel_pos = o_w - c_obj
    R_T = R_obj.transpose(-1, -2)
    o_local = torch.einsum("nij,nj->ni", R_T, rel_pos)
    d_local = torch.einsum("nij,nj->ni", R_T, d_w)

    # Allocate path lengths for all candidate pairs
    all_L = torch.zeros(N_pairs, dtype=DTYPE, device=device)

    # 3. Dispatch by object type
    obj_types = objects["type"][sparse_obj_indices]

    # --- OBB Handling ---
    # Build reverse map: global_obj_id -> local OBB index into half_sizes
    is_obb = (obj_types == OBJECT_TYPE_OBB)
    if is_obb.any():
        obb_object_ids = objects["obb_object_ids"]  # (N_obb,) global IDs of OBBs
        global_to_obb_local = torch.full((N_obj,), -1, dtype=torch.long, device=device)
        global_to_obb_local[obb_object_ids] = torch.arange(obb_object_ids.shape[0], device=device)

        global_obj_ids = sparse_obj_indices[is_obb]
        obb_local_ids = global_to_obb_local[global_obj_ids]
        half_sizes_sub = objects["half_sizes"][obb_local_ids]

        t_in, t_out, hits = ray_obb_intersection_local(
            o_local[is_obb], d_local[is_obb], half_sizes_sub
        )

        L = torch.clamp(t_out - t_in, min=0.0) * d_w_norm[is_obb]
        L = torch.where(hits, L, torch.zeros_like(L))
        all_L[is_obb] = L

    # --- Poly Handling ---
    # Build reverse map: global_obj_id -> local poly index into plane arrays
    is_poly = (obj_types == OBJECT_TYPE_CONVEX_POLY)
    if is_poly.any():
        poly_object_ids = objects["poly_object_ids"]  # (N_poly,) global IDs of polys
        global_to_poly_local = torch.full((N_obj,), -1, dtype=torch.long, device=device)
        global_to_poly_local[poly_object_ids] = torch.arange(poly_object_ids.shape[0], device=device)

        poly_local_indices = global_to_poly_local[sparse_obj_indices[is_poly]]

        plane_normals = objects["plane_normals_local"][poly_local_indices]
        plane_offsets = objects["plane_offsets_local"][poly_local_indices]

        plane_mask = None
        if "num_planes" in objects:
            num_p = objects["num_planes"][poly_local_indices]
            P_max = plane_offsets.shape[1]
            idx = torch.arange(P_max, device=device).view(1, -1)
            plane_mask = idx < num_p.view(-1, 1)

        t_in, t_out, hits = ray_convex_polyhedron_intersection_local(
            o_local[is_poly].unsqueeze(0),
            d_local[is_poly].unsqueeze(0),
            plane_normals,
            plane_offsets,
            plane_mask
        )

        t_in = t_in.squeeze(0)
        t_out = t_out.squeeze(0)
        hits = hits.squeeze(0)

        L = torch.clamp(t_out - t_in, min=0.0) * d_w_norm[is_poly]
        L = torch.where(hits, L, torch.zeros_like(L))
        all_L[is_poly] = L

    # 4. Filter out zero-length (AABB hit but narrow-phase miss)
    nonzero_mask = all_L > 0
    return SparsePathLengths(
        ray_indices=sparse_ray_indices[nonzero_mask],
        obj_indices=sparse_obj_indices[nonzero_mask],
        lengths=all_L[nonzero_mask],
        N_rays=N_rays, N_obj=N_obj,
    )


def ray_object_path_lengths_world(
    ray_origins_world: Tensor,          # (N_rays, 3)
    ray_dirs_world: Tensor,             # (N_rays, 3)
    objects: Dict[str, Tensor],
    sparse_ray_indices: Tensor = None,  # (N_pairs,) indices of rays that hit AABB
    sparse_obj_indices: Tensor = None,  # (N_pairs,) indices of objs that hit AABB
) -> Tuple[Tensor, Tensor]:
    """
    Compute exact per-ray, per-object path lengths in world space.
    DENSE mode only — returns (N_rays, N_obj) tensors.

    For the sparse broad-phase path, use _compute_sparse_path_lengths() instead
    (called internally by ppdf_3d_local).

    This function is kept for backward compatibility with callers that expect
    dense output.
    """
    # Ensure float dtype / device consistency
    ray_origins_world = ray_origins_world.to(dtype=DTYPE)
    ray_dirs_world = ray_dirs_world.to(dtype=DTYPE)

    centers = objects["centers"]
    rotations = objects["rotations"]

    N_rays = int(ray_origins_world.shape[0])
    N_obj = int(centers.shape[0])
    device = ray_origins_world.device

    if N_obj == 0:
        L_empty = torch.zeros((N_rays, 0), dtype=DTYPE, device=device)
        hit_empty = torch.zeros((N_rays, 0), dtype=torch.bool, device=device)
        return L_empty, hit_empty

    # Allocate global outputs (Default 0 for no intersection)
    L_objects = torch.zeros((N_rays, N_obj), dtype=DTYPE, device=device)
    hit_mask_all = torch.zeros((N_rays, N_obj), dtype=torch.bool, device=device)

    # =========================================================
    # PATH A: SPARSE COMPUTATION (Using Broad Phase Results)
    # =========================================================
    if sparse_ray_indices is not None and sparse_obj_indices is not None:
        if sparse_ray_indices.numel() == 0:
            return L_objects, hit_mask_all

        # 1. Gather world data for specific pairs only (No broadcasting!)
        #    (N_pairs, 3)
        o_w = ray_origins_world[sparse_ray_indices]
        d_w = ray_dirs_world[sparse_ray_indices]

        c_obj = centers[sparse_obj_indices]
        R_obj = rotations[sparse_obj_indices] # (N_pairs, 3, 3)

        # 2. Transform to Local Space (Sparse)
        #    p_local = R^T @ (p_world - center)
        rel_pos = o_w - c_obj
        R_T = R_obj.transpose(-1, -2)

        # Element-wise batch matmul: (N_pairs, 3, 3) @ (N_pairs, 3, 1) -> (N_pairs, 3)
        o_local = torch.einsum("nij,nj->ni", R_T, rel_pos)
        d_local = torch.einsum("nij,nj->ni", R_T, d_w)

        # Rotation is norm-preserving — compute once, reuse for OBB and poly
        d_w_norm = d_w.norm(dim=-1)  # (N_pairs,)

        # 3. Dispatch based on Object Type
        obj_types = objects["type"][sparse_obj_indices]

        # --- OBB Handling ---
        # Robust indexing: build reverse map global_id -> local OBB index
        is_obb = (obj_types == OBJECT_TYPE_OBB)
        if is_obb.any():
            mask_obb = is_obb
            obb_object_ids = objects["obb_object_ids"]
            global_to_obb_local = torch.full((N_obj,), -1, dtype=torch.long, device=device)
            global_to_obb_local[obb_object_ids] = torch.arange(obb_object_ids.shape[0], device=device)

            global_obj_ids = sparse_obj_indices[mask_obb]
            obb_local_ids = global_to_obb_local[global_obj_ids]
            half_sizes_sub = objects["half_sizes"][obb_local_ids]

            t_in, t_out, hits = ray_obb_intersection_local(
                o_local[mask_obb], d_local[mask_obb], half_sizes_sub
            )

            L = torch.clamp(t_out - t_in, min=0.0) * d_w_norm[mask_obb]
            L = torch.where(hits, L, torch.zeros_like(L))

            L_objects.index_put_(
                (sparse_ray_indices[mask_obb], sparse_obj_indices[mask_obb]),
                L
            )
            hit_mask_all.index_put_(
                (sparse_ray_indices[mask_obb], sparse_obj_indices[mask_obb]),
                hits
            )

        # --- Poly Handling ---
        # Robust indexing: build reverse map global_id -> local poly index
        is_poly = (obj_types == OBJECT_TYPE_CONVEX_POLY)
        if is_poly.any():
            mask_poly = is_poly
            poly_object_ids = objects["poly_object_ids"]
            global_to_poly_local = torch.full((N_obj,), -1, dtype=torch.long, device=device)
            global_to_poly_local[poly_object_ids] = torch.arange(poly_object_ids.shape[0], device=device)

            poly_local_indices = global_to_poly_local[sparse_obj_indices[mask_poly]]

            plane_normals = objects["plane_normals_local"][poly_local_indices]
            plane_offsets = objects["plane_offsets_local"][poly_local_indices]

            plane_mask = None
            if "num_planes" in objects:
                num_p = objects["num_planes"][poly_local_indices]
                P_max = plane_offsets.shape[1]
                idx = torch.arange(P_max, device=device).view(1, -1)
                plane_mask = idx < num_p.view(-1, 1)

            t_in, t_out, hits = ray_convex_polyhedron_intersection_local(
                o_local[mask_poly].unsqueeze(0),
                d_local[mask_poly].unsqueeze(0),
                plane_normals,
                plane_offsets,
                plane_mask
            )

            t_in = t_in.squeeze(0)
            t_out = t_out.squeeze(0)
            hits = hits.squeeze(0)

            L = torch.clamp(t_out - t_in, min=0.0) * d_w_norm[mask_poly]
            L = torch.where(hits, L, torch.zeros_like(L))

            L_objects.index_put_(
                (sparse_ray_indices[mask_poly], sparse_obj_indices[mask_poly]),
                L
            )
            hit_mask_all.index_put_(
                (sparse_ray_indices[mask_poly], sparse_obj_indices[mask_poly]),
                hits
            )

        return L_objects, hit_mask_all

    # =========================================================
    # PATH B: DENSE COMPUTATION (Broadcasting) - Fallback
    # =========================================================
    # 1) Transform rays into each object's local frame
    o_local, d_local = world_to_local_ray(
        ray_origins_world, ray_dirs_world, centers, rotations
    )

    # Rotation is norm-preserving — compute world norm once, index per object type
    d_w_norm = ray_dirs_world.norm(dim=-1)  # (N_rays,)

    # 2) OBB objects
    obb_ids = objects.get("obb_object_ids", None)
    if obb_ids is not None and obb_ids.numel() > 0:
        half_sizes = objects["half_sizes"]
        o_obb = o_local[:, obb_ids, :]
        d_obb = d_local[:, obb_ids, :]

        t_enter_obb, t_exit_obb, hit_obb = ray_obb_intersection_local(
            o_obb, d_obb, half_sizes
        )
        # d_w_norm: (N_rays,) → unsqueeze to broadcast over N_obb objects
        L_obb = torch.clamp(t_exit_obb - t_enter_obb, min=0.0) * d_w_norm.unsqueeze(-1)
        L_obb = torch.where(hit_obb, L_obb, torch.zeros((), dtype=DTYPE, device=device))

        L_objects[:, obb_ids] = L_obb
        hit_mask_all[:, obb_ids] = hit_obb

    # 3) Convex poly objects
    poly_ids = objects.get("poly_object_ids", None)
    if poly_ids is not None and poly_ids.numel() > 0:
        plane_normals = objects["plane_normals_local"]
        plane_offsets = objects["plane_offsets_local"]

        plane_mask = None
        if "num_planes" in objects:
            num_planes = objects["num_planes"]
            P_max = plane_offsets.shape[1]
            idx = torch.arange(P_max, device=device).view(1, -1)
            plane_mask = idx < num_planes.view(-1, 1)

        o_poly = o_local[:, poly_ids, :]
        d_poly = d_local[:, poly_ids, :]

        t_enter_poly, t_exit_poly, hit_poly = ray_convex_polyhedron_intersection_local(
            o_poly, d_poly, plane_normals, plane_offsets, plane_mask
        )

        L_poly = torch.clamp(t_exit_poly - t_enter_poly, min=0.0) * d_w_norm.unsqueeze(-1)
        L_poly = torch.where(hit_poly, L_poly, torch.zeros((), dtype=DTYPE, device=device))

        L_objects[:, poly_ids] = L_poly
        hit_mask_all[:, poly_ids] = hit_poly

    return L_objects, hit_mask_all


def ppdf_ray_factor_3d(
    terms: RayPPDFTerms3D,
    mu_objects: Tensor,         # shape (K,) or broadcastable to (..., K)
    mu_detector: Tensor,        # scalar or broadcastable to terms.S_voxel
    sparse_ray_indices: Tensor = None,
    sparse_obj_indices: Tensor = None,
) -> Tensor:
    """
    Compute the PPDF summand for each ray (i, q, j):

        f_{iqj} = exp(- Σ_k μ_k L_{iqjk}) * (1 - exp(- μ_i S_{iqj})) * Ω_{iqj} / (4π)

    Parameters
    ----------
    terms : RayPPDFTerms3D
        Holds L_{iqjk}, S_{iqj}, Ω_{iqj} for the rays.

    mu_objects : Tensor
        Linear attenuation coefficients μ_k for each object k.
        Shape (K,) or broadcastable to terms.L_objects.

    mu_detector : Tensor
        Linear attenuation coefficient μ_i for detector material i.
        Can be scalar, or have extra dims matching S_voxel.

    Returns
    -------
    Tensor
        f_{iqj} for each ray, shape matching terms.S_voxel (i.e. (...,)).
        To get f_ppdf(j; i) you sum over q (detector voxels).
    """
    terms = terms.to()

    # Broadcast μ_k over ray dimensions
    # μ_k * L_{iqjk} → (..., K)
    mu_objects = mu_objects.to(dtype=DTYPE, device=terms.L_objects.device)
    while mu_objects.dim() < terms.L_objects.dim():
        mu_objects = mu_objects.unsqueeze(0)

    sum_muL = (mu_objects * terms.L_objects).sum(dim=-1)  # (...,)

    mu_detector = mu_detector.to(dtype=DTYPE, device=terms.S_voxel.device)

    attenuation_external = (-sum_muL).exp()                    # exp(- Σ_k μ_k L_{iqjk})
    attenuation_voxel = 1.0 - (-mu_detector * terms.S_voxel).exp()
    angle_term = terms.solid_angle / (4.0 * pi)

    return attenuation_external * attenuation_voxel * angle_term


def ppdf_3d_local(
    fov_voxel_centers: Tensor,
    det_voxel_centers: Tensor,
    objects: dict,
    mu_objects: Tensor,
    mu_detector: Tensor,
    solid_angle: Tensor,
    detector_index: int,
    # Broad phase sparse indices
    sparse_ray_indices: Tensor = None,
    sparse_obj_indices: Tensor = None,
    # Pre-computed rays (Issue #2 fix: avoid recomputation)
    o_world: Optional[Tensor] = None,
    d_world: Optional[Tensor] = None,
) -> Tensor:
    """
    3D analogue of ppdf_2d_local with Sparse Broad-Phase Support.

    When sparse_ray_indices/sparse_obj_indices are provided along with
    o_world/d_world, uses a fully sparse path that avoids allocating
    dense (N_rays, N_obj) tensors.
    """
    device = fov_voxel_centers.device
    dtype = DTYPE

    fov_voxel_centers = fov_voxel_centers.to(device=device, dtype=dtype)
    det_voxel_centers = det_voxel_centers.to(device=device, dtype=dtype)

    N_fov = fov_voxel_centers.shape[0]
    N_det_vox = det_voxel_centers.shape[0]

    # Build rays if not pre-computed (backward compatible)
    if o_world is None or d_world is None:
        rays = rays_3d_batch(fov_voxel_centers, det_voxel_centers)
        o_world = rays[..., 0, :].reshape(-1, 3)
        e_world = rays[..., 1, :].reshape(-1, 3)
        d_world = e_world - o_world

    N_rays = o_world.shape[0]

    # =========================================================
    # SPARSE PATH: use fully sparse computation (no dense L_objects)
    # =========================================================
    if sparse_ray_indices is not None and sparse_obj_indices is not None:
        sparse_result = _compute_sparse_path_lengths(
            o_world, d_world, objects,
            sparse_ray_indices, sparse_obj_indices
        )

        # --- Compute S_voxel from sparse hits ---
        # objects is guaranteed on device by compute_system_matrix_for_detector (Issue #3)
        det_index_per_obj = objects["detector_index"]
        det_voxel_index_per_obj = objects["detector_voxel_index"]

        # Build per-ray q-index (which detector voxel each ray targets)
        q_indices = torch.arange(N_det_vox, device=device, dtype=torch.long)
        q_grid = q_indices.unsqueeze(0).expand(N_fov, -1)
        q_per_ray = q_grid.reshape(-1)  # (N_rays,)

        S_flat = torch.zeros(N_rays, dtype=dtype, device=device)

        if sparse_result.ray_indices.numel() > 0:
            hit_det_idx = det_index_per_obj[sparse_result.obj_indices]
            hit_vox_idx = det_voxel_index_per_obj[sparse_result.obj_indices]
            hit_ray_q = q_per_ray[sparse_result.ray_indices]

            is_svoxel = (
                (hit_det_idx == int(detector_index))
                & (hit_vox_idx >= 0)
                & (hit_vox_idx == hit_ray_q)
            )
            if is_svoxel.any():
                S_flat.index_add_(
                    0,
                    sparse_result.ray_indices[is_svoxel],
                    sparse_result.lengths[is_svoxel],
                )

        # --- Compute external attenuation sum(mu_k * L_k) from sparse hits ---
        sum_muL = torch.zeros(N_rays, dtype=dtype, device=device)

        if sparse_result.ray_indices.numel() > 0:
            det_mask_per_obj = (
                (det_index_per_obj == int(detector_index))
                & (det_voxel_index_per_obj >= 0)
            )
            is_external = ~det_mask_per_obj[sparse_result.obj_indices]

            if is_external.any():
                ext_ray = sparse_result.ray_indices[is_external]
                ext_obj = sparse_result.obj_indices[is_external]
                ext_L = sparse_result.lengths[is_external]
                ext_mu = mu_objects[ext_obj]  # already on device (Issue #3)
                sum_muL.index_add_(0, ext_ray, ext_mu * ext_L)

        # --- Assemble PPDF ---
        solid_angle_flat = solid_angle.to(device=device, dtype=dtype).reshape(-1)
        mu_detector = mu_detector.to(device=device, dtype=dtype)

        attenuation_external = (-sum_muL).exp()
        attenuation_voxel = 1.0 - (-mu_detector * S_flat).exp()
        angle_term = solid_angle_flat / (4.0 * pi)

        f_flat = attenuation_external * attenuation_voxel * angle_term
        f_iqj = f_flat.view(N_fov, N_det_vox).transpose(0, 1)
        return f_iqj

    # =========================================================
    # DENSE FALLBACK PATH (backward compatible)
    # =========================================================
    L_objects, _ = ray_object_path_lengths_world(
        o_world,
        d_world,
        objects,
    )

    # Compute S_{iqj}: length inside detector voxel (i, q)
    # objects already on device (Issue #3)
    det_index_per_obj = objects["detector_index"]
    det_voxel_index_per_obj = objects["detector_voxel_index"]

    det_mask = (det_index_per_obj == int(detector_index)) & (det_voxel_index_per_obj >= 0)

    if det_mask.any():
        L_det = L_objects[:, det_mask]
        voxel_ids_for_det_objs = det_voxel_index_per_obj[det_mask].long()

        q_indices = torch.arange(N_det_vox, device=device, dtype=torch.long)
        q_grid = q_indices.unsqueeze(0).expand(N_fov, -1)
        q_per_ray = q_grid.reshape(-1)

        same_voxel = (q_per_ray.view(-1, 1) == voxel_ids_for_det_objs.view(1, -1))
        S_flat = (L_det * same_voxel.to(dtype)).sum(dim=1)
    else:
        S_flat = torch.zeros(N_rays, dtype=dtype, device=device)

    # External attenuation
    if L_objects.shape[1] != mu_objects.shape[0]:
        raise ValueError("L_objects columns must match mu_objects length.")

    external_mask = ~det_mask

    if external_mask.any():
        L_external = L_objects[:, external_mask]
        mu_external = mu_objects[external_mask]
    else:
        L_external = torch.zeros((N_rays, 0), dtype=dtype, device=device)
        mu_external = mu_objects.new_zeros((0,))

    # Solid angle handling
    solid_angle = solid_angle.to(device=device, dtype=dtype)
    solid_angle_flat = solid_angle.reshape(-1)

    # Pack terms and compute
    terms = RayPPDFTerms3D(
        L_objects=L_external,
        S_voxel=S_flat,
        solid_angle=solid_angle_flat,
    )

    f_flat = ppdf_ray_factor_3d(terms, mu_external, mu_detector)

    f_iqj = f_flat.view(N_fov, N_det_vox).transpose(0, 1)
    return f_iqj
