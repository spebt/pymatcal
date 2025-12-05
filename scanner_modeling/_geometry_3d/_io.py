# pymatcal/scanner_modeling/_geometry_3d/_io.py

from typing import Dict, Tuple, Optional

import torch
from torch import Tensor

from .._config import DTYPE


# Integer codes for object types (to avoid string comparisons in hot loops)
OBJECT_TYPE_OBB = 0
OBJECT_TYPE_CONVEX_POLY = 1


def _get_layout_entry(
    layout_idx: int,
    scanner_layouts_data: Dict,
) -> Dict:
    """
    Internal helper: fetch per-position entry from scanner_layouts_data.

    We mirror the 2D convention where layouts are stored under keys like
    "position 000", "position 001", ...

    Parameters
    ----------
    layout_idx : int
        Index of the desired scanner position.
    scanner_layouts_data : Dict
        Dictionary-like layout container. The expected key is
        f"position {layout_idx:03d}".

    Returns
    -------
    Dict
        Layout dictionary for the requested position.
    """
    key = f"position {layout_idx:03d}"
    if key not in scanner_layouts_data:
        raise KeyError(f"Layout key '{key}' not found in scanner_layouts_data.")
    return scanner_layouts_data[key]

def build_convex_union_convex_poly_block(
    centers: Tensor,
    rotations: Tensor,
    plane_normals_local: Tensor,
    plane_offsets_local: Tensor,
    material_index,
    detector_index=-1,
    detector_voxel_index=-1,
    num_planes: Tensor | None = None,
) -> Dict[str, Tensor]:
    """
    Helper to represent a complex component as a finite union of convex
    polyhedral cells for use in the 3D scanner layout.

    Each convex cell is represented via its own (center, rotation,
    plane_normals_local, plane_offsets_local). All cells share the same
    material_index and detector metadata, which can be passed as scalars
    or per-cell integer arrays.

    Parameters
    ----------
    centers : Tensor, shape (N_cells, 3)
        World-space centers of each convex cell, in mm.

    rotations : Tensor, shape (N_cells, 3, 3)
        Rotation matrices R_world_from_local for each cell.

    plane_normals_local : Tensor, shape (N_cells, P_max, 3)
        Local-space plane normals for each cell.

    plane_offsets_local : Tensor, shape (N_cells, P_max)
        Local-space plane offsets d_k in the plane inequality
        n_k · x <= d_k.

    material_index : int or Tensor
        Material index (μ_k) for the entire component. If scalar, it is
        broadcast to all cells. If Tensor, must have shape (N_cells,).

    detector_index : int or Tensor, optional
        Detector index i for all cells, or -1 if not a detector.

    detector_voxel_index : int or Tensor, optional
        Detector voxel index q for all cells, or -1 if not applicable.

    num_planes : Tensor, shape (N_cells,), optional
        Number of active planes per cell. If omitted, all planes along
        axis 1 of plane_normals_local / plane_offsets_local are assumed
        to be active.

    Returns
    -------
    convex_poly_block : Dict[str, Tensor]
        Dictionary suitable for use as the "convex_poly" entry in a
        scanner_layouts_data["position XXX"] layout, as consumed by
        load_scanner_geometry_3d_from_layout.
    """
    device = centers.device
    dtype = DTYPE

    centers = centers.to(dtype=dtype, device=device)
    rotations = rotations.to(dtype=dtype, device=device)
    plane_normals_local = plane_normals_local.to(dtype=dtype, device=device)
    plane_offsets_local = plane_offsets_local.to(dtype=dtype, device=device)

    N_cells = centers.shape[0]

    if rotations.shape != (N_cells, 3, 3):
        raise ValueError(
            f"rotations must have shape (N_cells, 3, 3), got {tuple(rotations.shape)}"
        )
    if plane_normals_local.shape[0] != N_cells or plane_offsets_local.shape[0] != N_cells:
        raise ValueError(
            "plane_normals_local and plane_offsets_local must have "
            "first dimension equal to N_cells"
        )

    def _as_int_tensor(x, default_value: int) -> Tensor:
        if x is None:
            x = default_value
        if not isinstance(x, Tensor):
            x = torch.as_tensor(x, dtype=torch.int64, device=device)
        else:
            x = x.to(dtype=torch.int64, device=device)

        if x.ndim == 0:
            x = x.expand(N_cells)
        elif x.shape == (N_cells,):
            # already per-cell
            pass
        else:
            raise ValueError(
                f"metadata tensor must be scalar or shape (N_cells,), got {tuple(x.shape)}"
            )
        return x

    material_index = _as_int_tensor(material_index, default_value=0)
    detector_index = _as_int_tensor(detector_index, default_value=-1)
    detector_voxel_index = _as_int_tensor(detector_voxel_index, default_value=-1)

    if num_planes is not None:
        if not isinstance(num_planes, Tensor):
            num_planes = torch.as_tensor(num_planes, dtype=torch.int64, device=device)
        else:
            num_planes = num_planes.to(dtype=torch.int64, device=device)

        if num_planes.ndim == 0:
            num_planes = num_planes.expand(N_cells)
        elif num_planes.shape != (N_cells,):
            raise ValueError(
                f"num_planes must be scalar or shape (N_cells,), got {tuple(num_planes.shape)}"
            )

    convex_poly_block: Dict[str, Tensor] = {
        "centers": centers,
        "rotations": rotations,
        "plane_normals_local": plane_normals_local,
        "plane_offsets_local": plane_offsets_local,
        "material_index": material_index,
        "detector_index": detector_index,
        "detector_voxel_index": detector_voxel_index,
    }
    if num_planes is not None:
        convex_poly_block["num_planes"] = num_planes

    return convex_poly_block


def load_scanner_geometry_3d_from_layout(
    layout_idx: int,
    scanner_layouts_data: Dict,
) -> Dict[str, Tensor]:
    """
    Load 3D scanner geometry for a given layout and assemble a unified
    object dictionary.

    This is the 3D analogue of `load_scanner_geometry_from_layout` in 2D,
    but here we work with volumetric objects (voxels, plates, etc.) and
    represent them as geometric primitives.

    Expected structure of `scanner_layouts_data`
    --------------------------------------------
    We assume that `scanner_layouts_data` contains entries keyed by
    "position {layout_idx:03d}", each of which is a dict with (optional)
    sub-dicts `"obb"` and `"convex_poly"`. For example:

        scanner_layouts_data["position 000"] = {
            "obb": {
                "centers": Tensor (N_obb, 3),
                "rotations": Tensor (N_obb, 3, 3),
                "half_sizes": Tensor (N_obb, 3),
                "material_index": Tensor (N_obb,),
                "detector_index": Tensor (N_obb,),
                "detector_voxel_index": Tensor (N_obb,),
            },
            "convex_poly": {
                "centers": Tensor (N_poly, 3),
                "rotations": Tensor (N_poly, 3, 3),
                "plane_normals_local": Tensor (N_poly, P_max, 3),
                "plane_offsets_local": Tensor (N_poly, P_max),
                # optional; if omitted, all planes are used
                "num_planes": Tensor (N_poly,),  # integer-valued
                "material_index": Tensor (N_poly,),
                "detector_index": Tensor (N_poly,),
                "detector_voxel_index": Tensor (N_poly,),
            },
        }

    All distances must be in millimetres (DISTANCE_UNIT) and will be cast
    to the global DTYPE.

    Returns
    -------
    objects : Dict[str, Tensor]
        Unified object description. Keys:

        Core per-object arrays (N_obj, ...):
            - "centers"              : (N_obj, 3)
            - "rotations"            : (N_obj, 3, 3)
            - "material_index"       : (N_obj,)
            - "detector_index"       : (N_obj,)
            - "detector_voxel_index" : (N_obj,)
            - "type"                 : (N_obj,), int codes
                                      OBJECT_TYPE_OBB or OBJECT_TYPE_CONVEX_POLY

        Type-specific data and index mappings:
            - "obb_object_ids"       : (N_obb,) indices into the core arrays
            - "half_sizes"           : (N_obb, 3) half-lengths along local axes

            - "poly_object_ids"      : (N_poly,) indices into the core arrays
            - "plane_normals_local"  : (N_poly, P_max, 3)
            - "plane_offsets_local"  : (N_poly, P_max)
            - "num_planes"           : (N_poly,) or None if not provided

        Type code constants are also returned for convenience:
            - "OBJECT_TYPE_OBB"
            - "OBJECT_TYPE_CONVEX_POLY"

    Notes
    -----
    This function does *not* generate geometry from 2D layouts or from
    high-level scanner configuration; it only concatenates already-
    constructed 3D primitives into a unified structure for downstream
    raytracing / PPDF code.
    """

    layout = _get_layout_entry(layout_idx, scanner_layouts_data)

    obb = layout.get("obb", None)
    poly = layout.get("convex_poly", None)

    # ----- Infer counts -----
    N_obb = 0
    N_poly = 0

    if obb is not None and "centers" in obb and obb["centers"] is not None:
        N_obb = int(obb["centers"].shape[0])

    if poly is not None and "centers" in poly and poly["centers"] is not None:
        N_poly = int(poly["centers"].shape[0])

    N_obj = N_obb + N_poly

    if N_obj == 0:
        # No objects: return an empty, but well-formed dictionary
        empty = torch.empty(0, dtype=DTYPE)
        return {
            "centers": empty.view(0, 3),
            "rotations": empty.view(0, 3, 3),
            "material_index": empty,
            "detector_index": empty,
            "detector_voxel_index": empty,
            "type": empty.to(dtype=torch.int64),
            "OBJECT_TYPE_OBB": OBJECT_TYPE_OBB,
            "OBJECT_TYPE_CONVEX_POLY": OBJECT_TYPE_CONVEX_POLY,
        }

    # ----- Allocate unified arrays (on the same device as the first present block) -----
    # Prefer OBB device, then poly device as fallback.
    device: Optional[torch.device] = None
    if N_obb > 0:
        device = obb["centers"].device
    elif N_poly > 0:
        device = poly["centers"].device

    centers = torch.empty((N_obj, 3), dtype=DTYPE, device=device)
    rotations = torch.empty((N_obj, 3, 3), dtype=DTYPE, device=device)
    material_index = torch.empty((N_obj,), dtype=torch.int64, device=device)
    detector_index = torch.empty((N_obj,), dtype=torch.int64, device=device)
    detector_voxel_index = torch.empty((N_obj,), dtype=torch.int64, device=device)
    type_codes = torch.empty((N_obj,), dtype=torch.int64, device=device)

    objects: Dict[str, Tensor] = {}

    # ----- Fill OBB objects -----
    obb_object_ids: Optional[Tensor] = None
    half_sizes: Optional[Tensor] = None

    if N_obb > 0:
        start = 0
        end = N_obb
        obb_object_ids = torch.arange(start, end, dtype=torch.int64, device=device)

        centers[start:end] = obb["centers"].to(dtype=DTYPE, device=device)
        rotations[start:end] = obb["rotations"].to(dtype=DTYPE, device=device)
        half_sizes = obb["half_sizes"].to(dtype=DTYPE, device=device)

        # Metadata
        material_index[start:end] = obb["material_index"].to(
            dtype=torch.int64, device=device
        )
        detector_index[start:end] = obb["detector_index"].to(
            dtype=torch.int64, device=device
        )
        detector_voxel_index[start:end] = obb["detector_voxel_index"].to(
            dtype=torch.int64, device=device
        )

        type_codes[start:end] = OBJECT_TYPE_OBB

        objects["obb_object_ids"] = obb_object_ids
        objects["half_sizes"] = half_sizes

    # ----- Fill convex polyhedron objects -----
    poly_object_ids: Optional[Tensor] = None
    plane_normals_local: Optional[Tensor] = None
    plane_offsets_local: Optional[Tensor] = None
    num_planes: Optional[Tensor] = None

    if N_poly > 0:
        start = N_obb
        end = N_obb + N_poly
        poly_object_ids = torch.arange(start, end, dtype=torch.int64, device=device)

        centers[start:end] = poly["centers"].to(dtype=DTYPE, device=device)
        rotations[start:end] = poly["rotations"].to(dtype=DTYPE, device=device)

        plane_normals_local = poly["plane_normals_local"].to(
            dtype=DTYPE, device=device
        )
        plane_offsets_local = poly["plane_offsets_local"].to(
            dtype=DTYPE, device=device
        )
        num_planes = poly.get("num_planes", None)
        if num_planes is not None:
            num_planes = num_planes.to(dtype=torch.int64, device=device)

        # Metadata
        material_index[start:end] = poly["material_index"].to(
            dtype=torch.int64, device=device
        )
        detector_index[start:end] = poly["detector_index"].to(
            dtype=torch.int64, device=device
        )
        detector_voxel_index[start:end] = poly["detector_voxel_index"].to(
            dtype=torch.int64, device=device
        )

        type_codes[start:end] = OBJECT_TYPE_CONVEX_POLY

        objects["poly_object_ids"] = poly_object_ids
        objects["plane_normals_local"] = plane_normals_local
        objects["plane_offsets_local"] = plane_offsets_local
        if num_planes is not None:
            objects["num_planes"] = num_planes

    # ----- Attach core arrays and type codes -----
    objects["centers"] = centers
    objects["rotations"] = rotations
    objects["material_index"] = material_index
    objects["detector_index"] = detector_index
    objects["detector_voxel_index"] = detector_voxel_index
    objects["type"] = type_codes

    # For convenience, return type codes as part of the dict
    objects["OBJECT_TYPE_OBB"] = torch.tensor(
        OBJECT_TYPE_OBB, dtype=torch.int64, device=device
    )
    objects["OBJECT_TYPE_CONVEX_POLY"] = torch.tensor(
        OBJECT_TYPE_CONVEX_POLY, dtype=torch.int64, device=device
    )

    return objects
