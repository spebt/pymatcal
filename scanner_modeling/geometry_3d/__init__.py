# pymatcal/scanner_modeling/geometry_3d/__init__.py
__all__ = [
    "fov_tensor_dict_3d",
    "fov_corners_vertices_3d",
    "voxels_coordinates_3d",
    "ConvexPolyhedronBatch",
    "obb_planes_from_center_extents",
    "local_to_world_points",
    "world_to_local_points",
    "local_to_world_dirs",
    "world_to_local_dirs",
    "world_to_local_rays",
    "load_scanner_geometry_3d_from_layout",
    "OBJECT_TYPE_OBB",
    "OBJECT_TYPE_CONVEX_POLY",
    "rotation_from_azimuth_tilt",
    "apply_transform",
    "build_convex_union_convex_poly_block",
    "world_to_local_ray",
    "object_to_world_aabb",
    "solid_angle_triangle",
    "solid_angle_triangles_batch",
    "solid_angle_detector_voxel",
    "reorder_detector_voxel_faces_front_first", # <--- Added
]

from .._geometry_3d._utils_3d import (
    fov_tensor_dict_3d,
    fov_corners_vertices_3d,
    voxels_coordinates_3d,
    rotation_from_azimuth_tilt,
    apply_transform,
    reorder_detector_voxel_faces_front_first, # <--- Added
)

from .._geometry_3d._polyhedron import (
    ConvexPolyhedronBatch,
    obb_planes_from_center_extents,
    local_to_world_points,
    world_to_local_points,
    local_to_world_dirs,
    world_to_local_dirs,
    world_to_local_rays,
    world_to_local_ray,
)

from .._geometry_3d._io import (
    load_scanner_geometry_3d_from_layout,
    OBJECT_TYPE_OBB,
    OBJECT_TYPE_CONVEX_POLY,
    build_convex_union_convex_poly_block,
)

from .._geometry_3d._aabb_3d import object_to_world_aabb

from .._geometry_3d._solid_angle_3d import (
    solid_angle_triangle,
    solid_angle_triangles_batch,
    solid_angle_detector_voxel,
)