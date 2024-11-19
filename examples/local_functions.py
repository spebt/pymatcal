import numpy as np
import math

def get_fov_voxel_center(
    ids: np.ndarray, nvx: np.ndarray, mmpvx: np.ndarray
) -> np.ndarray:
    """Vectorized version to calculate multiple voxel centers simultaneously"""
    assert np.all(ids < np.prod(nvx)), "Invalid voxel indices!"
    
    zids = ids // (nvx[0] * nvx[1])
    xyids = ids % (nvx[0] * nvx[1])
    yids = xyids // nvx[0]
    xids = xyids % nvx[0]
    
    coords = np.column_stack((xids, yids, zids))
    return (coords - np.append(nvx[:2], 0) * 0.5) * mmpvx
