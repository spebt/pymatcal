import numpy as np
from ._utils import set_module


@set_module('pymatcal')
def get_solid_angles(abpairs: np.ndarray, incs: np.ndarray) -> np.ndarray:
    """
    Calculate the solid angles of the detector units subdivisions, B, to the given set of FOV voxel subdivision centroids, A.

    :param abpairs: An array of shape (M x N, 6) containing the coordinates of FOV voxel subdivision centroids and detector unit subdivison centroids.
    :type abpairs: numpy.ndarray
    :param incs: An array of shape (3,) containing the x,y,z sizes of the detector unit subdivisions in millimeter .
    :type incs: numpy.ndarray
    :return: An array of shape (M x N,) containing the solid angles of the AB pairs.
    :rtype: np.ndarray
    """
    subAreas = np.array([incs[1] * incs[2], incs[2] * incs[0], incs[0] * incs[1]])
    ab_vec = abpairs[:, 3:] - abpairs[:, 0:3]
    ab_lens = np.linalg.norm(ab_vec, axis=1)
    norm_areas = np.abs(ab_vec * subAreas).T / ab_lens
    return np.sum(norm_areas / (ab_lens**2), axis=0)

@set_module('pymatcal')
def get_norm_areas(abpairs: np.ndarray, incs: np.ndarray) -> np.ndarray:
    """
    Calculate the areas normal to the rays defined by the AB pairs. Where,
    - B represent the detector units subdivisions
    - A reprensent the FOV voxel subdivision centroids

    :param abpairs: An array of shape (M x N, 6) containing the coordinates of FOV voxel subdivision centroids and detector unit subdivison centroids.
    :type abpairs: np.ndarray
    :param incs: An array of shape (3,) containing the x,y,z sizes of the detector unit subdivisions in millimeter .
    :type incs: np.ndarray
    :return: An array of shape (M x N,) containing the areas normal to the rays defined by the AB pairs.
    :rtype: np.ndarray
    """
    subAreas = np.array([incs[1] * incs[2], incs[2] * incs[0], incs[0] * incs[1]])
    ab_vec = abpairs[:, 3:] - abpairs[:, 0:3]
    ab_lens = np.linalg.norm(ab_vec, axis=1)
    norm_areas = np.abs(ab_vec * subAreas).T / ab_lens
    return np.sum(norm_areas)
