import numpy as np
from .coord_transform import *
from ._utils import set_module


@set_module('pymatcal')
def get_img_subdivs(mmpvx, nsubs):
    """
    Get the subdivisions of an image.

    :param mmpvx: The maximum values of the x, y, and z coordinates.
    :type mmpvx: array-like
    :param nsubs: The number of subdivisions in the x, y, and z directions.
    :type nsubs: array-like
    :return: A dictionary containing the coordinates and increments of the subdivisions.
    :rtype: dict
    """
    xlin = np.linspace(
        0, mmpvx[0], int(nsubs[0]) + 1
    )
    ylin = np.linspace(
        0, mmpvx[1], int(nsubs[1]) + 1
    )
    zlin = np.linspace(
        0, mmpvx[2], int(nsubs[2]) + 1
    )
    return {"coords": np.array(
        np.meshgrid(
            0.5 * (xlin[1:] + xlin[:-1]),
            0.5 * (ylin[1:] + ylin[:-1]),
            0.5 * (zlin[1:] + zlin[:-1]),
        )
    ).T.reshape(int(nsubs.prod()), 3)-mmpvx*0.5,
        "incs": mmpvx/nsubs
    }

@set_module('pymatcal')
def get_det_subdivs(focs: np.ndarray, nsubs: np.ndarray):
    """
    Get the subdivisions of a detector.

    :param focs: The focal points of the detector.
    :type focs: ndarray
    :param nsubs: The number of subdivisions in the x, y, and z directions.
    :type nsubs: ndarray
    :return: A dictionary containing the subdivisions and increments.
    :rtype: dict
        - geoms (ndarray): The subdivisions of the detector.
        - incs (ndarray): The increments of the subdivisions.
    """
    nsubs_prod = np.prod(nsubs)
    focs_incrs = (focs[1:6:2] - focs[:6:2])/nsubs
    incr_map = np.indices(np.flip(nsubs)+1)
    map1 = np.flip(incr_map[:, :nsubs[2], :nsubs[1],
                            :nsubs[0]].reshape((3, nsubs_prod)).T, axis=1)*focs_incrs
    map2 = np.flip(incr_map[:, 1:nsubs[2]+1, 1:nsubs[1]+1,
                            1:nsubs[0]+1].reshape((3, nsubs_prod)).T, axis=1)*focs_incrs
    new_seq = np.array([0, 3, 1, 4, 2, 5])
    out = np.hstack((map1, map2, np.zeros(
        (nsubs_prod, 1)), np.zeros((nsubs_prod, 1))))
    return {'geoms': np.hstack((out[:, new_seq], np.zeros(
        (nsubs_prod, 1)), np.zeros(
        (nsubs_prod, 1))))+focs[np.array([0, 0, 2, 2, 4, 4, 6, 7])],
        'incs': focs_incrs
    }

@set_module('pymatcal')
def get_centroids(geoms: np.ndarray) -> np.ndarray:
    """
    Get the centroids of the subdivisions.

    :param geoms: The subdivisions of the detector.
    :type geoms: ndarray
    :return: The centroids of the subdivisions.
    :rtype: ndarray
    """
    return (geoms[:, :6:2]+geoms[:, 1:6:2])*0.5

@set_module('pymatcal')
def append_subdivs(geoms: np.ndarray, focs: np.ndarray, subdiv_geoms: np.ndarray):
    """
    Append subdivisions to the existing array of geometries.

    :param geoms: The existing subdivisions.
    :type geoms: ndarray
    :param focs: The detector units geometries in focus.
    :type focs: ndarray
    :param subdiv_geoms: The subdivisions geometries to be appended.
    :type subdiv_geoms: ndarray
    :return: The updated array of geometries.
    :rtype: ndarray
    """
    blocks = geoms[geoms[:, 6] != focs[6]]
    return np.vstack((blocks, subdiv_geoms))
