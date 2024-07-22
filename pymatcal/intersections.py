import numpy as np
from ._utils import set_module


@set_module('pymatcal')
def get_AB_pairs(pAs, pBs):
    """
    Generate pairs of points from two arrays of points.

    :param pAs: Array of points A.
    :type pAs: numpy.ndarray
    :param pBs: Array of points B.
    :type pBs: numpy.ndarray
    :return: Array of pairs of points, where each row contains the coordinates of a pair (Ax, Ay, Az, Bx, By, Bz).
    :rtype: numpy.ndarray
    """
    na, nb = (len(pAs), len(pBs))
    return np.concatenate((np.repeat(pAs, nb, axis=0), np.tile(pBs, (na, 1))), axis=1)


@set_module('pymatcal')
def findt_2d(geom, abpairs):
    """
    Find the intersection points between a geometry and a set of line segments in 2D.

    :param geoms: Array of geometries, where each geometry is represented as [x0, x1, y0, y1,z0, z1, sequence, mu].
    :type geoms: numpy.ndarray
    :param abpairs: Array of rays, where each ray is represented as [A_x, A_y, A_z, B_x, B_y, B_z].
    :return: A 2D array containing the intersection points [t0, t1].
    :rtype: numpy.ndarray
    """

    # Case 1: intersects on face x = x_0 or face x = x_1
    # Note that A_x never equals B_x.
    tx0 = (abpairs[:, 0] - geom[0]) / (abpairs[:, 0] - abpairs[:, 3])
    tx1 = (abpairs[:, 0] - geom[1]) / (abpairs[:, 0] - abpairs[:, 3])
    yx0 = tx0 * (abpairs[:, 4] - abpairs[:, 1]) + abpairs[:, 1]
    yx1 = tx1 * (abpairs[:, 4] - abpairs[:, 1]) + abpairs[:, 1]
    condition_x0 = np.all(
        np.array([yx0 > geom[2], yx0 < geom[3], tx0 > 0, tx0 < 1]), axis=0
    )
    condition_x1 = np.all(
        np.array([yx1 > geom[2], yx1 < geom[3], tx1 > 0, tx1 < 1]), axis=0
    )
    # Case 2: intersects on face y = y_0 or face y = y_1
    # Note: we exclude the case when A_y equals B_y
    abpairs_y = abpairs[abpairs[:, 1] != abpairs[:, 4]]
    ty0 = (abpairs_y[:, 1] - geom[2]) / (abpairs_y[:, 1] - abpairs_y[:, 4])
    ty1 = (abpairs_y[:, 1] - geom[3]) / (abpairs_y[:, 1] - abpairs_y[:, 4])
    xy0 = ty0 * (abpairs_y[:, 3] - abpairs_y[:, 0]) + abpairs_y[:, 0]
    xy1 = ty1 * (abpairs_y[:, 3] - abpairs_y[:, 0]) + abpairs_y[:, 0]
    condition_y0 = np.all(
        np.array([xy0 > geom[0], xy0 < geom[1], ty0 > 0, ty0 < 1]), axis=0
    )
    condition_y1 = np.all(
        np.array([xy1 > geom[0], xy1 < geom[1], ty1 > 0, ty1 < 1]), axis=0
    )
    t_sorted = np.sort(
        np.array(
            [
                np.where(condition_x0, tx0, 0),
                np.where(condition_x1, tx1, 0),
                np.where(condition_y0, ty0, 0),
                np.where(condition_y1, ty1, 0),
            ]
        ).T,
        axis=1,
    )
    # To deal with duplicated intersections and single intersections
    t_count = np.count_nonzero(t_sorted, axis=1)
    t_list = np.append(t_sorted, np.ones((abpairs.shape[0], 1)), axis=1)
    mask = np.array(
        [
            [True, True, False, False, False],
            [False, False, False, True, True],
            [False, False, True, True, False],
            [False, True, False, True, False],
            [True, False, False, True, False],
        ]
    )
    t_mask = mask[t_count]
    return t_list[t_mask].reshape((abpairs.shape[0], 2))


@set_module('pymatcal')
def get_intersects_2d(geoms: np.ndarray, abpairs: np.ndarray) -> np.ndarray:
    """
    Calculate the intersection points between a set of geometries and given set of rays in 2D.

    :param geoms: Array of geometries, where each geometry is represented as [x0, x1, y0, y1,z0, z1, sequence, mu].
    :type geoms: numpy.ndarray
    :param abpairs: Array of rays, where each ray is represented as [Ax, Ay, Az, Bx, By, Bz].
    :type abpairs: numpy.ndarray
    :return: Array of intersection points, where each intersection point is represented as [x, y, z].
    :rtype: numpy.ndarray
    """
    ab_vec = abpairs[:, 3:] - abpairs[:, 0:3]
    ts = np.array([findt_2d(geom, abpairs) for geom in geoms])
    intersects = np.array(
        [
            [ab_vec.T * ts[iGeom, :, idx] + abpairs[:, 0:3].T for idx in [0, 1]]
            for iGeom in range(0, geoms.shape[0])
        ]
    )
    return np.swapaxes(intersects, 3, 1)[np.any(ts != 0, axis=2)]


@set_module('pymatcal')
def get_intersections_2d(geoms: np.ndarray, abpairs: np.ndarray):
    """
    Calculate the intersection length between a set of geometries and given set of rays in 2D.

    :param geoms: Array of geometries, where each geometry is represented as [x0, x1, y0, y1,z0, z1, sequence, mu].
    :type geoms: numpy.ndarray
    :param abpairs: Array of rays, where each ray is represented as [Ax, Ay, Az, Bx, By, Bz].
    :type abpairs: numpy.ndarray
    :return: Dictionary containing the intersection lengths and t values.
        - 'intersections': Array of intersection points, where elements are intersection lengths.
        - 'ts': Array of t values, where each t value represents the parameterization of the intersection point along the rays.
    :rtype: dict
    """
    ab_vec = abpairs[:, 3:] - abpairs[:, 0:3]
    ab_length = np.linalg.norm(ab_vec, axis=1)
    ts = np.array([findt_2d(geom, abpairs) for geom in geoms])
    return {'intersections': ab_length.T * (ts[:, :, 1] - ts[:, :, 0]),
            'ts': ts}
