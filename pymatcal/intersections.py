import numpy as np

# findt is depreciated, DO NOT USE


def findt(geomsOnPath, pA, pB):
    nGeoms = geomsOnPath.shape[0]
    ty = np.zeros((nGeoms, 2))
    tz = np.zeros((nGeoms, 2))
    condition_y = np.full((nGeoms, 2), False)
    condition_z = np.full((nGeoms, 2), False)
    # Case 1: intersects on face x = x_0 or face x = x_1
    # Note that A_x never equals B_x.
    tx = (geomsOnPath[:, 0:2] - pA[0]) / (pB[0] - pA[0])
    yx = tx * (pB[1] - pA[1]) + pA[1]
    zx = tx * (pB[2] - pA[2]) + pA[2]
    condition_x = np.where(
        (yx - np.vstack((geomsOnPath[:, 2], geomsOnPath[:, 2])).T)
        * (yx - np.vstack((geomsOnPath[:, 3], geomsOnPath[:, 3])).T)
        <= 1e-10,
        True,
        False,
    ) * np.where(
        (zx - np.vstack((geomsOnPath[:, 4], geomsOnPath[:, 4])).T)
        * (zx - np.vstack((geomsOnPath[:, 5], geomsOnPath[:, 5])).T)
        <= 1e-10,
        True,
        False,
    )
    # Case 2: intersects on face y = y_0 or face y = y_1
    # Note: we exclude the case when A_y equals B_y
    if pB[1] - pA[1] != 0:
        ty = (geomsOnPath[:, 2:4] - pA[1]) / (pB[1] - pA[1])
        xy = ty * (pB[0] - pA[0]) + pA[0]
        zy = ty * (pB[2] - pA[2]) + pA[2]
        condition_y = np.where(
            (xy - np.vstack((geomsOnPath[:, 0], geomsOnPath[:, 0])).T)
            * (xy - np.vstack((geomsOnPath[:, 1], geomsOnPath[:, 1])).T)
            <= 1e-10,
            True,
            False,
        ) * np.where(
            (zy - np.vstack((geomsOnPath[:, 4], geomsOnPath[:, 4])).T)
            * (zy - np.vstack((geomsOnPath[:, 5], geomsOnPath[:, 5])).T)
            <= 1e-10,
            True,
            False,
        )

    # Case 3: intersects on face z = z_0 or face z = z_1
    # Note: we exclude the case when A_z equals B_z
    if pB[2] - pA[2] != 0:
        tz = (geomsOnPath[:, 4:6] - pA[2]) / (pB[2] - pA[2])
        xz = tz * (pB[0] - pA[0]) + pA[0]
        yz = tz * (pB[1] - pA[1]) + pA[1]
        condition_z = np.where(
            (xz - np.vstack((geomsOnPath[:, 0], geomsOnPath[:, 0])).T)
            * (xz - np.vstack((geomsOnPath[:, 1], geomsOnPath[:, 1])).T)
            <= 1e-10,
            True,
            False,
        ) * np.where(
            (yz - np.vstack((geomsOnPath[:, 2], geomsOnPath[:, 2])).T)
            * (yz - np.vstack((geomsOnPath[:, 3], geomsOnPath[:, 3])).T)
            <= 1e-10,
            True,
            False,
        )
    conditions = np.swapaxes(
        np.array([condition_x, condition_y, condition_z]), 0, 1)
    tArr = np.swapaxes(np.array([tx, ty, tz]), 0, 1)
    dlmulist = []
    ab_length = np.linalg.norm(pB - pA)
    for id in range(0, nGeoms - 1):
        t = np.unique(tArr[id][conditions[id]])
        t = t[np.nonzero(t < 1)]
        if t.size > 1:
            dl = np.abs(t[0] - t[1]) * ab_length
            dlmulist.append(dl * geomsOnPath[id, 7])
    t = np.unique(tArr[-1][conditions[-1]])
    try:
        dl = np.abs(t[0] - t[1]) * ab_length
        dlmulist.append(dl * geomsOnPath[-1, 7])
    except Exception as err:
        print("t: ", t, "\npA: ", pA, "\npB: ", pB, "\nGeom:", geomsOnPath[-1])
    return np.array(dlmulist)


def get_AB_pairs(pAs, pBs):
    na, nb = (len(pAs), len(pBs))
    return np.concatenate((np.repeat(pAs, nb, axis=0), np.tile(pBs, (na, 1))), axis=1)


def findt_2d(geom, abpairs):
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


def get_intersects_2d(geoms: np.ndarray, abpairs: np.ndarray):
    # abpairs = get_AB_pairs(pAs, pBs)
    ab_vec = abpairs[:, 3:] - abpairs[:, 0:3]
    ts = np.array([findt_2d(geom, abpairs) for geom in geoms])
    intersects = np.array(
        [
            [ab_vec.T * ts[iGeom, :, idx] + abpairs[:, 0:3].T for idx in [0, 1]]
            for iGeom in range(0, geoms.shape[0])
        ]
    )
    return np.swapaxes(intersects, 3, 1)[np.any(ts != 0, axis=2)]


def get_intersections_2d(geoms: np.ndarray, abpairs: np.ndarray):
    # abpairs = get_AB_pairs(pAs, pBs)
    ab_vec = abpairs[:, 3:] - abpairs[:, 0:3]
    ab_length = np.linalg.norm(ab_vec, axis=1)
    ts = np.array([findt_2d(geom, abpairs) for geom in geoms])
    return {'intersections': ab_length.T * (ts[:, :, 1] - ts[:, :, 0]),
            'ts': ts}
