import numpy as np


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
    conditions = np.swapaxes(np.array([condition_x, condition_y, condition_z]), 0, 1)
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


# def get_intersections(geoms,pAs,pBs):
#     # Case 1: intersects on face x = x_0 or face x = x_1
#     # Note that A_x never equals B_x.
#     tx = (geoms[:, 0:2] - pA[0]) / (pB[0] - pA[0])
#     yx = tx * (pB[1] - pA[1]) + pA[1]
#     zx = tx * (pB[2] - pA[2]) + pA[2]
#     # Case 2: intersects on face y = y_0 or face y = y_1
#     # Note: we exclude the case when A_y equals B_y
#     if pB[1] - pA[1] != 0:
#         ty = (geomsOnPath[:, 2:4] - pA[1]) / (pB[1] - pA[1])
#         xy = ty * (pB[0] - pA[0]) + pA[0]
#         zy = ty * (pB[2] - pA[2]) + pA[2]


#     # Case 3: intersects on face z = z_0 or face z = z_1
#     # Note: we exclude the case when A_z equals B_z
#     if pB[2] - pA[2] != 0:
#         tz = (geomsOnPath[:, 4:6] - pA[2]) / (pB[2] - pA[2])
#         xz = tz * (pB[0] - pA[0]) + pA[0]
#         yz = tz * (pB[1] - pA[1]) + pA[1]


def get_AB_pairs(pAs, pBs):
    na, nb = (len(pAs), len(pBs))
    return np.concatenate(
        (np.repeat(pAs, nb, axis=0), np.tile(pBs, (na, 1))), axis=1
    )


def get_intersections_2d(geoms, pAs, pBs):
    nb = len(pAs)
    abpairs = get_AB_pairs(pAs, pBs)
    for geom in geoms:
        o = 0.5 * (geom[0:-2:2] + geom[1:-2:2])
        diagLength = np.linalg.norm(geom[1:-2:2] - geom[0:-2:2])*0.5
        print(np.cross(pAs[:,]-o,pBs[:,]-o).shape)
    # # Loop through subdivisions
    # for pA in pAs:
    #     oa = pA - o
    #     for pB in pBs:
    #         ob = pB - o
    #         ab_length = np.linalg.norm(pB - pA)
    #         dist = np.linalg.norm(np.cross(oa, ob), axis=1) / ab_length

    # # Line AB will not intersect with z planes in the 2D case

    # # Geometries on the gamma ray path
    # geomsOnPath = geoms[np.where(dist < diagLength, True, False)]
    # nGeoms = geomsOnPath.shape[0]
    # if nGeoms > 0:
    #     geomsOnPath = np.append(geomsOnPath, sensDetSubGeom).reshape(nGeoms + 1, 8)
    # else:
    #     geomsOnPath = np.array([sensDetSubGeom])

    # # Case 1: intersects on face x = x_0 or face x = x_1
    # # Note that A_x never equals B_x.
    # tx = (geoms[:, 0:2] - pA[0]) / (pB[0] - pA[0])
    # yx = tx * (pB[1] - pA[1]) + pA[1]
    # zx = tx * (pB[2] - pA[2]) + pA[2]
    # # Case 2: intersects on face y = y_0 or face y = y_1
    # # Note: we exclude the case when A_y equals B_y
    # if pB[1] - pA[1] != 0:
    #     ty = (geomsOnPath[:, 2:4] - pA[1]) / (pB[1] - pA[1])
    #     xy = ty * (pB[0] - pA[0]) + pA[0]
    #     zy = ty * (pB[2] - pA[2]) + pA[2]
