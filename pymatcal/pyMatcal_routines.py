import numpy as np
from scipy.signal import find_peaks
from ctypes import *

libray_cuboid_intersection = CDLL("libray_cuboid_intersection.so")
get_intersection = libray_cuboid_intersection.get_intersection
# define prototypes
ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
get_intersection.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1]
get_intersection.restype = c_float





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


# Return the geometries on the path of the gamma ray
def PPFDForPair(geoms, pAs, pBs, subInc, muDet):
    # O is the center of the cuboid
    o = 0.5 * (geoms[:, 0:-2:2] + geoms[:, 1:-2:2])
    edges = geoms[:, 1:-2:2] - geoms[:, 0:-2:2]
    diagLength = np.linalg.norm(0.5 * edges, axis=1)
    halfSubIncs = 0.5 * subInc
    result = 0

    # Loop through subdivisions
    for pA in pAs:
        oa = pA - o
        for pB in pBs:
            ob = pB - o
            ab_length = np.linalg.norm(pB - pA)
            dist = np.linalg.norm(np.cross(oa, ob), axis=1) / ab_length

            # target sensitive detector subdivision
            sensDetSubGeom = np.array(
                [
                    pB[0] - halfSubIncs[0],
                    pB[0] + halfSubIncs[0],
                    pB[1] - halfSubIncs[1],
                    pB[1] + halfSubIncs[1],
                    pB[2] - halfSubIncs[2],
                    pB[2] + halfSubIncs[2],
                    -1,
                    muDet,
                ]
            )

            # Geometries on the gamma ray path
            geomsOnPath = geoms[np.where(dist < diagLength, True, False)]
            nGeoms = geomsOnPath.shape[0]
            if nGeoms > 0:
                geomsOnPath = np.append(geomsOnPath, sensDetSubGeom).reshape(
                    nGeoms + 1, 8
                )
            else:
                geomsOnPath = np.array([sensDetSubGeom])
            # print(sensDetSubGeom.shape)

            dlmu = findt(geomsOnPath, pA, pB)
            try:
                attenuationTerm = np.exp(np.sum(-dlmu[:-1]))
                absorptionTerm = 1 - np.exp(-dlmu[-1])
            except Exception as err:
                print(err)

            sensDetSubAreas = np.array(
                [
                    subInc[1] * subInc[2],
                    subInc[0] * subInc[2],
                    subInc[0] * subInc[1],
                ]
            )
            solidAngleTerm = np.sum(np.abs(pB - pA) * sensDetSubAreas) / (
                ab_length**3 * 4.0 * np.pi
            )

            # print("Attenuation Term: ",attenuationTerm)
            # print("Absorption  Term: ",absorptionTerm)
            # print("Solid Angle Term: ",solidAngleTerm)
            # print("Product:          ",attenuationTerm*absorptionTerm*solidAngleTerm)
            result += attenuationTerm * absorptionTerm * solidAngleTerm
            # result += absorptionTerm * solidAngleTerm
    result = result / pAs.shape[0]
    # if result > 0.0001:
    # print(result)
    return result


def multiplexingIndex(matxymap, target_c, image_c, imageVxpms, imageNx, imageNy):
    n_angles = 1000
    hist_arr = []
    pre_x_idx = 0
    pre_y_idx = 0
    thetas = []
    start_angle = 2.4
    end_angle = 4.5
    radius_det = np.linalg.norm(target_c - image_c)
    for theta in np.arange(start_angle, end_angle, np.pi / n_angles):
        y_cord = radius_det * np.sin(theta) + target_c[1]
        x_cord = radius_det * np.cos(theta) + target_c[0]
        x_idx = int(np.floor(x_cord * imageVxpms[0]))
        y_idx = int(np.floor(y_cord * imageVxpms[1]))

        if x_idx > 0 and x_idx < imageNx and y_idx > 0 and y_idx < imageNy:
            if pre_x_idx != x_idx or pre_y_idx != y_idx:
                hist_arr.append(matxymap[x_idx, y_idx])
                thetas.append(theta)
        pre_x_idx = x_idx
        pre_y_idx = y_idx
    signal = np.asarray(hist_arr)
    thetas = np.asarray(thetas)
    maximum60p = max(signal) * 0.6
    # print(signal)
    peaks = find_peaks(signal, height=0, prominence=maximum60p)[0]
    result = 0
    if len(peaks) > 1:
        result = 1
    return result


def collimatorDefine(center1, center2, size1, size2):
    block1 = np.array([[1, 2, 0.5, center1 - size1 * 0.5, -1, 1, 1, 10]])
    block2 = np.array(
        [[1, 2, center1 + size1 * 0.5, center2 - size2 * 0.5, -1, 1, 2, 10]]
    )
    block3 = np.array([[1, 2, center2 + size2 * 0.5, 24.5, -1, 1, 2, 10]])
    return np.concatenate((block1, block2, block3), axis=0)


def visibilities(geoms, pAs, pBs):
    for pA in pAs:
        for pB in pBs:
            total_length = 0
            for cuboid in geoms:
                # pA_arr = (c_float * len(pA))(*pA)
                # pB_arr = (c_float * len(pB))(*pB)
                # cuboid_arr = (c_float * len(cuboid))(*cuboid)
                # total_length += get_intersection(cuboid_arr, pA_arr, pB_arr)
                total_length += get_intersection(cuboid, pA, pB)
            if total_length < 1e-8:
                return 1
    return 0
