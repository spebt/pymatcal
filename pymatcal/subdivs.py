import numpy as np
from coord_transform import *


def get_img_subdivs(mmpvx, nsubs):
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


def get_det_subdivs(focs: np.ndarray, nsubs: np.ndarray):
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


def get_centroids(geoms: np.ndarray) -> np.ndarray:
    return (geoms[:, :6:2]+geoms[:, 1:6:2])*0.5


def append_subdivs(geoms: np.ndarray, focs: np.ndarray, subdiv_geoms: np.ndarray):
    blocks = geoms[geoms[:, 6] != focs[6]]
    return np.vstack((blocks, subdiv_geoms))
