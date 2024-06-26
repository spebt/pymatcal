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


def get_det_subdivs(geom, nsubs):
    xlin = np.linspace(geom[0], geom[1], int(nsubs[0]) + 1)
    ylin = np.linspace(geom[2], geom[3], int(nsubs[1]) + 1)
    zlin = np.linspace(geom[4], geom[5], int(nsubs[2]) + 1)
    return {"coords": np.array(np.meshgrid(0.5 * (xlin[1:] + xlin[:-1]), 0.5 * (
        ylin[1:] + ylin[:-1]), 0.5 * (zlin[1:] + zlin[:-1]))).T.reshape(int(nsubs.prod()), 3),
        "incs": np.array(
        [xlin[1] - xlin[0], ylin[1] - ylin[0], zlin[1] - zlin[0]]
    )}
