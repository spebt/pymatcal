import numpy as np


def get_solid_angles(abpairs: np.ndarray, incs: np.ndarray):
    subAreas = np.array([incs[1] * incs[2], incs[2] * incs[0], incs[0] * incs[1]])
    ab_vec = abpairs[:, 3:] - abpairs[:, 0:3]
    ab_lens = np.linalg.norm(ab_vec, axis=1)
    norm_areas = np.abs(ab_vec * subAreas).T / ab_lens
    return np.sum(norm_areas / (ab_lens**2), axis=0)


def get_norm_areas(abpairs: np.ndarray, incs: np.ndarray):
    subAreas = np.array([incs[1] * incs[2], incs[2] * incs[0], incs[0] * incs[1]])
    ab_vec = abpairs[:, 3:] - abpairs[:, 0:3]
    ab_lens = np.linalg.norm(ab_vec, axis=1)
    norm_areas = np.abs(ab_vec * subAreas).T / ab_lens
    return np.sum(norm_areas)
