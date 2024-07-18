import numpy as np
from .get_config import *
from .intersections import *
from .coord_transform import *
from .solid_angle import *
from .subdivs import *
from .coord_transform import *
from ._utils import set_module

def get_pair_ppdf(ida: np.uint64, idb: np.uint64, img_subdivs: dict, config: dict) -> np.float64:
    det_dimy = np.max(config['det geoms'][:, 3]) - \
        np.min(config['det geoms'][:, 2])
    pointA = get_img_voxel_center(ida, config['img nvx'], config['mmpvx'])
    geomB = config['active dets'][idb]
    det_subdivs = get_det_subdivs(
        geomB, config['det nsub'])
    geoms = append_subdivs(
        config['det geoms'], geomB, det_subdivs['geoms'])
    pBs = get_centroids(det_subdivs['geoms'])
    pAs = coord_transform(get_mtransform(
        config['angle'], -config['dist'], det_dimy*0.5), img_subdivs['coords']+pointA)
    abpairs = get_AB_pairs(pAs, pBs)
    ret = get_intersections_2d(geoms, abpairs)
    idx_absorp = ret['ts'][:, :, 1] == 1
    idx_attenu = ret['ts'][:, :, 1] != 1
    segs_absorp = np.where(idx_absorp, ret['intersections'], 0)
    segs_attenu = np.where(idx_attenu, ret['intersections'], 0)
    subdivs_sa = get_solid_angles(abpairs, det_subdivs['incs'])
    term_attenu = np.exp(-np.sum(segs_attenu.T*geoms[:, 7], axis=1))
    term_absorp = 1-np.exp(-np.sum(segs_absorp.T*geoms[:, 7], axis=1))
    term_solida = subdivs_sa / (4*np.pi)
    ab_rays_ppd = term_attenu*term_absorp*term_solida
    return np.sum(ab_rays_ppd)/np.prod(config['img nsub'])


def get_pair_ppdf_area(ida: np.uint64, idb: np.uint64, img_subdivs: dict, config: dict) -> np.float64:
    det_dimy = np.max(config['det geoms'][:, 3]) - \
        np.min(config['det geoms'][:, 2])
    pointA = get_img_voxel_center(ida, config['img nvx'], config['mmpvx'])
    geomB = config['active dets'][idb]
    det_subdivs = get_det_subdivs(
        geomB, config['det nsub'])
    geoms = append_subdivs(
        config['det geoms'], geomB, det_subdivs['geoms'])
    pBs = get_centroids(det_subdivs['geoms'])
    pAs = coord_transform(get_mtransform(
        config['angle'], -config['dist'], det_dimy*0.5), img_subdivs['coords']+pointA)
    abpairs = get_AB_pairs(pAs, pBs)
    ret = get_intersections_2d(geoms, abpairs)
    idx_absorp = ret['ts'][:, :, 1] == 1
    idx_attenu = ret['ts'][:, :, 1] != 1
    segs_absorp = np.where(idx_absorp, ret['intersections'], 0)
    segs_attenu = np.where(idx_attenu, ret['intersections'], 0)
    subdivs_sa = get_norm_areas(abpairs, det_subdivs['incs'])
    term_attenu = np.exp(-np.sum(segs_attenu.T*geoms[:, 7], axis=1))
    term_absorp = 1-np.exp(-np.sum(segs_absorp.T*geoms[:, 7], axis=1))
    term_solida = subdivs_sa / (4*np.pi)
    ab_rays_ppd = term_attenu*term_absorp*term_solida
    return np.sum(ab_rays_ppd)/np.prod(config['img nsub'])


def get_pair_ppdf_binary(ida: np.uint64, idb: np.uint64, img_subdivs: dict, config: dict) -> np.float64:
    det_dimy = np.max(config['det geoms'][:, 3]) - \
        np.min(config['det geoms'][:, 2])
    pointA = get_img_voxel_center(ida, config['img nvx'], config['mmpvx'])
    geomB = config['active dets'][idb]
    det_subdivs = get_det_subdivs(
        geomB, config['det nsub'])
    geoms = append_subdivs(
        config['det geoms'], geomB, det_subdivs['geoms'])
    pBs = get_centroids(det_subdivs['geoms'])
    pAs = coord_transform(get_mtransform(
        config['angle'], -config['dist'], det_dimy*0.5), img_subdivs['coords']+pointA)
    abpairs = get_AB_pairs(pAs, pBs)
    ret = get_intersections_2d(geoms, abpairs)
    idx_attenu = ret['ts'][:, :, 1] != 1
    segs_attenu = np.where(idx_attenu, ret['intersections'], 0)
    return 0 if np.any(np.sum(segs_attenu, axis=0) == 0) else 1
