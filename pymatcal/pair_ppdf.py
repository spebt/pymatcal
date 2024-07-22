import numpy
from .get_config import *
from .intersections import *
from .coord_transform import *
from .solid_angle import *
from .subdivs import *
from .coord_transform import *
from ._utils import set_module

@set_module('pymatcal')
def get_pair_ppdf(ida: numpy.uint64, idb: numpy.uint64, img_subdivs: dict, config: dict) -> numpy.float64:
    """
    Calculate the pair projection probability density function (PPDF) for a pair of FOV voxel and detector unit.

    :param ida: The ID of the FOV voxel.
    :type ida: numpy.uint64
    :param idb: The ID of the second detector.
    :type idb: numpy.uint64
    :param img_subdivs: A dictionary containing the subdivisions of the image.
    :type img_subdivs: dict
    :param config: A dictionary containing the configuration parameters.
    :type config: dict
    :return: The PPDF value of the pair.
    :rtype: numpy.float64
    :raises: None
    """
    det_dimy = numpy.max(config['det geoms'][:, 3]) - \
        numpy.min(config['det geoms'][:, 2])
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
    segs_absorp = numpy.where(idx_absorp, ret['intersections'], 0)
    segs_attenu = numpy.where(idx_attenu, ret['intersections'], 0)
    subdivs_sa = get_solid_angles(abpairs, det_subdivs['incs'])
    term_attenu = numpy.exp(-numpy.sum(segs_attenu.T*geoms[:, 7], axis=1))
    term_absorp = 1-numpy.exp(-numpy.sum(segs_absorp.T*geoms[:, 7], axis=1))
    term_solida = subdivs_sa / (4*numpy.pi)
    ab_rays_ppd = term_attenu*term_absorp*term_solida
    return numpy.sum(ab_rays_ppd)/numpy.prod(config['img nsub'])

@set_module('pymatcal')
def get_pair_ppdf_area(ida: numpy.uint64, idb: numpy.uint64, img_subdivs: dict, config: dict) -> numpy.float64:
    """
    Calculate the pair projection probability density function (PPDF) for a pair of FOV voxel and detector unit. 
    This function uses the projection area as the solid angle. The solid angle will not decrease as the distance between the FOV voxel and the detector unit increases.

    :param ida: The ID of the FOV voxel.
    :type ida: numpy.uint64
    :param idb: The ID of the second detector.
    :type idb: numpy.uint64
    :param img_subdivs: A dictionary containing the subdivisions of the image.
    :type img_subdivs: dict
    :param config: A dictionary containing the configuration parameters.
    :type config: dict
    :return: The PPDF value of the pair.
    :rtype: numpy.float64
    :raises: None
    """

    det_dimy = numpy.max(config['det geoms'][:, 3]) - \
        numpy.min(config['det geoms'][:, 2])
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
    segs_absorp = numpy.where(idx_absorp, ret['intersections'], 0)
    segs_attenu = numpy.where(idx_attenu, ret['intersections'], 0)
    subdivs_sa = get_norm_areas(abpairs, det_subdivs['incs'])
    term_attenu = numpy.exp(-numpy.sum(segs_attenu.T*geoms[:, 7], axis=1))
    term_absorp = 1-numpy.exp(-numpy.sum(segs_absorp.T*geoms[:, 7], axis=1))
    term_solida = subdivs_sa / (4*numpy.pi)
    ab_rays_ppd = term_attenu*term_absorp*term_solida
    return numpy.sum(ab_rays_ppd)/numpy.prod(config['img nsub'])

@set_module('pymatcal')
def get_pair_ppdf_binary(ida: numpy.uint64, idb: numpy.uint64, img_subdivs: dict, config: dict) -> numpy.float64:
    """
    Calculate the pair projection probability density function (PPDF) for a pair of FOV voxel and detector unit. The PPDF value is binary, where 1 indicates that the ray is NOT blocked and 0 indicates that the pair is blocked.

    :param ida: The ID of the FOV voxel.
    :type ida: numpy.uint64
    :param idb: The ID of the second detector.
    :type idb: numpy.uint64
    :param img_subdivs: A dictionary containing the subdivisions of the image.
    :type img_subdivs: dict
    :param config: A dictionary containing the configuration parameters.
    :type config: dict
    :return: The PPDF value of the pair.
    :rtype: numpy.float64
    :raises: None
    """

    det_dimy = numpy.max(config['det geoms'][:, 3]) - \
        numpy.min(config['det geoms'][:, 2])
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
    segs_attenu = numpy.where(idx_attenu, ret['intersections'], 0)
    return 0 if numpy.any(numpy.sum(segs_attenu, axis=0) == 0) else 1
