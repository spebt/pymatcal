import numpy
from .get_config import *
from .intersections import *
from .coord_transform import *
from .solid_angle import *
from .subdivs import *
from .coord_transform import *
from ._utils import set_module

__all__ = ["get_pair_ppdf", "get_pair_ppdf_area", "get_pair_ppdf_binary"]


@set_module("pymatcal")
def get_pair_ppdf(
    ida: numpy.uint64,
    idb: numpy.uint64,
    idrot: int,
    idr: int,
    idt: int,
    fov_subdivs: dict,
    config: dict,
) -> numpy.float64:
    """
    Calculate the pair projection probability density function (PPDF) for a pair of FOV voxel and detector unit.

    :param ida: The ID of the FOV voxel.
    :type ida: numpy.uint64
    :param idb: The ID of the second detector.
    :type idb: numpy.uint64
    :param idrot: The ID of the rotation.
    :type idrot: int
    :param idr: The ID of the r-shift.
    :type idr: int
    :param idt: The ID of the t-shift.
    :type idt: int
    :param fov_subdivs: A dictionary containing the subdivisions of the image.
    :type fov_subdivs: dict
    :param config: A dictionary containing the configuration parameters.
    :type config: dict
    :return: The PPDF value of the pair.
    :rtype: numpy.float64
    :raises: None
    """
    det_dimy = numpy.max(config["det geoms"][:, 3]) - numpy.min(
        config["det geoms"][:, 2]
    )
    pointA = get_fov_voxel_center(ida, config["fov nvx"], config["mmpvx"])
    geomB = config["active dets"][idb]
    det_subdivs = get_det_subdivs(geomB, config["det nsub"])
    geoms = append_subdivs(config["det geoms"], geomB, det_subdivs["geoms"])
    pBs = get_centroids(det_subdivs["geoms"])
    angle = config["rotation"][idrot]
    rshift = config["r shift"][idr]
    tshift = config["t shift"][idt]
    pAs = coord_transform(
        get_mtransform(angle, -rshift, det_dimy * 0.5 - tshift),
        fov_subdivs["coords"] + pointA,
    )
    abpairs = get_AB_pairs(pAs, pBs)
    ret = get_intersections_2d(geoms, abpairs)
    idx_absorp = ret["ts"][:, :, 1] == 1
    idx_attenu = ret["ts"][:, :, 1] != 1
    segs_absorp = numpy.where(idx_absorp, ret["intersections"], 0)
    segs_attenu = numpy.where(idx_attenu, ret["intersections"], 0)
    subdivs_sa = get_solid_angles(abpairs, det_subdivs["incs"])
    term_attenu = numpy.exp(-numpy.sum(segs_attenu.T * geoms[:, 7], axis=1))
    term_absorp = 1 - numpy.exp(-numpy.sum(segs_absorp.T * geoms[:, 7], axis=1))
    term_solida = subdivs_sa / (4 * numpy.pi)
    ab_rays_ppd = term_attenu * term_absorp * term_solida
    return numpy.sum(ab_rays_ppd) / numpy.prod(config["fov nsub"])


@set_module("pymatcal")
def get_pair_ppdf_area(
    ida: numpy.uint64,
    idb: numpy.uint64,
    idrot: int,
    idr: int,
    idt: int,
    fov_subdivs: dict,
    config: dict,
) -> numpy.float64:
    """
    Calculate the pair projection probability density function (PPDF) for a pair of FOV voxel and detector unit.
    This function uses the projection area as the solid angle. The solid angle will not decrease as the distance between the FOV voxel and the detector unit increases.

    :param ida: The ID of the FOV voxel.
    :type ida: numpy.uint64
    :param idb: The ID of the second detector.
    :type idb: numpy.uint64
    :param idrot: The ID of the rotation.
    :type idrot: int
    :param idr: The ID of the r-shift.
    :type idr: int
    :param idt: The ID of the t-shift.
    :type idt: int
    :param fov_subdivs: A dictionary containing the subdivisions of the image.
    :type fov_subdivs: dict
    :param config: A dictionary containing the configuration parameters.
    :type config: dict
    :return: The PPDF value of the pair.
    :rtype: numpy.float64
    :raises: None
    """

    det_dimy = numpy.max(config["det geoms"][:, 3]) - numpy.min(
        config["det geoms"][:, 2]
    )
    pointA = get_fov_voxel_center(ida, config["fov nvx"], config["mmpvx"])
    geomB = config["active dets"][idb]
    det_subdivs = get_det_subdivs(geomB, config["det nsub"])
    geoms = append_subdivs(config["det geoms"], geomB, det_subdivs["geoms"])
    pBs = get_centroids(det_subdivs["geoms"])
    angle = config["rotation"][idrot]
    rshift = config["r shift"][idr]
    tshift = config["t shift"][idt]
    pAs = coord_transform(
        get_mtransform(angle, -rshift, det_dimy * 0.5 - tshift),
        fov_subdivs["coords"] + pointA,
    )
    abpairs = get_AB_pairs(pAs, pBs)
    ret = get_intersections_2d(geoms, abpairs)
    idx_absorp = ret["ts"][:, :, 1] == 1
    idx_attenu = ret["ts"][:, :, 1] != 1
    segs_absorp = numpy.where(idx_absorp, ret["intersections"], 0)
    segs_attenu = numpy.where(idx_attenu, ret["intersections"], 0)
    subdivs_sa = get_norm_areas(abpairs, det_subdivs["incs"])
    term_attenu = numpy.exp(-numpy.sum(segs_attenu.T * geoms[:, 7], axis=1))
    term_absorp = 1 - numpy.exp(-numpy.sum(segs_absorp.T * geoms[:, 7], axis=1))
    term_solida = subdivs_sa / (4 * numpy.pi)
    ab_rays_ppd = term_attenu * term_absorp * term_solida
    return numpy.sum(ab_rays_ppd) / numpy.prod(config["fov nsub"])


@set_module("pymatcal")
def get_pair_ppdf_binary(
    ida: numpy.uint64,
    idb: numpy.uint64,
    idrot: int,
    idr: int,
    idt: int,
    fov_subdivs: dict,
    config: dict,
) -> numpy.float64:
    """
    Calculate the pair projection probability density function (PPDF) for a pair of FOV voxel and detector unit. The PPDF value is binary, where 1 indicates that the ray is NOT blocked and 0 indicates that the pair is blocked.

    :param ida: The ID of the FOV voxel.
    :type ida: numpy.uint64
    :param idb: The ID of the second detector.
    :type idb: numpy.uint64
    :param idrot: The ID of the rotation.
    :type idrot: int
    :param idr: The ID of the r-shift.
    :type idr: int
    :param idt: The ID of the t-shift.
    :type idt: int
    :param fov_subdivs: A dictionary containing the subdivisions of the image.
    :type fov_subdivs: dict
    :param config: A dictionary containing the configuration parameters.
    :type config: dict
    :return: The PPDF value of the pair.
    :rtype: numpy.float64
    :raises: None
    """

    det_dimy = numpy.max(config["det geoms"][:, 3]) - numpy.min(
        config["det geoms"][:, 2]
    )
    pointA = get_fov_voxel_center(ida, config["fov nvx"], config["mmpvx"])
    geomB = config["active dets"][idb]
    det_subdivs = get_det_subdivs(geomB, config["det nsub"])
    geoms = append_subdivs(config["det geoms"], geomB, det_subdivs["geoms"])
    pBs = get_centroids(det_subdivs["geoms"])
    angle = config["rotation"][idrot]
    rshift = config["r shift"][idr]
    tshift = config["t shift"][idt]
    pAs = coord_transform(
        get_mtransform(angle, -rshift, det_dimy * 0.5 - tshift),
        fov_subdivs["coords"] + pointA,
    )
    abpairs = get_AB_pairs(pAs, pBs)
    ret = get_intersections_2d(geoms, abpairs)
    idx_attenu = ret["ts"][:, :, 1] != 1
    segs_attenu = numpy.where(idx_attenu, ret["intersections"], 0)
    return 0 if numpy.any(numpy.sum(segs_attenu, axis=0) == 0) else 1

"""
@set_module("pymatcal")
def get_pair_ppdf(
    ida_array: np.ndarray,  # Array of FOV voxel IDs
    idb: np.uint64,         # Single detector ID
    idrot: int,
    idr: int,
    idt: int,
    fov_subdivs: dict,
    config: dict
) -> np.ndarray:
    
    Calculate the pair projection probability density function (PPDF) for a single detector unit
    and multiple FOV voxels.

    :param ida_array: An array of FOV voxel IDs.
    :type ida_array: np.ndarray
    :param idb: The ID of the detector unit.
    :type idb: np.uint64
    :param idrot: The ID of the rotation.
    :type idrot: int
    :param idr: The ID of the r-shift.
    :type idr: int
    :param idt: The ID of the t-shift.
    :type idt: int
    :param fov_subdivs: A dictionary containing subdivisions of the image.
    :type fov_subdivs: dict
    :param config: A dictionary containing configuration parameters.
    :type config: dict
    :return: An array of PPDF values corresponding to each FOV voxel in ida_array.
    :rtype: np.ndarray
    

    # Get dimensions related to detectors from configuration
    det_dimy = np.max(config["det geoms"][:, 3]) - np.min(config["det geoms"][:, 2])

    # Get centers for all FOV voxels in ida_array
    pointA_array = np.array([get_fov_voxel_center(ida, config["fov nvx"], config["mmpvx"]) for ida in ida_array])

    # Get geometry for this specific detector unit (idb)
    geomB = config["active dets"][idb]
    
    # Get subdivisions for this detector unit
    det_subdivs = get_det_subdivs(geomB, config["det nsub"])
    
    # Append subdivisions to detector geometry and get centroids (pBs)
    geoms = append_subdivs(config["det geoms"], geomB, det_subdivs["geoms"])
    
    pBs = get_centroids(det_subdivs["geoms"])

    # Apply transformation based on rotation, r-shift, and t-shift
    angle = config["rotation"][idrot]
    rshift = config["r shift"][idr]
    tshift = config["t shift"][idt]

    # Transform coordinates for all FOV voxels in pointA_array
    pAs_array = coord_transform(
        get_mtransform(angle, -rshift, det_dimy * 0.5 - tshift),
        fov_subdivs["coords"] + pointA_array[:, None]  # Broadcasting over all points
    )

    # Initialize an empty list to store PPDF results for each FOV voxel
    ppdf_results = []

    # Loop over each transformed set of points (one per FOV voxel)
    for pAs in pAs_array:
        abpairs = get_AB_pairs(pAs, pBs)
        ret = get_intersections_2d(geoms, abpairs)

        idx_absorp = ret["ts"][:, :, 1] == 1
        idx_attenu = ret["ts"][:, :, 1] != 1

        segs_absorp = np.where(idx_absorp, ret["intersections"], 0)
        segs_attenu = np.where(idx_attenu, ret["intersections"], 0)

        subdivs_sa = get_solid_angles(abpairs, det_subdivs["incs"])

        term_attenu = np.exp(-np.sum(segs_attenu.T * geoms[:, 7], axis=1))
        term_absorp = 1 - np.exp(-np.sum(segs_absorp.T * geoms[:, 7], axis=1))
        term_solida = subdivs_sa / (4 * np.pi)

        ab_rays_ppd = term_attenu * term_absorp * term_solida

        # Sum up PPDF values for this particular FOV voxel and store it in results list
        ppdf_results.append(np.sum(ab_rays_ppd) / np.prod(config["fov nsub"]))

    return np.array(ppdf_results)  # Return as a NumPy array
"""
