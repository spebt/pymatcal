import yaml
import numpy as np
import jsonschema
import json
import os
import re
from ._utils import set_module

package_dir = os.path.dirname(os.path.abspath(__file__))

@set_module('pymatcal')
def __parse_dist(str: str) -> float:
    """
    Parses a string representing a distance value with unit and returns the distance in millimeters.

    :param str: The string representing the distance value with unit.
    :type str: str
    :return: The distance value in millimeters.
    :rtype: float
    :raises SyntaxError: If the input string is not in the correct format or the unit is invalid.
    """
    p = "^([0-9]+)(\\.[0-9]+)? ([a-z]+)$"
    result = re.match(p, str)
    unit = ''
    value = 0
    ngroups = len(result.groups())
    # print(ngroups)
    if ngroups == 2:
        value = float(result.group(1))
        unit = result.group(2)
    elif ngroups == 3:
        if result.group(2) is None:
            value = float(result.group(1))
            unit = result.group(3)
        else:
            value = float(result.group(1)+result.group(2))
            unit = result.group(3)
    else:
        raise SyntaxError("Invalid detector to FOV distance!!")

    if unit == 'mm':
        return value
    elif unit == 'cm':
        return value*10
    elif unit == 'm':
        return value*1000
    else:
        raise SyntaxError("Invalid detector to FOV distance unit!!")




@set_module('pymatcal')
def get_config(confName: str):
    """
    Load and validate a configuration file in YAML format.

    :param confName: The name of the configuration file.
    :type confName: str
    :return: A dictionary containing the parsed configuration values.
    :rtype: dict
    :raises: Exception if the configuration file fails validation or parsing.
    """
    schema = {}
    schemaFName = os.path.join(package_dir, "config_schema.json")
    # print(schemaFName)
    with open(schemaFName, "r") as data:
        schema = json.load(data)
        # print(type(schema))
    with open(confName, "r") as stream:
        try:
            yamlConfig = yaml.safe_load(stream)
            jsonschema.validate(instance=yamlConfig, schema=schema)
        except Exception as err:
            print("Error:", "Failed validating configuration file!!")
            print("Error Messages:\n%s" % err.message)
            raise
    mydict = {}
    try:
        geoms = np.asarray(
            yamlConfig["detector"]["detector geometry"], dtype="d"
        )
        mydict["det geoms"] = np.asarray(
            yamlConfig["detector"]["detector geometry"], dtype="d"
        )
        indices = np.asarray(
            yamlConfig["detector"]["active geometry indices"], dtype=np.int32
        )
        active_dets = []
        for idx in indices:
            active_dets.append(geoms[geoms[:, 6] == idx][0])
        mydict["active indices"] = indices
        mydict["active dets"] = np.array(active_dets)
        mydict["det nsub"] = np.asarray(
            yamlConfig["detector"]["N-subdivision xyz"], dtype=np.int32)

        mydict["img nsub"] = np.asarray(
            yamlConfig["image"]["N-subdivision xyz"], dtype=np.int32)

        mydict["img nvx"] = np.asarray(
            yamlConfig["image"]["N-voxels xyz"], dtype=np.int32)

        mydict["mmpvx"] = np.asarray(
            yamlConfig["image"]["mm-per-voxel xyz"], dtype="d")
        mydict["dist"] = __parse_dist(
            yamlConfig["detector-to-image"]["radial distance"])
        if "rotation" in yamlConfig["detector-to-image"].keys():
            mydict["angle"] = float(
                yamlConfig["detector-to-image"]["rotation"])
        else:
            mydict["angle"] = 0.0

    except Exception as err:
        print("Parse Error!\n%s" % err)
        raise
    return mydict

@set_module('pymatcal')
def get_img_voxel_center(id: np.uint64, nvx: np.ndarray, mmpvx: np.ndarray) -> np.ndarray:
    """
    Calculate the center coordinates of a voxel given its index.

    :param id: The index of the voxel.
    :type id: np.uint64
    :param nvx: The number of voxels in each dimension.
    :type nvx: np.ndarray
    :param mmpvx: The size of each voxel in millimeters.
    :type mmpvx: np.ndarray
    :return: The center coordinates of the voxel.
    :rtype: np.ndarray
    :raises AssertionError: If the given voxel index is invalid.
    """
    # make sure the 1-D index given is valid
    assert (id < np.prod(nvx)), 'Invalid voxel index!'
    # index order, slowest to quickest changing: z -> y -> x
    zid = id // (nvx[0] * nvx[1])
    xyid = id % (nvx[0] * nvx[1])
    yid = xyid // nvx[0]
    xid = xyid % nvx[0]
    # print('z -> y -> x:', '%d -> %d -> %d'%(zid, yid, xid))
    return (np.array([xid, yid, zid]) - np.append(nvx[:2], 0) * 0.5) * mmpvx


def get_procIds(ntasks: np.uint64, nprocs: np.uint64) -> np.ndarray:
    """
    Calculate the indices of tasks assigned to each process.

    :param ntasks: The total number of tasks.
    :type ntasks: np.uint64
    :param nprocs: The total number of processes.
    :type nprocs: np.uint64
    :return: A 2D array containing the start and end indices of tasks assigned to each process.
    :rtype: np.ndarray
    """
    nPerProc_add = np.zeros(nprocs)
    nPerProc_add[0:ntasks % nprocs] = np.ones(ntasks % nprocs)
    idxsPerProc = np.cumsum(
        np.insert(np.ones(nprocs)*(ntasks // nprocs)+nPerProc_add, 0, 0), dtype=np.uint32)
    idxsPerProc = np.vstack((idxsPerProc[:-1], idxsPerProc[1:]))
    return idxsPerProc


get_procIds.__module__ = "pymatcal"
