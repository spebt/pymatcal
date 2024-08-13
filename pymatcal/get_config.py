import yaml
import numpy as np
from jsonschema import Draft7Validator
import re
from ._utils import set_module
import importlib.resources as _resources
import json as _json
import pymatcal._schema as _schema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

__all__ = ["get_config", "get_fov_voxel_center", "get_procIds"]
# import importlib.resources
# schema_dir = importlib.resources.files("pymatcal._schemas")
# schema_path = pathlib.Path(schema_dir , 'config_schema.json')


@set_module("pymatcal")
def __parse_transformation_data(idata:dict) -> dict:
    if idata["format"] == "range":
        try: 
            start = float(idata["start"])
            ns = int(idata["N"])
            step = float(idata["step"])
        except: raise SyntaxError("Invalid transformation range data!!")
        return start + np.arange(0, ns) * step
    elif idata["format"] == "list":
        try: iarr = np.array(idata["data"], dtype="d")
        except: raise SyntaxError("Invalid transformation data enumerated")
        if len(iarr) == 0 :
            raise SyntaxError("Invalid transformation data enumerated, at least 1 number!!")
        return iarr
    else:
        raise SyntaxError("Invalid transformation data format!!")




def __get_schema_registry():
    # Load the schema
    _schema_dir = _resources.files(_schema)
    _schema_version = "v1"
    _basenames = ["main", "detector", "relation", "FOV", "transformation_data"]

    schema_registry = Registry()
    for _basename in _basenames:
        loaded = Resource(
            contents=_json.load(
                open(_schema_dir / _schema_version / f"{_basename}.json", "r")
            ),
            specification=DRAFT7,
        )
        
        schema_registry = loaded @ schema_registry
    return schema_registry


@set_module("pymatcal")
def get_config(confName: str):
    """
    Load and validate a configuration file in YAML format.

    :param confName: The name of the configuration file.
    :type confName: str
    :return: A dictionary containing the parsed configuration values.
    :rtype: dict
    :raises: Exception if the configuration file fails validation or parsing.
    """
    _schema_registry = __get_schema_registry()
    validator = Draft7Validator(schema=_schema_registry.contents('/v1/main.json'),registry=_schema_registry)
    with open(confName, "r") as stream:
        try:
            yamlConfig = yaml.safe_load(stream)
            validator.validate(instance=yamlConfig)
        except Exception as err:
            print("Error:", "Failed validating configuration file!!")
            print("Error Messages:\n%s" % err.message)
            raise
    mydict = {}
    try:
        geoms = np.asarray(yamlConfig["detector"]["detector geometry"], dtype="d")
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
            yamlConfig["detector"]["N-subdivision xyz"], dtype=np.int32
        )

        mydict["fov nsub"] = np.asarray(
            yamlConfig["FOV"]["N-subdivision xyz"], dtype=np.int32
        )

        mydict["fov nvx"] = np.asarray(
            yamlConfig["FOV"]["N-voxels xyz"], dtype=np.int32
        )

        mydict["mmpvx"] = np.asarray(yamlConfig["FOV"]["mm-per-voxel xyz"], dtype="d")
        mydict["rotation"] = __parse_transformation_data(yamlConfig["relation"]["rotation"])
        mydict["r shift"] = __parse_transformation_data(yamlConfig["relation"]["radial shift"])
        mydict["t shift"] = __parse_transformation_data(yamlConfig["relation"]["tangential shift"])
        # mydict["relation"] = __parse_relation(
        #     yamlConfig["relation"]
        # )
    except Exception as err:
        print("Parse Error!\n%s" % err)
        raise
    return mydict


@set_module("pymatcal")
def get_fov_voxel_center(
    id: np.uint64, nvx: np.ndarray, mmpvx: np.ndarray
) -> np.ndarray:
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
    assert id < np.prod(nvx), "Invalid voxel index!"
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
    nPerProc_add[0 : ntasks % nprocs] = np.ones(ntasks % nprocs)
    idxsPerProc = np.cumsum(
        np.insert(np.ones(nprocs) * (ntasks // nprocs) + nPerProc_add, 0, 0),
        dtype=np.uint32,
    )
    idxsPerProc = np.vstack((idxsPerProc[:-1], idxsPerProc[1:]))
    return idxsPerProc


get_procIds.__module__ = "pymatcal"
