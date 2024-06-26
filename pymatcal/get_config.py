import yaml
import numpy as np
import jsonschema
import json
import os
import re

package_dir = os.path.dirname(os.path.abspath(__file__))


def __parse_dist(str: str):
    p = "^([0-9]+)(\\.[0-9]+)? ([a-z]+)$"
    result = re.match(p, str)
    unit = ''
    value = 0
    ngroups = len(result.groups())
    # print(ngroups)
    match ngroups:
        case 2:
            value = float(result.group(1))
            unit = result.group(2)
        case 3:
            if result.group(2) is None:
                value = float(result.group(1))
                unit = result.group(3)
            else:
                value = float(result.group(1)+result.group(2))
                unit = result.group(3)
        case _:
            raise SyntaxError("Invalid detector to FOV distance!!")
    match unit:
        case 'mm':
            return value
        case 'cm':
            return value*10
        case 'm':
            return value*1000
        case _:
            raise SyntaxError("Invalid detector to FOV distance unit!!")


def get_config(confName: str):
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
        mydict["det geoms"] = np.asarray(
            yamlConfig["detector"]["detector geometry"], dtype="d"
        )
        indices = np.asarray(
            yamlConfig["detector"]["active geometry indices"], dtype=np.int32
        )
        mydict["active det"] = np.asarray(
            mydict["det geoms"][mydict["det geoms"][:, 6] == indices], dtype='d')
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
