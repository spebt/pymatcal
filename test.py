import yaml
import numpy as np
import pyMatcal_routines as myfunc

configFileName = "configs/config.yml"
with open(configFileName, "r") as stream:
    try:
        yamlConfig = yaml.safe_load(stream)
    except yaml.YAMLError as err:
        print(err)


# Read in the geometry
try:
    systemGeom = np.asarray(yamlConfig["detector geometry"])
    sensGeomIds = np.asarray(yamlConfig["detector"]["sensitive geometry indices"])
    sensGeom = systemGeom[sensGeomIds]
    detSubs = np.asarray(yamlConfig["detector"]["crystal n subdivision xyz"])
    # Calculate Image space N subdivision and size.
    imageDims = np.asarray(yamlConfig["image"]["dimension xyz"])
    imageVxpms = np.asarray(yamlConfig["image"]["voxel per mm xyz"])
    imageSubs = np.asarray(yamlConfig["image"]["subdivision xyz"])
    angle_rad = yamlConfig["image"]["detector rotation"]
    x_shift = yamlConfig["image"]["detector x-shift"]
except yaml.YAMLError as err:
    print("Error reading the configurations!", err)
    exit(1)

yMin = np.amin(systemGeom[:, 2])
yMax = np.amax(systemGeom[:, 3])
trans_x = imageDims[0] * 0.5
trans_y = imageDims[1] * 0.5
y_shift = 0.5 * (yMax - yMin)
print(
    "y-shift: ", y_shift, "x-shift", x_shift, "trans-x: ", trans_x, "trans_y", trans_y
)
imageCoord = np.array([0, 0, 0])
print("Input: ", imageCoord)
imageCoord = myfunc.coord_transform(
    angle_rad, x_shift, y_shift, trans_x, trans_y, imageCoord
)
print("Transformed: ", imageCoord)
