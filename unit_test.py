import numpy as np
import yaml
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
    sensGeom = systemGeom[np.where(systemGeom[:, 6] == sensGeomIds)]
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
imageNxyz = imageDims * imageVxpms
mmPerVoxel = 1.0 / imageVxpms


# Task assignment
nImgVoxels = imageNxyz.prod()
nTasks = sensGeom.shape[0] * nImgVoxels


taskIdMin = 0
taskIdMax = nTasks
# print("Rank: %d" % rank, taskIdMin, taskIdMax)

print(
    "y-shift: ",
    y_shift,
    "x-shift: ",
    x_shift,
    "trans-x: ",
    trans_x,
    "trans_y: ",
    trans_y,
)
imgSubCentroid = np.array([[0, 50, 0]])
imageCoord = myfunc.coord_transform(
    angle_rad, x_shift, y_shift, trans_x, trans_y, imgSubCentroid
)
print(imageCoord)
imgSubLinespace_x = np.linspace(0, mmPerVoxel[0], imageSubs[0] + 1)
imgSubLinespace_y = np.linspace(0, mmPerVoxel[1], imageSubs[1] + 1)
imgSubLinespace_z = np.linspace(0, mmPerVoxel[2], imageSubs[2] + 1)
imgSubCentroid = np.array(
    np.meshgrid(
        0.5 * (imgSubLinespace_x[1:] + imgSubLinespace_x[:-1]),
        0.5 * (imgSubLinespace_y[1:] + imgSubLinespace_y[:-1]),
        0.5 * (imgSubLinespace_z[1:] + imgSubLinespace_z[:-1]),
    )
).T.reshape(imageSubs.prod(), 3)
detGeomId = 10
pA = np.array([0, 0, 0])
subCentroids = pA + imgSubCentroid
subInc = np.array([1.5, 1, 0.5])
geom = sensGeom[0]
# print(geom)
xlin = np.linspace(geom[0], geom[1], detSubs[0] + 1)
x_c = 0.5 * (xlin[1:] + xlin[:-1])
ylin = np.linspace(geom[2], geom[3], detSubs[1] + 1)
y_c = 0.5 * (ylin[1:] + ylin[:-1])
zlin = np.linspace(geom[4], geom[5], detSubs[2] + 1)
z_c = 0.5 * (zlin[1:] + zlin[:-1])
detSubCentroid = np.array(np.meshgrid(x_c, y_c, z_c)).T.reshape(detSubs.prod(), 3)
imgSubCentroids = myfunc.coord_transform(
        0, x_shift, y_shift, trans_x, trans_y, subCentroids)
ppdf=myfunc.PPFDForPair(systemGeom, imgSubCentroids, detSubCentroid, subInc, geom[7])
print(ppdf)
