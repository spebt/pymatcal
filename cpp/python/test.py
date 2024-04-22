from ctypes import *
import pyMatcal_routines as myfunc
import yaml
import numpy as np

libray_cuboid_intersection = CDLL("libray_cuboid_intersection.so")
get_intersection = libray_cuboid_intersection.get_intersection
# define prototypes
ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
get_intersection.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1]
get_intersection.restype = c_float


def pair_visibility(geoms, pAs, pBs):
    for pA in pAs:
        for pB in pBs:
            total_length = 0
            for cuboid in geoms:
                # pA_arr = (c_float * len(pA))(*pA)
                # pB_arr = (c_float * len(pB))(*pB)
                # cuboid_arr = (c_float * len(cuboid))(*cuboid)
                # total_length += get_intersection(cuboid_arr, pA_arr, pB_arr)
                total_length += get_intersection(cuboid, pA, pB)
            if total_length < 1e-8:
                return 1
    return 0


configFileName = "./config-multiplexing.yml"
with open(configFileName, "r") as stream:
    try:
        yamlConfig = yaml.safe_load(stream)
    except yaml.YAMLError as err:
        print(err)


# Read in the geometry
try:
    systemGeom = np.asarray(yamlConfig["detector geometry"])
    sensGeomIds = np.asarray(yamlConfig["detector"]["sensitive geometry indices"])

    detSubs = np.asarray(yamlConfig["detector"]["crystal n subdivision xyz"])
    # Calculate Image space N subdivision and size.
    imageDims = np.asarray(yamlConfig["image"]["dimension xyz"])
    imageVxpms = np.asarray(yamlConfig["image"]["voxel per mm xyz"])
    imageSubs = np.asarray(yamlConfig["image"]["subdivision xyz"])
    angle_rad = yamlConfig["image"]["detector rotation"]
    x_shift = yamlConfig["image"]["detector x-shift"]

    center1s = np.array(yamlConfig["optimization"]["1st aperture center y (mm)"])
    center2s = np.array(yamlConfig["optimization"]["2nd aperture center y (mm)"])
    size1s = np.array(yamlConfig["optimization"]["1st aperture size (mm)"])
    size2s = np.array(yamlConfig["optimization"]["2nd aperture size (mm)"])

except yaml.YAMLError as err:
    print("Error reading the configurations!", err)
    exit(1)

passiveCuboids = np.concatenate(
    (
        systemGeom[np.where(systemGeom[:, 6] == 0)],
        systemGeom[np.where(systemGeom[:, 6] == 1)],
    ),
    axis=0,
)

sensitiveCuboids = systemGeom[np.where(systemGeom[:, 6] == 3)]
print(sensitiveCuboids)
print(passiveCuboids.shape)


yMin = np.amin(systemGeom[:, 2])
yMax = np.amax(systemGeom[:, 3])
trans_x = imageDims[0] * 0.5
trans_y = imageDims[1] * 0.5
y_shift = 0.5 * (yMax - yMin)
imageNxyz = imageDims * imageVxpms
mmPerVoxel = 1.0 / imageVxpms

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

nImgVoxels = imageNxyz.prod()
nTasks = int(sensitiveCuboids.shape[0] * nImgVoxels)
output = np.zeros(nTasks, dtype=float)

for taskId in range(0, nTasks):
    # local sensitive detector Id
    detGeomId = taskId // nImgVoxels
    # local image voxel Id
    imgVoxelId = taskId % nImgVoxels

    # local image voxel Id x,y,z
    imgVoxelIdx = imgVoxelId // (imageNxyz[1] * imageNxyz[2])
    imgVoxelIdyz = imgVoxelId % (imageNxyz[1] * imageNxyz[2])
    imgVoxelIdy = imgVoxelIdyz // imageNxyz[2]
    imgVoxelIdz = imgVoxelIdyz % imageNxyz[2]

    # local image voxel coordinates in image reference frame, Cartesian coordinate system
    voxelBaseCoords = mmPerVoxel * np.array([imgVoxelIdx, imgVoxelIdy, imgVoxelIdz])
    # subdivisions coordinates
    localImgSubCentroids = voxelBaseCoords + imgSubCentroid

    # Detector voxel subdivision centroid coordinates
    geom = sensitiveCuboids[int(detGeomId)]

    xlin = np.linspace(geom[0], geom[1], detSubs[0] + 1)
    x_c = 0.5 * (xlin[1:] + xlin[:-1])
    ylin = np.linspace(geom[2], geom[3], detSubs[1] + 1)
    y_c = 0.5 * (ylin[1:] + ylin[:-1])
    zlin = np.linspace(geom[4], geom[5], detSubs[2] + 1)
    z_c = 0.5 * (zlin[1:] + zlin[:-1])

    detSubIncrement = np.array(
        [xlin[1] - xlin[0], ylin[1] - ylin[0], zlin[1] - zlin[0]]
    )

    localImgSubCentroids = myfunc.coord_transform(
        angle_rad, x_shift, y_shift, trans_x, trans_y, localImgSubCentroids
    )
    detSubCentroids = np.array(np.meshgrid(x_c, y_c, z_c)).T.reshape(detSubs.prod(), 3)
    visibility = 0
    for pA in localImgSubCentroids:
        for pB in detSubCentroids:
            total_length = 0
            for cuboid in passiveCuboids:
                # pA_arr = (c_float * len(pA))(*pA)
                # pB_arr = (c_float * len(pB))(*pB)
                # cuboid_arr = (c_float * len(cuboid))(*cuboid)
                # total_length += get_intersection(cuboid_arr, pA_arr, pB_arr)
                total_length += get_intersection(cuboid, pA, pB)
            print(total_length)
            if total_length < 1e-8:
                visibility = 1


    # visibility = pair_visibility(passiveCuboids, localImgSubCentroids, detSubCentroids)

    # visibility = myfunc.visibilities(systemGeom, localImgSubCentroids, detSubCentroid)
    if visibility > 0.5:
        print(taskId)
    output[taskId] = visibility

print("\nSaving to file: %s" % yamlConfig["out npz filename"])
np.savez_compressed(yamlConfig["out npz filename"], sysmat=output)