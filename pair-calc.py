import yaml
import numpy as np
import pyMatcal_routines as myfunc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
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
    angle_rad = yamlConfig["detector"]["detector rotation"]
    x_shift = yamlConfig["detector"]["detector x-shift"]
    y_shift = 0.5 * yamlConfig["detector"]["detector y-dimension"]
    trans_x = yamlConfig["image"]["x dimension"]
    trans_y = yamlConfig["image"]["y dimension"]

except yaml.YAMLError as err:
    print("Error reading the configurations!", err)
    exit(1)
imageNxyz = imageDims * imageVxpms

# Task assignment
nImgVoxels = imageNxyz.prod()
nTasks = sensGeom.shape[0] * nImgVoxels
nTasksPerProc = np.ceil(nTasks / nproc)
if rank == 0:
    print("N total Tasks: %d, N tasks per proc: %.1f\n" % (nTasks, nTasksPerProc))
# print(imageNxyz, imageNxyz.prod())

taskIdMin = rank * nTasksPerProc
taskIdMax = min(taskIdMin + nTasksPerProc, nTasks)
# print("Rank: %d" % rank, taskIdMin, taskIdMax)
for taskId in np.arange(taskIdMin, taskIdMax):
    detGeomId = taskId // nImgVoxels
    imgVoxelId = taskId % nImgVoxels
    imgVoxelIdx = imgVoxelId // (imageNxyz[1] * imageNxyz[2])
    imgVoxelIdyz = imgVoxelId % (imageNxyz[1] * imageNxyz[2])
    imgVoxelIdy = imgVoxelIdyz // imageNxyz[2]
    imgVoxelIdz = imgVoxelIdyz % imageNxyz[2]

    imageVoxelIds = np.array([imgVoxelIdx, imgVoxelIdy, imgVoxelIdz])
    imageVoxelCoords = imageVoxelIds / imageVxpms
    geom = sensGeom[int(detGeomId)]
    imageCoord = myfunc.coord_transform(
        angle_rad, x_shift, y_shift, trans_x, trans_y, imageVoxelCoords
    )
    # Detector unit subdivision centroid coordinates
    xlin = np.linspace(geom[0], geom[1], detSubs[0] + 1)
    x_c = 0.5 * (xlin[1:] + xlin[:-1])
    ylin = np.linspace(geom[2], geom[3], detSubs[1] + 1)
    y_c = 0.5 * (ylin[1:] + ylin[:-1])
    zlin = np.linspace(geom[4], geom[5], detSubs[2] + 1)
    z_c = 0.5 * (zlin[1:] + zlin[:-1])
    centers = np.array(np.meshgrid(x_c, y_c, z_c))
    centers = centers.T.reshape(detSubs.prod(), 3)

    # if rank == 0:
    #     print("taskId %2d: (Det: %2d, Img: %2d)" % (taskId, detGeomId, imgVoxlId))
