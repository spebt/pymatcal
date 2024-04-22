import yaml
import numpy as np
import pyMatcal_routines as myfunc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
configFileName = "configs/config-multiplexing.yml"
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

sensGeom = []
for index in sensGeomIds:
    sensGeom.append(systemGeom[np.nonzero(systemGeom[:, 6] == index)].flatten())
sensGeom=np.array(sensGeom)

yMin = np.amin(systemGeom[:, 2])
yMax = np.amax(systemGeom[:, 3])
trans_x = imageDims[0] * 0.5
trans_y = imageDims[1] * 0.5
y_shift = 0.5 * (yMax - yMin)
imageNxyz = imageDims * imageVxpms
mmPerVoxel = 1.0 / imageVxpms


# Task assignment
nImgVoxels = imageNxyz.prod()
nTasks = int(sensGeom.shape[0] * nImgVoxels)
nTasksPerProc = int(np.ceil(nTasks / nproc))
sysmat = None
if rank == 0:
    print("N total Tasks: %d, N tasks per proc: %.1f\n" % (nTasks, nTasksPerProc))
    sysmat = np.empty(nTasksPerProc * nproc, dtype=float)
# print(imageNxyz, imageNxyz.prod())

taskIdMin = rank * nTasksPerProc
taskIdMax = min(taskIdMin + nTasksPerProc, nTasks)
nTasks_thisproc = taskIdMax - taskIdMin
sysmatProc = np.zeros(nTasksPerProc)
print("Rank: %5d" % rank, "Working on Tasks ID: ", taskIdMin, " to ", taskIdMax)


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


for taskId in np.arange(taskIdMin, taskIdMax):
    taskId = int(taskId)
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
    geom = sensGeom[int(detGeomId)]

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
    detSubCentroid = np.array(np.meshgrid(x_c, y_c, z_c)).T.reshape(detSubs.prod(), 3)
    # print(detGeomId,imgVoxelId)
    sysmatProc[taskId % nTasksPerProc] = myfunc.PPFDForPair(
        systemGeom, localImgSubCentroids, detSubCentroid, detSubIncrement, geom[7]
    )


comm.Gather(sysmatProc, sysmat, root=0)
comm.Barrier()
if rank == 0:
    output = np.zeros(nTasks, dtype=float)
    output[:] = sysmat[0:nTasks]
    matxymap = sysmat.reshape(int(imageNxyz[0]), int(imageNxyz[1]), sensGeom.shape[0])
    target_c = np.array([
            (sensGeom[0, 0] + sensGeom[0, 1]) * 0.5 + x_shift + trans_x,
            (sensGeom[0, 2] + sensGeom[0, 3]) * 0.5 - y_shift + trans_y,
        ])
    image_c = (imageDims*0.5)[0:1]
    imageNx=imageNxyz[0]
    imageNy=imageNxyz[1]
    multiplexingIndex = myfunc.multiplexingIndex(matxymap[:,:,0],target_c,image_c,imageVxpms,imageNx,imageNy)
    print(multiplexingIndex)
    # # print("sysmat sum: ", np.sum(sysmat))
    # print("\nSaving to file: %s" % yamlConfig["out npz filename"])
    # np.savez_compressed(yamlConfig["out npz filename"], sysmat=output)
MPI.Finalize()
