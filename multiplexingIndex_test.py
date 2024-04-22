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

# Test the multiplexing index function
identifier="subdiv-9x9x1"
npzFname="data/"+identifier+".npz"
with np.load(npzFname) as data:
    sysmat = data["sysmat"]
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