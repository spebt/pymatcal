import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


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


# with np.load(yamlConfig["out npz filename"]) as data:
#     sysmat = data["sysmat"]
#     # print(sysmat.shape)

with np.load("test.npz") as data:
    sysmat = data["sysmat"]
    print(sysmat.shape)
imageNxyz = imageDims * imageVxpms
nSensDets = sensGeom.shape[0]
# print(imageNxyz,nSensDets)
matxymap = sysmat.reshape(int(imageNxyz[0]),int(imageNxyz[1])).T

print("x_shift + trans_x: ",x_shift + trans_x,"\ntrans_y - y_shift: ",- y_shift + trans_y)

fig, ax = plt.subplots()
imshow=ax.imshow(matxymap, origin="lower")

detectors = np.array(yamlConfig["detector geometry"])

det_xy = np.array(
    [detectors[:, 0] + x_shift + trans_x, detectors[:, 2] - y_shift + trans_y]
).T
det_inc_xy = np.array(
    [(detectors[:, 1] - detectors[:, 0]), (detectors[:, 3] - detectors[:, 2])]
).T

target = sensGeom.flatten()
# print(target)
target_xy = [(target[0] + x_shift + trans_x) , (target[2] - y_shift + trans_y) ]
target_inc_xy = [(target[1] - target[0]), (target[3] - target[2]) ]
# print(target_xy)
target_rect = plt.Rectangle(
    target_xy, target_inc_xy[0], target_inc_xy[1], color="r", ec="none"
)

rect_list = [
    Rectangle(xy, inc_xy[0], inc_xy[1])
    for xy, inc_xy in zip(det_xy, det_inc_xy)
]
# rect_list=[Rectangle((1,1),2,2)]
pc = PatchCollection(rect_list, ec="none")
ax.add_collection(pc)
ax.add_patch(target_rect)
ax.set_xlim(-10,150)
ax.set_ylim(-100,100)
plt.colorbar(imshow)
plt.show()
