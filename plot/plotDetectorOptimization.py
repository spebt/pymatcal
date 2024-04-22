import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

configFname = sys.argv[1]

with open(configFname, "r") as file:
    configs = yaml.safe_load(file)
detectorGeoms = np.array(configs["detector geometry"])
# collimatorGeoms = np.array(configs["collimator geometry"])
# geoms = np.concatenate((collimatorGeoms, detectorGeoms), axis=0)
geoms = detectorGeoms
target_indices = np.array(configs["detector"]["sensitive geometry indices"])
detectors = geoms[1:-1]
# print(target_indices)
targets = []
for index in target_indices:
    targets.append(detectors[np.nonzero(detectors[:, 6] == index)].flatten())
targets = np.array(targets)
# print(targets)
geom_xy = np.array([geoms[:, 0], geoms[:, 2]]).T
geom_inc_xy = np.array([(geoms[:, 1] - geoms[:, 0]), (geoms[:, 3] - geoms[:, 2])]).T

targets_xy = np.array([targets[:, 0], targets[:, 2]]).T
targets_inc_xy = np.array(
    [targets[:, 1] - targets[:, 0], targets[:, 3] - targets[:, 2]]
).T
fig, ax = plt.subplots(figsize=(6, 10))
target_rect_list = [
    Rectangle(xy, inc_xy[0], inc_xy[1])
    for xy, inc_xy in zip(targets_xy, targets_inc_xy)
]
rect_list = [
    Rectangle(xy, inc_xy[0], inc_xy[1]) for xy, inc_xy in zip(geom_xy, geom_inc_xy)
]


def collimatorDefine(center1, center2, size1, size2):
    block1 = np.array([[1, 2, 0.5, center1 - size1 * 0.5, -1, 1, 1, 10]])
    block2 = np.array(
        [[1, 2, center1 + size1 * 0.5, center2 - size2 * 0.5, -1, 1, 2, 10]]
    )
    block3 = np.array([[1, 2, center2 + size2 * 0.5, 24.5, -1, 1, 2, 10]])
    return np.concatenate((block1,block2,block3),axis=0)
try:
    center1s = np.array(configs["optimization"]["1st aperture center y (mm)"])
    center2s = np.array(configs["optimization"]["2nd aperture center y (mm)"])
    size1s = np.array(configs["optimization"]["1st aperture size (mm)"])
    size2s = np.array(configs["optimization"]["2nd aperture size (mm)"])
except yaml.YAMLError as err:
    print("Error reading the configurations!", err)
    exit(1)

collimatorGeoms = collimatorDefine(center1s[0],center2s[4],size1s[0],size2s[0])
collimatorGeoms_xy = np.array([collimatorGeoms[:, 0], collimatorGeoms[:, 2]]).T
collimatorGeoms_inc_xy = np.array([(collimatorGeoms[:, 1] - collimatorGeoms[:, 0]), (collimatorGeoms[:, 3] - collimatorGeoms[:, 2])]).T
collimator_list = [
    Rectangle(xy, inc_xy[0], inc_xy[1]) for xy, inc_xy in zip(collimatorGeoms_xy, collimatorGeoms_inc_xy)
]
collimator_pc = PatchCollection(collimator_list, ec="black")

pc = PatchCollection(rect_list, ec="black")
targets_pc = PatchCollection(target_rect_list, ec="black", fc="orange")
ax.add_collection(pc)
ax.add_collection(collimator_pc)
ax.add_collection(targets_pc)
ax.set_xticks(np.arange(1, 14, 3))
ax.set_xticklabels(np.arange(1, 14, 3), size=20)
ax.set_yticks(np.arange(0, 25, 3))
ax.set_yticklabels(np.arange(0, 25, 3), size=20)
ax.set_xlabel("detector X dimension (mm)", size=20)
ax.set_ylabel("detector Y dimension (mm)", size=20)
ax.plot(1, 1)
# ax.grid()
ax.set(aspect="equal")
plt.tight_layout()

outFname = "diagram.png"
print("Read configuration:   ", configFname)
fig.savefig(outFname)
print("The plot is saved as: ", outFname)
