import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


pA=[-25, 12, 0]
pB=[12.25, 8.75, 0]

cuboids=np.array([[1, 5, 0.5, 10.75, -1, 1, 1, 10]])
cuboids_xy = np.array([cuboids[:, 0], cuboids[:, 2]]).T
cuboids_inc_xy = np.array(
    [(cuboids[:, 1] - cuboids[:, 0]), (cuboids[:, 3] - cuboids[:, 2])]
).T
rect_list = [
    Rectangle(xy, inc_xy[0], inc_xy[1]) for xy, inc_xy in zip(cuboids_xy, cuboids_inc_xy)
]


fig, ax = plt.subplots(figsize=(10, 6))

ax.set_aspect("equal")
pc = PatchCollection(rect_list, ec="black")
ax.add_collection(pc)
ax.scatter([pA[0], pB[0]], [pA[1], pB[1]], fc="black")
ax.plot([pA[0], pB[0]], [pA[1], pB[1]], color="black")
# ax.scatter(inter_x, inter_y, color="b")

plt.show()