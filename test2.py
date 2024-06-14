import pymatcal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
angle_rad = np.pi/6.0
x_shift = 0
y_shift = 0
input_np = np.array([[5, 5, 0]])
# print(pymatcal.coord_transform_v2(angle_rad, x_shift, y_shift, input_np))
mr, mt = pymatcal.get_mtransform(angle_rad, x_shift, y_shift)
# print(pymatcal.coord_transform(mr,mt,input_np))


# randomly generated points
rng = np.random.default_rng()
npx = 4
ydata = rng.integers(low=0, high=180, size=npx)
xdata = rng.integers(low=0, high=180, size=npx)
data1 = np.vstack([xdata, ydata, np.zeros(npx)]).T
# rads1 = np.arctan(data1[:,0]-90)/(data1[:,1]-90)
data2 = pymatcal.coord_transform(
    mr, mt, data1-np.array((90, 90, 0)))+np.array((90, 90, 0))
mr, mt = pymatcal.get_mtransform(0, 20, -30)
data3 = pymatcal.coord_transform(
    mr, mt, data1-np.array((90, 90, 0)))+np.array((90, 90, 0))
fig, axs = plt.subplots(1,2,figsize=(16, 10))
for ax in axs:
    ax.set_xlim(-70, 250)
    ax.set_ylim(-70, 250)
    ax.set_aspect('equal')
axs[0].add_patch(mpl.patches.Polygon(data1[:, 0:2], fc='orange'))
axs[0].add_patch(mpl.patches.Polygon(data2[:, 0:2], alpha=0.5, fc='blue'))
axs[0].plot([data1[0, 0], 90],
        [data1[0, 1], 90], c='k',ls='--')
axs[0].plot([data2[0, 0], 90],
        [data2[0, 1], 90], c='k',ls='--')
axs[0].add_patch(mpl.patches.Circle((90, 90), 2, fc='k'))

axs[1].add_patch(mpl.patches.Polygon(data1[:, 0:2], fc='orange'))
axs[1].add_patch(mpl.patches.Polygon(data3[:, 0:2], alpha=0.5, fc='blue'))
axs[1].plot([data1[0, 0], data3[0,0]],
        [data1[0, 1], data3[0, 1]], c='k',ls='--')
# axs[1].plot([data2[0, 0], 90],
#         [data3[0, 1], 90], c='k',ls='--')
axs[1].add_patch(mpl.patches.Circle((90, 90), 2, fc='k'))
plt.show()
