import sys
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import matplotlib as mpl
import pymatcal
import numpy as np
import h5py


det_idx = int(sys.argv[1])
config = pymatcal.get_config('configs/all_ppdfs.yml')
data = np.zeros((config['active dets'].shape[0], np.prod(config['img nvx'])))
with h5py.File('myfile.hdf5', 'r') as f:
    print('Keys: %s' % f.keys())
    data = f['test'][...]
print(data.shape)


fig, ax = plt.subplots(figsize=(12, 10))

det_dimy = np.max(config['det geoms'][:, 3]) - \
    np.min(config['det geoms'][:, 2])
img_dims = config['img nvx']*config['mmpvx']
mtrans = pymatcal.get_mtransform(
    -config['angle'], config['dist']+img_dims[0]*0.5, img_dims[1]*0.5-det_dimy*0.5)
xy = pymatcal.coord_transform((mtrans[0], np.array(
    [0, 0, 0])), config['det geoms'][:, (0, 2, 4)]+mtrans[1])

det_list = [
    mpl.patches.Rectangle(xy,
                          geom[1] - geom[0], geom[3] - geom[2], angle=config['angle'],
                          rotation_point=(xy[0], xy[1]))
    for xy, geom in zip(xy, config["det geoms"])
]

act_det_xy = pymatcal.coord_transform((mtrans[0], np.array(
    [0, 0, 0])), config['active dets'][:, (0, 2, 4)]+mtrans[1])



act_det_list = [
    mpl.patches.Rectangle(xy,
                          geom[1] - geom[0], geom[3] - geom[2], angle=config['angle'],
                          rotation_point=(xy[0], xy[1]))
    for xy, geom in zip(np.array([act_det_xy[det_idx]]), np.array([config["active dets"][det_idx]]))
]
pc_det = mpl.collections.PatchCollection(
    det_list, fc=(0.5, 0.5, 0., 1), ec="none", zorder=10
)
pc_act_det = mpl.collections.PatchCollection(
    act_det_list, fc=(1, 0, 0, 1), ec="none", zorder=10
)

print([(xy,geom) for xy, geom in zip(np.array([act_det_xy[0]]), np.array([config["active dets"][0]]))])
fig.colorbar(ax.imshow(data[det_idx, :].reshape(
    config['img nvx'][0], config['img nvx'][1]), origin='lower'))
ax.add_collection(pc_det)
ax.add_collection(pc_act_det)
ax.plot(0, 0)

plt.show()
