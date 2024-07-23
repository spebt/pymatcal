import sys
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import matplotlib as mpl
import pymatcal
import numpy as np
import h5py


# det_idx = int(sys.argv[1])
det_idx = 0
config_fname=sys.argv[2]
data_fname=sys.argv[1]
config = pymatcal.get_config(config_fname)
data = np.zeros((config['active dets'].shape[0], np.prod(config['img nvx'])))
with h5py.File(data_fname, 'r') as f:
    print('Keys: %s' % f.keys())
    data = f['test'][...]
print(data.shape)




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
data_fname_base = data_fname.split('/')[-1].split('.')[0]
out_fname = data_fname_base + '_data_det_%03d.png'%det_idx

# xlim_det = np.array([np.min(xy[:, 0]), np.max(config["det geoms"][:, 1])])
ylim_det = np.array([np.min(xy[:, 1]), np.max(config["det geoms"][:, 3])])
# xlim = np.array([0,np.ceil(xlim_det[1]*0.12)])
ylim = np.array([min(0,ylim_det[0]),max(img_dims[1],ylim_det[1]*1.01)])
# print(xlim)
# print(ylim)
fig, ax = plt.subplots(figsize=(20, 10))
mpl.rcParams.update({'font.size': 16})
# print([(xy,geom) for xy, geom in zip(np.array([act_det_xy[0]]), np.array([config["active dets"][0]]))])
fig.colorbar(ax.imshow(data[det_idx, :].reshape(
    config['img nvx'][0], config['img nvx'][1]), origin='lower'))
ax.add_collection(pc_det)
ax.add_collection(pc_act_det)
# ax.set_xlim(xlim)
fig.suptitle(data_fname+' '*2+'Detector %d'%det_idx)
ax.set_ylim(ylim)
ax.plot(0, 0)
fig.tight_layout()




print('Save figure to: ', out_fname)
fig.savefig(out_fname)
# plt.show()
