import sys
sys.path.insert(0, '..')
import pymatcal
import numpy as np
import h5py
import re


configFname = sys.argv[1]
config = pymatcal.get_config(configFname)
outFname = re.match('^(.+).yml',configFname).group(1)+'.hdf5'
NA = np.prod(config['img nvx'])
NB = config['active dets'].shape[0]
NR = 1
if 'rotations' in config:
    NR = config['rotations'].shape[0]

idmap = np.indices((NA, NB)).reshape(2, NA*NB).T
img_subdivs = pymatcal.get_img_subdivs(config['mmpvx'], config['img nsub'])
ntasks = np.prod(config['img nvx'])*config['active dets'].shape[0]
# f = h5py.File(outFname, 'w')
# dset = f.create_dataset('test', (NB,NA), dtype=np.float64)
# for idx in range(0,ntasks):
#     dset[idmap[idx,1],idmap[idx,0]] = pymatcal.get_pair_ppdf(idmap[idx,0],idmap[idx,1],img_subdivs,config)
#     # dset[idmap[idx,1],idmap[idx,0]] = 1
# f.close()
