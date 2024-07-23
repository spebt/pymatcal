import sys
sys.path.insert(0, '..')
import pymatcal
import numpy as np
import re


configFname = sys.argv[1]
config = pymatcal.get_config(configFname)
outFname = re.match('^(.+).yml',configFname).group(1)+'.hdf5'
NA = np.prod(config['img nvx'])
NB = config['active dets'].shape[0]
idmap = np.indices((NA, NB)).reshape(2, NA*NB).T
img_subdivs = pymatcal.get_img_subdivs(config['mmpvx'], config['img nsub'])
print(idmap.shape)
# Use 64 processors 
nprocs = 64

ntasks = np.prod(config['img nvx'])*config['active dets'].shape[0]

pymatcal.get_procIds(ntasks, nprocs)
procTaskIds = pymatcal.get_procIds(ntasks, nprocs)


for pId in range(nprocs):
    if pId // 10 == 0 or pId == nprocs - 1:
        print('Process {:<5d} tId_min: {:5,} tId_max: {:5,}'.format(
            pId, procTaskIds[0, pId], procTaskIds[1, pId]))
