import sys
sys.path.insert(0, '..')
import pymatcal
import numpy as np
import h5py
from mpi4py import MPI
import re


configFname = sys.argv[1]
config = pymatcal.get_config(configFname)
outFname = re.match('^(.+).yml',configFname).group(1)+'.hdf5'
NA = np.prod(config['img nvx'])
NB = config['active dets'].shape[0]
idmap = np.indices((NA, NB)).reshape(2, NA*NB).T
img_subdivs = pymatcal.get_img_subdivs(config['mmpvx'], config['img nsub'])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

ntasks = np.prod(config['img nvx'])*config['active dets'].shape[0]
procTaskIds = None
if rank == 0:
    print('Configurations:')
    print('{:30s}{:,}'.format('N total tasks:', ntasks))
    print('{:30s}{:,}\n'.format('N Process:', nprocs))
    procTaskIds = pymatcal.get_procIds(ntasks, nprocs)
procTaskIds_recv = np.empty(2, dtype=np.uint32)
comm.Scatter(procTaskIds, procTaskIds_recv, root=0)
f = h5py.File(outFname, 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset('test', (NB,NA), dtype=np.float64)

for idx in range(procTaskIds_recv[0],procTaskIds_recv[1]):
    dset[idmap[idx,1],idmap[idx,0]] = pymatcal.get_pair_ppdf(idmap[idx,0],idmap[idx,1],img_subdivs,config)
    # dset[idmap[idx,1],idmap[idx,0]] = 1
f.close()
