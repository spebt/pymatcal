# Remember to `module load mpi` in Fedora
import h5py
import sys
sys.path.insert(0, '..')
import pymatcal
import numpy as np
from mpi4py import MPI
import yaml



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
config = pymatcal.get_config('configs/mpi_sequence.yml')
ntasks = np.prod(config['img nvx'])*config['active dets'].shape[0]
procTaskIds = None
if rank == 0:
    print('Configurations:')
    print('{:30s}{:,}'.format('N total tasks:', ntasks))
    print('{:30s}{:,}\n'.format('N Process:', nprocs))
    procTaskIds = pymatcal.get_procIds(ntasks, nprocs)
procTaskIds_recv = np.empty(2, dtype=np.uint32)
comm.Scatter(procTaskIds, procTaskIds_recv, root=0)
print('Rank:', rank, procTaskIds_recv)

f = h5py.File('myfile.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset('test', (ntasks,), dtype=np.float64)
dset[procTaskIds_recv[0]:procTaskIds_recv[1]]= rank
f.close()
# MPI.Finalize()
