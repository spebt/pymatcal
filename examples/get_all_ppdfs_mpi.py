import sys

# sys.path.insert(0, "..")
import pymatcal
import numpy as np
import h5py
from mpi4py import MPI
import re

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

configFname = sys.argv[1]
config = pymatcal.get_config(configFname)
outFname = re.match("^(.+).yaml", configFname).group(1) + ".hdf5"

Nfov = np.prod(config["fov nvx"])
Ndet = config["active dets"].shape[0]
Nrot = config["rotation"].shape[0]
Nrshift = config["r shift"].shape[0]
Ntshift = config["t shift"].shape[0]


ntasks = Nfov * Ndet * Nrot * Nrshift * Ntshift
idmap = np.indices((Ntshift, Nrshift, Nrot, Ndet, Nfov)).reshape(5, ntasks).T
fov_subdivs = pymatcal.get_fov_subdivs(config["mmpvx"], config["fov nsub"])

procTaskIds = None
if rank == 0:
    print("Configurations:")
    print("{:30s}{:,}".format("N total tasks:", ntasks))
    print("{:30s}{:,}\n".format("N Process:", nprocs))
    procTaskIds = pymatcal.get_procIds(ntasks, nprocs)

procTaskIds_recv = np.empty(2, dtype=np.uint32)
comm.Scatter(procTaskIds, procTaskIds_recv, root=0)
f = h5py.File(outFname, "w", driver="mpio", comm=MPI.COMM_WORLD)


dset = f.create_dataset("sysmat", (Ntshift, Nrshift, Nrot, Ndet, Nfov), dtype=np.float64)

for idx in range(procTaskIds_recv[0], procTaskIds_recv[1]):
    dset[idmap[idx, 0], idmap[idx, 1], idmap[idx, 2], idmap[idx, 3], idmap[idx, 4]] = (
        pymatcal.get_pair_ppdf(
            idmap[idx, 4],
            idmap[idx, 3],
            idmap[idx, 2],
            idmap[idx, 1],
            idmap[idx, 0],
            fov_subdivs,
            config,
        )
    )
f.close()
