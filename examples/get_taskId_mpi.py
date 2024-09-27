import re
import sys

import h5py
import numpy as np
import pymatcal
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

configFname = sys.argv[1]
config = pymatcal.get_config(configFname)
match = re.match("^(.+?)[.](yaml|yml)$", configFname)
if match is not None:
    outFname = match.group(1) + ".hdf5"
else:
    raise ValueError("Invalid config file name")

f = h5py.File(outFname, "w", driver="mpio", comm=MPI.COMM_WORLD)

Nfov = np.prod(config["fov nvx"])
Ndet = config["active dets"].shape[0]
Nrot = config["rotation"].shape[0]
Nrshift = config["r shift"].shape[0]
Ntshift = config["t shift"].shape[0]
dset = f.create_dataset(
    "sysmat", (Ntshift, Nrshift, Nrot, Ndet, Nfov), dtype=np.float64
)

ntasks = Nfov * Ndet * Nrot * Nrshift * Ntshift
idmap = np.indices((Ntshift, Nrshift, Nrot, Ndet, Nfov)).reshape(5, ntasks).T
fov_subdivs = pymatcal.get_fov_subdivs(config["mmpvx"], config["fov nsub"])

procTaskIds = pymatcal.get_procIds(ntasks, nprocs)

if rank == 0:
    print("Configurations:")
    print("{:30s}{:,}".format("N total tasks:", ntasks))
    print("{:30s}{:,}\n".format("N Process:", nprocs))


for idx in range(procTaskIds[0, rank], procTaskIds[1, rank]):
    dset[idmap[idx, 0], idmap[idx, 1], idmap[idx, 2], idmap[idx, 3], idmap[idx, 4]] = (
        idx
    )
f.close()
