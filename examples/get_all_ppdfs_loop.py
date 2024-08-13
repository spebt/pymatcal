import sys
sys.path.insert(0, '..')
import pymatcal
import numpy as np
import h5py
import re


configFname = sys.argv[1]
config = pymatcal.get_config(configFname)
outFname = re.match('^(.+).yml',configFname).group(1)+'.hdf5'
Nfov = np.prod(config["fov nvx"])
Ndet = config["active dets"].shape[0]
Nrot = config["rotation"].shape[0]
Nrshift = config["r shift"].shape[0]
Ntshift = config["t shift"].shape[0]


ntasks = Nfov * Ndet * Nrot * Nrshift * Ntshift
idmap = np.indices((Ntshift, Nrshift, Nrot, Ndet, Nfov)).reshape(5, ntasks).T

fov_subdivs = pymatcal.get_fov_subdivs(config['mmpvx'], config['img nsub'])
f = h5py.File(outFname, "w")
dset = f.create_dataset("sysmat", (Ntshift, Nrshift, Nrot, Ndet, Nfov), dtype=np.float64)
for idx in range(0, ntasks):
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
