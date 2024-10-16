import yaml
import h5py
import numpy as np


def read_dict_from_hdf5(h5group):
    """
    Read a dictionary from an HDF5 file.
    """
    d = {}
    for key in h5group.keys():
        if isinstance(h5group[key], h5py.Group):
            #            print(h5group[key].attrs)
            d[key] = read_dict_from_hdf5(h5group[key])
            # print(d[key].attrs)
        elif isinstance(h5group[key], h5py.Dataset):
            d[key] = h5group[key][:]
            # print(h5group[key])

    for attrkey in h5group.attrs.keys():
        d[attrkey] = h5group.attrs[attrkey]

    return d


h5f = h5py.File("test.hdf5", "r")
configs = read_dict_from_hdf5(h5f["configs"])
print(configs)
h5f.close()
