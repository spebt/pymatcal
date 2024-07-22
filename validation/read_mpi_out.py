import numpy as np
import h5py
# data = np.fromfile('datafile.noncontig', dtype=np.uint32)
with h5py.File('myfile.hdf5', 'r') as f:
    print('Keys: %s' % f.keys())
    data = f['test'][:]
    print(np.allclose(data[:2500],0))
    print(np.allclose(data[5000:7500],1))
    print(np.allclose(data[2500:5000],2))
    print(np.allclose(data[7500:10000],3))
# print(data.shape)
