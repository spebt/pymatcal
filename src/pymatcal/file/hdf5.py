import yaml
import h5py

def save_dict2hdf5(d, h5f, group="configs"):
    h5group = h5f.create_group(group)
    for key in d.keys():
        if isinstance(d[key], dict):
            save_dict2hdf5(d[key], h5f, group + "/" + key)
        elif isinstance(d[key], str):
            h5group.attrs[key] = d[key]
        else:
            h5group.create_dataset(key, data=d[key])