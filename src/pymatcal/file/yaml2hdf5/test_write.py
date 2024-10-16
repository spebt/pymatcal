import yaml
import h5py

config = yaml.safe_load(open("./panel_all_debug.yaml"))


def print_dict_keys(d, indent=0):
    for key in d.keys():
        print("  " * indent, key)
        if isinstance(d[key], dict):
            print_dict_keys(d[key], indent + 1)


def save_dict2hdf5(d, h5f, group="configs"):
    h5group = h5f.create_group(group)
    for key in d.keys():
        if isinstance(d[key], dict):
            save_dict2hdf5(d[key], h5f, group + "/" + key)
        elif isinstance(d[key], str):
            h5group.attrs[key] = d[key]
        else:
            h5group.create_dataset(key, data=d[key])


h5f = h5py.File("test.hdf5", "w")
save_dict2hdf5(config, h5f, group="configs")
h5f.close()
