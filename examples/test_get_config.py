import sys, os
import pymatcal
import numpy


# Get config file name from command line
def get_cfname_cmd(sys_argv, default_cfname):
    cfname = ""
    try:
        assert len(sys_argv) > 1, "No configuration file name provided!"
        cfname = sys.argv[1]
        assert os.path.exists(cfname), "Config file does not exist!"
    except Exception as e:
        print("Info:", e, "\nUse default config file name:", default_cfname)
        cfname = default_cfname
    return cfname


if __name__ == "__main__":
    default_cfname = "test_20240806.yml"
    cfname = get_cfname_cmd(sys.argv, default_cfname)
    config = pymatcal.get_config(cfname)
    # print(config.keys())
    Nfov = numpy.prod(config["fov nvx"])
    Ndet = config["active dets"].shape[0]
    Nrot = config["rotation"].shape[0]
    Nrshift = config["r shift"].shape[0]
    Ntshift = config["t shift"].shape[0]

    print("Configurations:")
    print("N fov voxels:    ", Nfov)
    print("N detectors:     ", Ndet)
    print("rotation:        ", config["rotation"])
    print("radial shift:    ", config["r shift"])
    print("tangential shift:", config["t shift"])

    ntasks = Nfov * Ndet * Nrot * Nrshift * Ntshift
    idmap = numpy.indices((Nrshift, Ntshift, Nrot, Ndet, Nfov)).reshape(5, ntasks).T
    print("N total tasks:", ntasks)
    print("idmap shape:", idmap.shape)
    idx = int(sys.argv[2])
    print("idmap[ %i ]:" % idx, idmap[idx])
    print('idmap elements upper bounds: [',
        numpy.max(idmap[:, 0]) + 1,
        numpy.max(idmap[:, 1]) + 1,
        numpy.max(idmap[:, 2]) + 1,
        numpy.max(idmap[:, 3]) + 1,
        numpy.max(idmap[:, 4]) + 1,
        ']'
    )
