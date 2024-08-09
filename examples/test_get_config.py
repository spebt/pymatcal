import sys,os
import pymatcal




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
    print(config.keys())
