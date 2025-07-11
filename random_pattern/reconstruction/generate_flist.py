if __name__ == "__main__":
    import os
    import numpy as np

    topdir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/scanner_layouts_77faff53af5863ca146878c7c496c75e"
    
    file_idxs = np.arange(0, 24)
    fnames = ["position_{:03d}_ppdfs.hdf5".format(i) for i in file_idxs]
    
    with open("/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/reconstruction/data/dataset_flist.csv", "w") as f:
        for fname in fnames:
            f.write(os.path.join(topdir, fname) + "\n")