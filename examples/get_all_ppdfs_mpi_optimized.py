import re
import sys
import h5py
import numpy as np
from local_functions import get_fov_voxel_center
import pymatcal
from mpi4py import MPI

time_start = MPI.Wtime()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

configFname = sys.argv[1]
config = pymatcal.get_config(configFname)
match = re.match("^(.+?)[.](yaml|yml)$", configFname)
if match is not None:
    outFname = match.group(1) + "_optimized.hdf5"
else:
    raise ValueError("Invalid config file name")

# Use context manager for proper file handling
with h5py.File(outFname, "w", driver="mpio", comm=MPI.COMM_WORLD) as f:
    Nfov = np.prod(config["fov nvx"])
    Ndet = config["active dets"].shape[0]
    Nrot = config["rotation"].shape[0]
    Nrshift = config["r shift"].shape[0]
    Ntshift = config["t shift"].shape[0]
    dset = f.create_dataset("sysmat", 
                           (Ntshift, Nrshift, Nrot, Ndet, Nfov), 
                           dtype=np.float64)

    ntasks = Ndet * Nrot * Nrshift * Ntshift
    idmap = np.indices((Ntshift, Nrshift, Nrot, Ndet)).reshape(4, ntasks).T
    fov_subdivs = pymatcal.get_fov_subdivs(config["mmpvx"], config["fov nsub"])

    procTaskIds = pymatcal.get_procIds(ntasks, nprocs)

    if rank == 0:
        print("Configurations:")
        print("{:30s}{:,}".format("N total tasks:", ntasks))
        print("{:30s}{:,}\n".format("N Process:", nprocs))

    # Process one detector at a time for all FOV voxels
    for idx in range(procTaskIds[0, rank], procTaskIds[1, rank]):
        idt, idr, idrot, idb = idmap[idx]
        
        # Calculate detector parameters once
        det_dimy = np.max(config["det geoms"][:, 3]) - np.min(config["det geoms"][:, 2])
        geomB = config["active dets"][idb]
        det_subdivs = pymatcal.get_det_subdivs(geomB, config["det nsub"])
        geoms = pymatcal.append_subdivs(config["det geoms"], geomB, det_subdivs["geoms"])
        pBs = pymatcal.get_centroids(det_subdivs["geoms"])
        
        # Calculate all FOV voxel centers at once
        fov_ids = np.arange(Nfov, dtype=np.uint64)
        all_pointsA = get_fov_voxel_center(fov_ids, config["fov nvx"], config["mmpvx"])
        
        # Transform all points
        transform_matrix = pymatcal.get_mtransform(
            config["rotation"][idrot],
            -config["r shift"][idr],
            det_dimy * 0.5 - config["t shift"][idt]
        )
        
        all_pAs = pymatcal.coord_transform(
            transform_matrix,
            fov_subdivs["coords"][None, :, :] + all_pointsA[:, None, :]
        )
        
        # Process detector pairs
        abpairs = pymatcal.get_AB_pairs(all_pAs.reshape(-1, 3), pBs)
        ret = pymatcal.get_intersections_2d(geoms, abpairs)
        
        # Calculate final results
        idx_absorp = ret["ts"][:, :, 1] == 1
        idx_attenu = ret["ts"][:, :, 1] != 1
        segs_absorp = np.where(idx_absorp, ret["intersections"], 0)
        segs_attenu = np.where(idx_attenu, ret["intersections"], 0)
        
        # Debug prints to check shapes and values
        print(f"Rank {rank} - Processing index {idx}")
        print("Shapes before calculations:")
        print(f"segs_absorp shape: {segs_absorp.shape}")
        print(f"segs_attenu shape: {segs_attenu.shape}")
        print(f"geoms shape: {geoms.shape}")
        
        subdivs_sa = pymatcal.get_solid_angles(abpairs, det_subdivs["incs"])
        term_attenu = np.exp(-np.sum(segs_attenu.T * geoms[:, 7], axis=1))
        term_absorp = 1 - np.exp(-np.sum(segs_absorp.T * geoms[:, 7], axis=1))
        term_solida = subdivs_sa / (4 * np.pi)
        
        print("Shapes after calculations:")
        print(f"term_attenu shape: {term_attenu.shape}")
        print(f"term_absorp shape: {term_absorp.shape}")
        print(f"term_solida shape: {term_solida.shape}")
        
        # Calculate results per voxel
        intermediate_results = term_attenu * term_absorp * term_solida
        results = intermediate_results.reshape(-1, np.prod(config["fov nsub"])).mean(axis=1)
        
        print(f"Results shape: {results.shape}")
        print(f"Expected Nfov: {Nfov}")
        
        # Store results
        dset[idt, idr, idrot, idb, :] = results

    # Ensure all processes are done before closing
    comm.Barrier()

time_end = MPI.Wtime()
print("Rank:", rank, "elapsed time:", str(time_end - time_start))
