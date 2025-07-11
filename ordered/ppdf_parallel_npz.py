import torch
import time
import h5py
import os
from mpi4py import MPI

from rich.progress import Progress, BarColumn
from rich.console import Console
from raytracer_2d import (
    load_scanner_geometry_npz,
    set_default_device_as_cpu,
    get_geom_dict,
    get_ppdf,
)

set_default_device_as_cpu()

fov_dict = {
    "n_pixels": torch.tensor([512, 512]),
    "mm_per_pixel": torch.tensor([0.25, 0.25]),
    "center": torch.tensor([0.0, 0.0]),
}

scanner_geometry_dir = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/scanner_cuboids_data"
file_paths = [
    os.path.join(scanner_geometry_dir, f"scanner_cuboids_{i:03d}.npz")
    for i in range(24)
]

output_dir = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/sysmats"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving HDF5 output files to: {output_dir}")


# Precompute total xtals per file
xtal_counts = []
total_xtals = 0
for path in file_paths:
    _, xtal_verts = load_scanner_geometry_npz(path)
    count = xtal_verts.shape[0]
    xtal_counts.append(count)
    total_xtals += count

fov_n_pixels = int(torch.prod(fov_dict["n_pixels"]))

def compute_range_bounds():
    """Compute row ranges for each file's xtals in final HDF5 dataset"""
    starts = [0]
    for count in xtal_counts[:-1]:
        starts.append(starts[-1] + count)
    ends = [s + c for s, c in zip(starts, xtal_counts)]
    return list(zip(starts, ends))

ranges = compute_range_bounds()


def worker(proc_id, file_indices):
    # Get MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each process works on a subset of files based on rank
    for idx in file_indices:
        filename = os.path.join(output_dir, f"scanner_ppdfs_{idx}.hdf5")
        
        # Open HDF5 file in append mode with MPI
        with h5py.File(filename, "w") as h5f:
            # Create dataset for each worker (this will be for the specific file geometry)
            xtal_count = xtal_counts[idx]
            h5f.create_dataset("ppdfs", shape=(xtal_count, fov_n_pixels), dtype="f")
            dataset = h5f["ppdfs"]

            # Load geometry and compute PPDF for each crystal
            plate_verts_2d, xtal_verts_2d = load_scanner_geometry_npz(file_paths[idx])
            geom_dict = get_geom_dict(plate_verts_2d, xtal_verts_2d, fov_dict)

            for i in range(xtal_verts_2d.shape[0]):
                try:
                    ppdf = get_ppdf(i, geom_dict=geom_dict).unsqueeze(0).numpy()
                    dataset[i] = ppdf  # Store PPDF in the respective HDF5 file
                    print(f"[Rank {rank}] File {idx}, xtal {i} done.")
                except Exception as e:
                    print(f"[Rank {rank}] Error on file {idx}, xtal {i}: {e}")


if __name__ == "__main__":
    # Get MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split the files evenly between the processes
    file_idx_chunks = [list(range(24))[i::size] for i in range(size)]
    
    # Each process runs on its chunk of files
    worker(rank, file_idx_chunks[rank])

    print("All processes completed.")

