# import torch
from torch import empty as empty_tensor, tensor, zeros, Tensor
import time
import h5py
import torch
# torch.set_num_threads(8) # We'll set this based on SLURM_CPUS_PER_TASK
import os

from mpi4py import MPI

# --- Raytracer and other imports ---
# Assuming raytracer_2d is in PYTHONPATH or current directory
from raytracer_2d import (
    set_default_device_as_cpu, # Call this if needed, though PyTorch defaults to CPU
    get_geom_dict,
    get_ppdf,
)

def load_scanner_layouts(filename: str):
    from torch import load as torch_load
    # import os # already imported

    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        raise FileNotFoundError(f"File {filename} does not exist.")
    filename_unique_id = filename.split(".")[0].split("_")[-1]
    # Each rank loads the full layout data. This could be a memory bottleneck
    # if the file is very large and many ranks run on one node.
    # Consider splitting this file per position if it's too big.
    scanner_layouts_data = torch_load(filename, weights_only=True)["layouts"]
    return scanner_layouts_data, filename_unique_id


if __name__ == "__main__":
    import sys

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # PyTorch threading: Use the number of CPUs allocated to this task by SLURM
    # Or default to 1 if not set (e.g., local run without SLURM)
    # Your original script set 8, make sure cpus-per-task in SLURM matches this.
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    
    if rank == 0:
        print(f"MPI initialized with {size} processes.")
        print(f"PyTorch threads per MPI process: {num_threads}")


    fov_dict = {
        "n_pixels": tensor([512, 512]),
        "mm_per_pixel": tensor([0.25, 0.25]),
        "center": tensor([0.0, 0.0]),
    }
    
    filename = "scanner_layouts/scanner_layouts_508640ff8dc59f67dd884fdbc89ee122.tensor"
    
    # Each rank loads the layout data.
    # IMPORTANT: If this file is very large (e.g., many GBs), this strategy might
    # lead to OOM if too many MPI ranks run on the same node.
    # In that case, you MUST split the `scanner_layouts_...tensor` file into
    # per-position files, and each rank loads only the files it needs.
    # For now, we assume it's manageable for each rank to load the whole thing.
    scanner_layouts_data, filename_unique_id = load_scanner_layouts(filename)

    n_total_positions = len(scanner_layouts_data)
    if rank == 0:
        print(f"Total number of positions: {n_total_positions}")

    output_hdf5_dir_base = f"scanner_layouts_raytracer_patched_{filename_unique_id:s}"
    if rank == 0: # Only rank 0 creates the main directory
        if not os.path.exists(output_hdf5_dir_base):
            try:
                os.makedirs(output_hdf5_dir_base)
            except FileExistsError:
                pass # another rank might have created it in a race, fine

    comm.Barrier() # Ensure directory is created before any rank tries to write

    # Distribute layout_idx among MPI processes
    # Each process handles indices: rank, rank + size, rank + 2*size, ...
    layout_indices_for_this_rank = list(range(rank, n_total_positions, size))
    
    if not layout_indices_for_this_rank:
        if rank == 0:
            print(f"Rank {rank} has no positions to process. This might happen if n_total_positions < size.")
    else:
        print(f"Rank {rank} processing positions: {layout_indices_for_this_rank[:3]}... (total {len(layout_indices_for_this_rank)})")


    for layout_idx in layout_indices_for_this_rank:
        print(f"[Rank {rank:03d}] Evaluating position {layout_idx:03d} ...")
        
        position_key = f"position {layout_idx:03d}"
        if position_key not in scanner_layouts_data:
            print(f"[Rank {rank:03d}] ERROR: Position key '{position_key}' not found in scanner_layouts_data. Skipping.")
            continue

        # Load the scanner geometry for the current position
        plate_verts_2d = scanner_layouts_data[position_key]["plate segments"].to("cpu")
        xtal_verts_2d = scanner_layouts_data[position_key]["detector units"].to("cpu")

        n_xtals = xtal_verts_2d.shape[0]
        # print(f"[Rank {rank:03d}] n_xtals for position {layout_idx:03d}: {n_xtals}") # Can be verbose
        
        geom_dict = get_geom_dict(plate_verts_2d, xtal_verts_2d, fov_dict)

        fov_n_pixels = int(fov_dict["n_pixels"].prod())
        # ppdf = empty_tensor(0, fov_n_pixels) # Not needed like this anymore

        elapsed_times_rank = zeros(n_xtals) # For this rank's timings

        output_hdf5_filename = f"{output_hdf5_dir_base}/position_{layout_idx:03d}_ppdfs.hdf5"

        try:
            with h5py.File(output_hdf5_filename, "w") as out_h5file:
                ppdf_dataset = out_h5file.create_dataset(
                    "ppdfs", shape=(n_xtals, fov_n_pixels), dtype="f"
                )
                
                position_start_time = time.time()
                for idx in range(n_xtals):
                    iter_start_time = time.time()
                    try:
                        # Ensure get_ppdf returns a CPU tensor if not already
                        ppdf_row = get_ppdf(idx, geom_dict=geom_dict).cpu().unsqueeze(0).numpy()
                        ppdf_dataset[idx] = ppdf_row
                    except Exception as e:
                        print(f"[Rank {rank:03d}] ID {idx} for position {layout_idx:03d}\nError: {e}")
                        # Decide if you want to sys.exit or continue
                        # For MPI, exiting one rank might hang others. Better to log and continue if possible.
                        # Or implement a collective abort. For now, let's log and potentially skip.
                        # For robustness, fill with NaNs or zeros
                        ppdf_dataset[idx] = zeros((1, fov_n_pixels), dtype="f").numpy() 
                    iter_end_time = time.time()
                    elapsed_times_rank[idx] = iter_end_time - iter_start_time
                position_end_time = time.time()

            avg_time_per_iter = elapsed_times_rank.mean().item() if n_xtals > 0 else 0
            total_time_position = position_end_time - position_start_time
            print(f"[Rank {rank:03d}] Position {layout_idx:03d} finished. Avg time per xtal iter: {avg_time_per_iter:.4f}s. Total for position: {total_time_position:.2f}s")

        except Exception as e:
            print(f"[Rank {rank:03d}] Failed to process or write HDF5 for position {layout_idx:03d}. Error: {e}")
            # Potentially remove partially written file
            if os.path.exists(output_hdf5_filename):
                try:
                    os.remove(output_hdf5_filename)
                    print(f"[Rank {rank:03d}] Removed partially written file: {output_hdf5_filename}")
                except OSError as oe:
                    print(f"[Rank {rank:03d}] Error removing file {output_hdf5_filename}: {oe}")


    if rank == 0:
        print("All assigned positions processed by all ranks.")