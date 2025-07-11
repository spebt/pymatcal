import numpy as np
import torch
import os
import h5py
import time
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        flist = f.readlines()
        flist = [f.strip() for f in flist]
        return flist

if __name__ == "__main__":
    # --- 1. Configuration ---
    # Use absolute paths for clarity
    base_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/reconstruction/data"
    flist_path = os.path.join(base_dir, "dataset_flist.csv") # Corrected typo .csv
    projs_path = os.path.join(base_dir, "derenzo-projs.npy")
    output_path = os.path.join(base_dir, "recon_mlem_torch_derenzo.npz")
    
    # Correct and consistent dimensions
    IMG_DIM = 512
    SFOV = IMG_DIM * IMG_DIM
    SPROJ = 726
    
    # Reconstruction parameters
    N_ITERATIONS = 100
    CONVERGENCE_TOLERANCE = 1e-4 # Stop if change is very small

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    print("Loading data...")
    flist = get_flist(flist_path)
    # Projection data is small enough to load fully into memory
    pdata_full = torch.from_numpy(np.load(projs_path)).to(device)

    # --- 3. Initialization ---
    estimate = torch.ones(SFOV, device=device, dtype=torch.float32)
    
    estimates_history = []
    times_history = []
    log_p_history = []

    # --- 4. Main MLEM Reconstruction Loop ---
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        main_task = progress.add_task("[green]MLEM Iterations", total=N_ITERATIONS)
        
        for it in range(N_ITERATIONS):
            iteration_start_time = time.time()
            estimate_prev = estimate.clone()

            # Initialize accumulators for this iteration
            back_projection = torch.zeros(SFOV, device=device, dtype=torch.float32)
            sensitivity_map = torch.zeros(SFOV, device=device, dtype=torch.float32)

            # Inner loop to process matrix files in batches
            inner_task = progress.add_task(f"[cyan]  Iter {it+1}/{N_ITERATIONS}", total=len(flist), transient=True)
            for i, fname in enumerate(flist):
                with h5py.File(fname, "r") as h5f:
                    m_chunk = torch.from_numpy(h5f["ppdfs"][:]).view(1, SPROJ, SFOV).to(device)
                
                p_chunk = pdata_full[i]

                # --- MLEM STEPS FOR ONE CHUNK ---
                # Forward project
                y_chunk = torch.matmul(m_chunk, estimate) # Shape: (1, 726)

                # Avoid division by zero
                y_chunk[y_chunk == 0] = 1.0
                
                # Ratio calculation
                # Ensure p_chunk has the same 2D shape as y_chunk for clean division
                r_chunk = p_chunk.view(1, -1) / y_chunk # Shape: (1, 726)
                
                # Backproject and accumulate
                # Add a dimension to r_chunk to make it a column vector for matmul
                # (1, 726) -> (1, 726, 1)
                r_chunk_vec = r_chunk.unsqueeze(-1) 
                # matmul: (1, 262144, 726) @ (1, 726, 1) -> (1, 262144, 1)
                back_projection_chunk = torch.matmul(m_chunk.transpose(1, 2), r_chunk_vec)
                # Add the result to the accumulator, squeezing out the unnecessary dimensions
                back_projection += back_projection_chunk.squeeze()
                
                # Accumulate sensitivity map
                sensitivity_map += torch.sum(m_chunk, dim=1).squeeze()
                
                progress.update(inner_task, advance=1)
            
            # --- UPDATE STEP AFTER PROCESSING ALL CHUNKS ---
            # Avoid division by zero in sensitivity map
            sensitivity_map[sensitivity_map == 0] = 1.0
            
            # Update the estimate
            estimate = estimate * (back_projection / sensitivity_map)
            
            # --- Store results and check for convergence ---
            iteration_time = time.time() - iteration_start_time
            times_history.append(iteration_time)
            # Store every 5th estimate to save space
            if it % 5 == 0:
                estimates_history.append(estimate.view(IMG_DIM, IMG_DIM).cpu().numpy())

            # Check for convergence
            diff = torch.norm(estimate - estimate_prev) / torch.norm(estimate_prev)
            if diff < CONVERGENCE_TOLERANCE:
                print(f"\nConvergence reached at iteration {it+1} (difference: {diff:.2e}). Stopping.")
                progress.update(main_task, completed=N_ITERATIONS) # Mark as complete
                break
                
            progress.update(main_task, advance=1, description=f"[green]MLEM Iterations (diff: {diff:.2e})")

    # --- 5. Save Final Results ---
    print("\nReconstruction complete. Saving results...")
    np.savez_compressed(
        output_path,
        estimates=np.array(estimates_history),
        times=np.array(times_history),
    )
    print(f"Results saved to: {output_path}")
