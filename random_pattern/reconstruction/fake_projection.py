import numpy as np
import time
import torch
import os
import h5py
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

# (get_flist function can remain the same)
def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        flist = f.readlines()
        flist = [f.strip() for f in flist]
        return flist

# We don't need get_matrix anymore, as we'll load one file at a time.

if __name__ == "__main__":
    
    # --- Setup ---
    torch.device("cpu")
    data_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/reconstruction/data"
    flist = get_flist(os.path.join(data_dir, "dataset_flist.csv"))
    
    # These define the EXPECTED dimensions from the system matrix files
    sfov_expected = 512 * 512
    sproj = 726

    # --- Phantom Loading and Resizing ---
    # ERROR 1 FIX: Use torch.load and the correct, direct path.
    phantom_filename = "/vscratch/grp-rutaoyao/Harsh/phantoms/hot_rods_phantom_32.0_mm_x_32.0_mm.pt"
    phantom_data = torch.load(phantom_filename)
    phantom_tensor = phantom_data["Phantom tensor"]

    # ERROR 2 FIX: The phantom (256x256) must be padded to match the matrix (512x512).
    # We will pad it with zeros to place it in the center of a 512x512 canvas.
    h, w = phantom_tensor.shape
    pad_h = (512 - h) // 2
    pad_w = (512 - w) // 2
    # The padding format is (pad_left, pad_right, pad_top, pad_bottom)
    phantom_padded = torch.nn.functional.pad(phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
    
    # Flatten the final, correctly-sized phantom
    phantom_flat = phantom_padded.view(-1)
    
    print(f"Original phantom shape: {phantom_tensor.shape}")
    print(f"Padded phantom shape:   {phantom_padded.shape}")
    if phantom_flat.shape[0] != sfov_expected:
        raise ValueError("FATAL: Padded phantom size does not match expected system matrix FOV.")

    # --- Batch Processing ---
    # MEMORY FIX: Process one system matrix file at a time instead of loading all at once.
    all_projs = []

    # Setup a nice progress bar
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("[green]Projecting...", total=len(flist))
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                # Load one chunk of the matrix
                matrix_chunk = torch.tensor(h5f["ppdfs"][:]).view(1, sproj, sfov_expected)
                
                # Perform matrix multiplication on just this chunk
                proj_chunk = torch.matmul(matrix_chunk, phantom_flat)
                all_projs.append(proj_chunk)
            
            progress.update(task, advance=1)

    # Combine the results from all the chunks
    final_projs = torch.cat(all_projs, dim=0)

    # --- Save the final result ---
    output_path = os.path.join(data_dir, "derenzo-projs.npy")
    np.save(output_path, final_projs.numpy())
    
    print("\nProjection complete!")
    print(f"Final projection shape: {final_projs.shape}")
    print(f"Saved projections to: {output_path}")