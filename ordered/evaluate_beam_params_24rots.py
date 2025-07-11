# Filename: evaluate_beam_params_24rots_v2.py

import numpy as np
import h5py
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    Column,
    TimeElapsedColumn, # Added for timing info
)
import matplotlib.pyplot as plt
import skimage as ski
import scipy
import os

FOV_X_PIXELS = 512
FOV_Y_PIXELS = 512
EXPECTED_DETECTORS_PER_FILE = 864 

def get_output_filename(ifname: str, suffix: str, ext: str, out_dir: str):
    """Generates the output filename in the specified output directory."""
    base_fname = os.path.basename(ifname)
    fname_wo_ext = os.path.splitext(base_fname)[0]
    # Construct new filename in the output directory
    return os.path.join(out_dir, fname_wo_ext + suffix + "." + ext)

def get_beam_params(fname: str, output_dir: str, pbar: Progress, task_id_detectors: int):
    """
    Calculates beam parameters (slope, intercept, r, length, T_sum)
    from flat PPDFs and saves them along with the filtered PPDF mask (3D).
    """
    fov_pixels_flat = FOV_X_PIXELS * FOV_Y_PIXELS

    try:
        with h5py.File(fname, "r") as f:
            ppdfs_flat = np.copy(f["ppdfs"][:]) # Shape (n_detectors, fov_pixels_flat)
            n_detectors_in_file = ppdfs_flat.shape[0]

            if n_detectors_in_file != EXPECTED_DETECTORS_PER_FILE:
                 print(f"\nWarning: File {fname} has {n_detectors_in_file} detectors, expected {EXPECTED_DETECTORS_PER_FILE}. Proceeding.")
            if ppdfs_flat.shape[1] != fov_pixels_flat:
                raise ValueError(f"Flat PPDF size mismatch in {fname}. Expected {fov_pixels_flat}, got {ppdfs_flat.shape[1]}")

    except Exception as e:
        print(f"\nError opening or reading {fname}: {e}")
        pbar.update(task_id_detectors, advance=EXPECTED_DETECTORS_PER_FILE) # Advance the task fully on error
        return

    # Initialize output arrays
    # Params: [slope, intercept, r, length_pixels, t_sum]
    all_params = np.zeros((n_detectors_in_file, 15, 5))
    # Store filtered PPDFs as 3D: (n_detectors, FOV_Y, FOV_X)
    filtered_ppdfs_3d = np.zeros((n_detectors_in_file, FOV_Y_PIXELS, FOV_X_PIXELS), dtype=np.uint8) # Use uint8 for labels
    n_strips_per_ppdf = np.zeros(n_detectors_in_file, dtype=int) # Shape (n_detectors_in_file,)

    for idp in range(n_detectors_in_file):
        ppdf = ppdfs_flat[idp].reshape(FOV_Y_PIXELS, FOV_X_PIXELS)
        image = ppdf.T # Transpose for easier row/column access (y -> row, x -> col) -> shape (FOV_X, FOV_Y)
        try:
            if np.all(np.isclose(ppdf, ppdf[0,0])): # Handle constant images
                 thresh = ppdf[0,0] + 1e-9 # Add small epsilon
            else:
                 thresh = ski.filters.threshold_li(image)

            objects = ski.measure.label(image > thresh, connectivity=2)
            large_objects = objects
            unique_lables_initial = np.unique(objects)

            if len(unique_lables_initial) > 2:
                large_objects = ski.morphology.remove_small_objects(objects, min_size=24) # Min size might need adjustment for 512x512

            unique_lables = np.unique(large_objects)
            unique_lables = unique_lables[np.nonzero(unique_lables)]
            n_strips_per_ppdf[idp] = len(unique_lables)

            # Create the filtered PPDF image (mask) for this detector
            ppdf_filt_single = np.zeros_like(large_objects, dtype=np.uint8) 
            max_strips_to_process = min(len(unique_lables), 15) # Limited to 15 strips, can modify if we want. 

            if len(unique_lables) > 15:
                print(f"Warning: Detector {idp} in {os.path.basename(fname)} has {len(unique_lables)} strips. Processing first 15.")
                n_strips_per_ppdf[idp] = 15 # Cap the count

            for idl in range(max_strips_to_process):
                label = unique_lables[idl]
                ppdf_filt_single[large_objects == label] = idl + 1

            filtered_ppdfs_3d[idp] = ppdf_filt_single.T

        except Exception as e:
            print(f"\nError during image processing for detector {idp} in {os.path.basename(fname)}: {e}")
            pbar.update(task_id_detectors, advance=1)
            continue # Skip param extraction for this detector

        # --- Parameter Extraction per Beam ---
        for idl in range(max_strips_to_process):
            # Get coordinates (y_row, x_col) from the *transposed* mask
            y_coords_transposed, x_coords_transposed = np.where(ppdf_filt_single == idl + 1)

            # Calculate T_sum (Sum of original 2D PPDF values within the mask)
            # Use original (non-transposed) PPDF coordinates
            t_sum = np.sum(ppdf[x_coords_transposed, y_coords_transposed]) # PPDF is (FOV_Y, FOV_X)

            # Calculate Centerline using coordinates from transposed mask (y_row, x_col)
            length_pixels = 0
            slope = 0
            intercept = 0
            r = 0

            if len(x_coords_transposed) < 2 or len(y_coords_transposed) < 2: # Need at least 2 points
                 all_params[idp, idl] = np.array([slope, intercept, r, length_pixels, t_sum])
                 continue

            # Determine dominant direction based on coordinates from transposed mask
            unique_x_t, _ = np.unique(x_coords_transposed, return_inverse=True) # Unique columns (original X)
            unique_y_t, _ = np.unique(y_coords_transposed, return_inverse=True) # Unique rows (original Y)
            n_unique_x_t = unique_x_t.shape[0]
            n_unique_y_t = unique_y_t.shape[0]

            x_c, y_c = [], [] # Store centerline points IN ORIGINAL PPDF COORDINATE SYSTEM (x, y)

            if n_unique_x_t >= n_unique_y_t: # More spread in x (cols) in transposed -> closer to horizontal in original PPDF
                x_c = unique_x_t # Original X coordinates
                y_c = np.zeros_like(x_c, dtype=float)
                for idx, ux in enumerate(unique_x_t):
                    # Average y_row values (original Y) for this x_col (original X)
                    y_c[idx] = np.mean(y_coords_transposed[x_coords_transposed == ux])
                primary_coord = x_c
            else: # More spread in y (rows) in transposed -> closer to vertical in original PPDF
                y_c = unique_y_t # Original Y coordinates
                x_c = np.zeros_like(y_c, dtype=float)
                for idx, uy in enumerate(unique_y_t):
                    # Average x_col values (original X) for this y_row (original Y)
                    x_c[idx] = np.mean(x_coords_transposed[y_coords_transposed == uy])
                primary_coord = y_c

            # Calculate Length from untrimmed centerline (using original PPDF coords x_c, y_c)
            if len(primary_coord) > 1:
                # Ensure calculation uses coordinates in the same system
                length_pixels = np.sqrt((x_c[-1] - x_c[0])**2 + (y_c[-1] - y_c[0])**2)
            else:
                length_pixels = 0

            # Trim centerline for robust slope fitting
            ntrim = int(np.floor(len(primary_coord) * 0.03))
            ntrim = np.max([ntrim, 1])

            if len(primary_coord) > 2 * ntrim + 1: # Need at least 2 points after trimming
                x_c_trimmed = x_c[ntrim:-ntrim]
                y_c_trimmed = y_c[ntrim:-ntrim]

                x_c_shifted = x_c_trimmed + 0.5 # Shift original X
                y_c_shifted = y_c_trimmed + 0.5 # Shift original Y

                # Linear Regression (x = slope * y + intercept) using original PPDF coordinates
                if np.all(np.isclose(x_c_shifted, x_c_shifted[0])): # Vertical line in original PPDF (x=const)
                    slope = np.inf # dx/dy = inf
                    intercept = x_c_shifted[0] # x = intercept
                    r = 1
                elif np.all(np.isclose(y_c_shifted, y_c_shifted[0])): # Horizontal line in original PPDF (y=const)
                    slope = 0 # dx/dy = 0
                    intercept = y_c_shifted[0] # y = intercept (regression is x on y) -> need intercept of x
                    # Let's re-evaluate intercept for slope=0 case:
                    # If slope=0, regression x = 0*y + intercept_x -> intercept_x = mean(x)
                    intercept = np.mean(x_c_shifted)
                    r = 1
                else:
                    try:
                        # Perform x on y regression
                        slope, intercept, r, p, se = scipy.stats.linregress(y_c_shifted, x_c_shifted)
                    except ValueError:
                        slope, intercept, r = 0, 0, 0
            else: # Not enough points after trimming
                slope, intercept, r = 0, 0, 0

            if np.abs(r) >= 0.75:
                all_params[idp, idl] = np.array([slope, intercept, r, length_pixels, t_sum])
            else:
                 # Store T_sum and length even if fit is poor
                 all_params[idp, idl] = np.array([0, 0, 0, length_pixels, t_sum])

        pbar.update(task_id_detectors, advance=1)

    dict_to_save = {
        "beam params": all_params,        # Shape (n_detectors, 15, 5)
        "filtered ppdfs": filtered_ppdfs_3d, # Shape (n_detectors, FOV_Y, FOV_X)
        "n strips per ppdf": n_strips_per_ppdf # Shape (n_detectors,)
    }
    ofname = get_output_filename(fname, "_beam_params", "npz", output_dir)
    np.savez_compressed(ofname, **dict_to_save)
    print(f"Saved beam parameters to: {ofname}")
    print(f'{"Max n_strips_per_ppdf":40s}: {np.max(n_strips_per_ppdf)} for {os.path.basename(fname)}')


if __name__ == "__main__":
    input_dir = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/sysmats"  # Directory containing system matrix files
    output_dir = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/npzs" # Directory to save _beam_params.npz files
    os.makedirs(output_dir, exist_ok=True)

    pbar = Progress(
        SpinnerColumn(),
        BarColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TextColumn("[{task.completed} of {task.total}]"),
        TimeElapsedColumn(),
        console=None,
    )

    n_rotations = 24
    task_rotations = pbar.add_task("[cyan]Processing rotations...", total=n_rotations, completed=0)
    task_detectors = pbar.add_task("[green]Processing PPDFs.....", total=EXPECTED_DETECTORS_PER_FILE, completed=0)

    with pbar:
        for i in range(n_rotations):
            pbar.reset(task_detectors, description="[green]Processing PPDFs.....")
            infname = os.path.join(input_dir, f"scanner_ppdfs_{i:02d}.hdf5")

            if not os.path.exists(infname):
                print(f"File {infname} not found, skipping rotation {i}.")
                pbar.update(task_detectors, advance=EXPECTED_DETECTORS_PER_FILE, description="[red]File Not Found! Skip")
                pbar.update(task_rotations, advance=1)
                continue

            print(f"\nProcessing {infname}...")
            get_beam_params(infname, output_dir, pbar, task_detectors)

            pbar.update(task_rotations, advance=1)
    print("\nBeam parameter extraction complete.")