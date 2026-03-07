import numpy as np
import scipy.sparse as sp
import h5py
import matplotlib.pyplot as plt
import os

def load_ppdfs_for_layout(ppdf_dir: str, layout_idx: int, n_xtals_to_load: int = -1):
    """Loads PPDFs from an HDF5 file for a specific layout. Auto-detects Sparse or Dense."""
    ppdf_filename = os.path.join(ppdf_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
    if not os.path.exists(ppdf_filename):
        print(f"PPDF file {ppdf_filename} does not exist.")
        return None
    
    with h5py.File(ppdf_filename, "r") as f:
        # Check if the file uses the new Sparse (CSR) format
        if "data" in f and "indices" in f and "indptr" in f:
            data = f["data"][:]
            indices = f["indices"][:]
            indptr = f["indptr"][:]
            shape = tuple(f.attrs["shape"])
            
            # Reconstruct the sparse matrix
            csr_mat = sp.csr_matrix((data, indices, indptr), shape=shape)
            
            # Slice crystals if needed
            if n_xtals_to_load != -1:
                csr_mat = csr_mat[:n_xtals_to_load, :]
            return csr_mat
            
        # Fallback to the old Dense format
        elif "ppdfs" in f:
            ppdfs_data = f["ppdfs"][:n_xtals_to_load] if n_xtals_to_load != -1 else f["ppdfs"][:]
            return ppdfs_data
            
        else:
            print(f"Error: Unknown data format in {ppdf_filename}")
            return None

def create_and_plot_sensitivity_histogram(sensitivity_map: np.ndarray, output_dir: str, file_suffix: str, title_suffix: str):
    """Generates and displays a histogram for a given sensitivity map."""
    fov_values = sensitivity_map[sensitivity_map > 0].flatten()
    if fov_values.size == 0:
        print("Warning: The sensitivity map contains no non-zero values.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(fov_values, bins=100, color='royalblue', alpha=0.75)
    plt.title(f'Histogram of Sensitivity Values - {title_suffix}')
    plt.xlabel('Sensitivity')
    plt.ylabel('Number of Pixels (Frequency)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    mean_sensitivity = np.mean(fov_values)
    plt.axvline(mean_sensitivity, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sensitivity:.4f}')
    plt.legend()
    plt.tight_layout()

    hist_filename = os.path.join(output_dir, f"sensitivity_histogram_{file_suffix}.png")
    plt.savefig(hist_filename)
    print(f"Saved sensitivity histogram to: {hist_filename}")
    plt.close()


if __name__ == "__main__":
    # --- Configuration ---
    ppdf_files_base_dir = "/vscratch/grp-rutaoyao/sid/data/mph_hourglass_single_position_base_2mm_18pinholes_rotated_elliptical_comp2/outputs/"
    num_layouts_to_load = 40
    layout_indices = list(range(num_layouts_to_load))

    FOV_PIXELS_X, FOV_PIXELS_Y = 512, 512
    MM_PER_PIXEL_X, MM_PER_PIXEL_Y = 0.25, 0.25
    num_crystals_to_sum = -1
    # --- End Configuration ---

    if not os.path.isdir(ppdf_files_base_dir):
        print(f"Error: PPDF directory '{ppdf_files_base_dir}' not found.")
        exit()

    print(f"Loading and aggregating PPDFs for layouts: {layout_indices}...")
    aggregated_ppdfs = None
    successful_loads = 0 

    for idx in layout_indices:
        print(f"  - Loading layout {idx:03d}...")
        ppdfs_for_layout = load_ppdfs_for_layout(ppdf_files_base_dir, idx, num_crystals_to_sum)
        
        if ppdfs_for_layout is None:
            print(f"Warning: Could not load data for layout {idx}. Skipping.")
            continue
            
        if aggregated_ppdfs is None:
            # Copy sparse matrix or cast dense to float32
            if sp.issparse(ppdfs_for_layout):
                aggregated_ppdfs = ppdfs_for_layout.copy()
            else:
                aggregated_ppdfs = ppdfs_for_layout.astype(np.float32)
        else:
            # Handles sparse + sparse, dense + dense, and mixed automatically
            aggregated_ppdfs += ppdfs_for_layout
        
        successful_loads += 1

    if aggregated_ppdfs is None:
        print("Error: No PPDF data was successfully loaded. Exiting.")
        exit()
    
    print(f"\nAggregation complete. Successfully loaded {successful_loads} layouts.")

    # --- Generate and NORMALIZE the sensitivity map ---
    if successful_loads > 0:
        # Summing logic depends on whether the final accumulated object is sparse or dense
        if sp.issparse(aggregated_ppdfs):
            # .sum(axis=0) on CSR returns a 1D np.matrix, use np.asarray().flatten() to make it standard
            sensitivity_map_1d = np.asarray(aggregated_ppdfs.sum(axis=0)).flatten() / successful_loads
        else:
            sensitivity_map_1d = np.sum(aggregated_ppdfs, axis=0) / successful_loads
    else:
        sensitivity_map_1d = np.zeros(FOV_PIXELS_X * FOV_PIXELS_Y)

    sensitivity_map_2d = sensitivity_map_1d.reshape((FOV_PIXELS_Y, FOV_PIXELS_X))
            
    # Determine plot extent
    extent = [-(MM_PER_PIXEL_X * FOV_PIXELS_X / 2), (MM_PER_PIXEL_X * FOV_PIXELS_X / 2),
              -(MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2), (MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2)]

    layouts_str_fname = f"aggregated_{successful_loads}layouts_normalized"
    layouts_str_title = f"Normalized Sensitivity Map ({successful_loads} Layouts)"

    # Plot and Save
    plt.figure(figsize=(8, 7))
    plt.imshow(sensitivity_map_2d, cmap='viridis', origin='lower', extent=extent)
    plt.colorbar(label='Average Sensitivity (Normalized)')
    plt.title(layouts_str_title)
    plt.xlabel('X (mm)'); plt.ylabel('Y (mm)')
    plt.axhline(0, color='white', linestyle=':', lw=0.5); plt.axvline(0, color='white', linestyle=':', lw=0.5)
    plt.tight_layout()
    map_filename = os.path.join(ppdf_files_base_dir, f"sensitivity_map_{layouts_str_fname}.png")
    plt.savefig(map_filename)
    print(f"\nSaved normalized sensitivity map to: {map_filename}")
    plt.close()
    
    # Generate histogram
    create_and_plot_sensitivity_histogram(sensitivity_map_2d, ppdf_files_base_dir, layouts_str_fname, layouts_str_title)