import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

def load_ppdfs_for_layout(ppdf_dir: str, layout_idx: int, n_xtals_to_load: int = -1):
    """Loads PPDFs from an HDF5 file for a specific layout."""
    ppdf_filename = os.path.join(ppdf_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
    if not os.path.exists(ppdf_filename):
        print(f"PPDF file {ppdf_filename} does not exist.")
        return None
    
    with h5py.File(ppdf_filename, "r") as f:
        # Selects all crystals if n_xtals_to_load is -1, otherwise slices the array
        ppdfs_data = f["ppdfs"][:n_xtals_to_load] if n_xtals_to_load != -1 else f["ppdfs"][:]
    return ppdfs_data

def create_and_plot_sensitivity_histogram(sensitivity_map: np.ndarray, output_dir: str, file_suffix: str, title_suffix: str):
    """Generates and displays a histogram for a given sensitivity map."""
    # --- 1. Prepare Data: Filter for values within the FOV ---
    fov_values = sensitivity_map[sensitivity_map > 0].flatten()
    if fov_values.size == 0:
        print("Warning: The sensitivity map contains no non-zero values.")
        return

    # --- 2. Create and Plot the Histogram ---
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

    # --- 3. Save the Histogram Plot ---
    hist_filename = os.path.join(output_dir, f"sensitivity_histogram_{file_suffix}.png")
    plt.savefig(hist_filename)
    print(f"Saved sensitivity histogram to: {hist_filename}")
    plt.close()


if __name__ == "__main__":
    # --- Configuration ---
    ppdf_files_base_dir = "/vscratch/grp-rutaoyao/sid/data-test/mph_hourglass_single_position_base_3mm_18pinholes_rotated_elliptical/outputs/"
    num_layouts_to_load = 40
    layout_indices = list(range(num_layouts_to_load))

    FOV_PIXELS_X, FOV_PIXELS_Y = 512, 512
    MM_PER_PIXEL_X, MM_PER_PIXEL_Y = 0.25, 0.25
    num_crystals_to_sum = -1
    # --- End Configuration ---

    if not os.path.isdir(ppdf_files_base_dir):
        print(f"Error: PPDF directory '{ppdf_files_base_dir}' not found.")
        exit()

    # --- Loop through layouts, load, and aggregate ---
    print(f"Loading and aggregating PPDFs for layouts: {layout_indices}...")
    aggregated_ppdfs = None
    successful_loads = 0 # NEW: Counter for successfully loaded layouts

    for idx in layout_indices:
        print(f"  - Loading layout {idx:03d}...")
        ppdfs_for_layout = load_ppdfs_for_layout(ppdf_files_base_dir, idx, num_crystals_to_sum)
        
        if ppdfs_for_layout is None:
            print(f"Warning: Could not load data for layout {idx}. Skipping.")
            continue
            
        if aggregated_ppdfs is None:
            aggregated_ppdfs = ppdfs_for_layout.astype(np.float32)
        else:
            aggregated_ppdfs += ppdfs_for_layout
        
        successful_loads += 1 # NEW: Increment the counter on a successful load

    if aggregated_ppdfs is None:
        print("Error: No PPDF data was successfully loaded. Exiting.")
        exit()
    
    print(f"\nAggregation complete. Successfully loaded {successful_loads} layouts.")

    # --- Generate and NORMALIZE the sensitivity map ---
    if successful_loads > 0:
        # UPDATED: Divide by the number of successful loads to get the average
        sensitivity_map_1d = np.sum(aggregated_ppdfs, axis=0) / successful_loads
    else:
        # This case is already handled by the exit above, but it's good practice
        sensitivity_map_1d = np.zeros(FOV_PIXELS_X * FOV_PIXELS_Y)

    sensitivity_map_2d = sensitivity_map_1d.reshape((FOV_PIXELS_Y, FOV_PIXELS_X))
            
    # Determine plot extent
    extent = [-(MM_PER_PIXEL_X * FOV_PIXELS_X / 2), (MM_PER_PIXEL_X * FOV_PIXELS_X / 2),
              -(MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2), (MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2)]

    # --- Create descriptive names for aggregated output files and titles ---
    layouts_str_fname = f"aggregated_{successful_loads}layouts_normalized"
    layouts_str_title = f"Normalized Sensitivity Map ({successful_loads} Layouts)"

    # Plot and Save the Aggregated and Normalized Sensitivity Map Image
    plt.figure(figsize=(8, 7))
    plt.imshow(sensitivity_map_2d, cmap='viridis', origin='lower', extent=extent)
    # UPDATED: Changed colorbar label to reflect normalization
    plt.colorbar(label='Average Sensitivity (Normalized)')
    plt.title(layouts_str_title)
    plt.xlabel('X (mm)'); plt.ylabel('Y (mm)')
    plt.axhline(0, color='white', linestyle=':', lw=0.5); plt.axvline(0, color='white', linestyle=':', lw=0.5)
    plt.tight_layout()
    map_filename = os.path.join(ppdf_files_base_dir, f"sensitivity_map_{layouts_str_fname}.png")
    plt.savefig(map_filename)
    print(f"\nSaved normalized sensitivity map to: {map_filename}")
    plt.close()
    
    # NEW: Also create a histogram for the normalized data
    create_and_plot_sensitivity_histogram(sensitivity_map_2d, ppdf_files_base_dir, layouts_str_fname, layouts_str_title)
