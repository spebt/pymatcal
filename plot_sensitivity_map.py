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
    ppdf_files_base_dir = "../data/mph_512pxx512px_128x128_19_rotations_1deg_col/outputs"
    # UPDATED: Specify the number of layouts to load and aggregate
    num_layouts_to_load = 1
    layout_indices = list(range(num_layouts_to_load)) # Creates a list like [0, 1]

    FOV_PIXELS_X, FOV_PIXELS_Y = 512, 512
    MM_PER_PIXEL_X, MM_PER_PIXEL_Y = 0.25, 0.25
    num_crystals_to_sum = -1
    # --- End Configuration ---

    if not os.path.isdir(ppdf_files_base_dir):
        print(f"Error: PPDF directory '{ppdf_files_base_dir}' not found.")
        exit()

    # --- NEW: Loop through specified layouts, load their data, and aggregate ---
    print(f"Loading and aggregating PPDFs for layouts: {layout_indices}...")
    aggregated_ppdfs = None

    for idx in layout_indices:
        print(f"  - Loading layout {idx:03d}...")
        ppdfs_for_layout = load_ppdfs_for_layout(ppdf_files_base_dir, idx, num_crystals_to_sum)
        
        if ppdfs_for_layout is None:
            print(f"Warning: Could not load data for layout {idx}. Skipping.")
            continue
            
        if aggregated_ppdfs is None:
            # Initialize the aggregated array with data from the first valid file
            aggregated_ppdfs = ppdfs_for_layout.astype(np.float32) # Use float for summation
        else:
            # Add data from subsequent files to the running total
            aggregated_ppdfs += ppdfs_for_layout

    if aggregated_ppdfs is None:
        print("Error: No PPDF data was successfully loaded. Exiting.")
        exit()
    
    print("\nAggregation complete.")

    # --- STEP 1: GENERATE THE SENSITIVITY MAP from the aggregated data ---
    sensitivity_map_1d = np.sum(aggregated_ppdfs, axis=0)
    sensitivity_map_2d = sensitivity_map_1d.reshape((FOV_PIXELS_Y, FOV_PIXELS_X))
            
    # Determine plot extent (no change)
    extent = [-(MM_PER_PIXEL_X * FOV_PIXELS_X / 2), (MM_PER_PIXEL_X * FOV_PIXELS_X / 2),
              -(MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2), (MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2)]

    # --- Create descriptive names for aggregated output files and titles ---
    # layouts_str_title = f"{num_layouts_to_load} Layouts ({'-'.join(map(str, layout_indices))})"
    layouts_str_fname = f"aggregated_{num_layouts_to_load}layouts"

    # Plot and Save the Aggregated Sensitivity Map Image
    plt.figure(figsize=(8, 7))
    plt.imshow(sensitivity_map_2d, cmap='viridis', origin='lower', extent=extent)
    plt.colorbar(label='Aggregated Sensitivity')
    # plt.title(f'Aggregated Sensitivity Map - {layouts_str_title}')
    plt.xlabel('X (mm)'); plt.ylabel('Y (mm)')
    plt.axhline(0, color='white', linestyle=':', lw=0.5); plt.axvline(0, color='white', linestyle=':', lw=0.5)
    plt.tight_layout()
    map_filename = os.path.join(ppdf_files_base_dir, f"sensitivity_map_{layouts_str_fname}.png")
    plt.savefig(map_filename)
    print(f"\nSaved aggregated sensitivity map to: {map_filename}")
    plt.close()

    # --- STEP 2: GENERATE THE HISTOGRAM FROM THE AGGREGATED MAP ---
    print("\nGenerating histogram from the aggregated sensitivity map...")
    create_and_plot_sensitivity_histogram(
        sensitivity_map_2d, 
        ppdf_files_base_dir, 
        file_suffix=layouts_str_fname, 
        title_suffix=layouts_str_title
    )
