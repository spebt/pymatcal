# NEW: Import sys for args and glob for file searching
import sys
import glob
import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# Assuming these utility functions are available and correctly pathed
from geometry_2d_utils import fov_tensor_dict
from ppdf_io import load_ppdfs_data_from_hdf5

# --- Configuration & Setup ---
# FOV setup (must match PPDF generation)
fov_config = fov_tensor_dict(
    n_pixels=(512, 512),
    mm_per_pixel=(0.25, 0.25),
    center_coordinates=(0.0, 0.0),
)
fov_y_dim = int(fov_config["n pixels"][0].item())
fov_x_dim = int(fov_config["n pixels"][1].item())
n_voxels_flat = fov_y_dim * fov_x_dim

# Column indices from your beam_properties header (0-indexed)
# As defined in beam_property_io.py
DETECTOR_IDX_COL = 1
FWHM_TANGENTIAL_COL = 4
FWHM_RADIAL_COL = 5
FWHM_VERTICAL = 1.0 # For 2D study

# --- Plotting Helper Function ---
def plot_ppds_map(ppds_data_2d, title_prefix, output_filename, fov_conf):
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")

    total_width_mm = fov_conf["size in mm"][1].item()
    total_height_mm = fov_conf["size in mm"][0].item()
    center_x_mm = fov_conf["center coordinates in mm"][1].item()
    center_y_mm = fov_conf["center coordinates in mm"][0].item()
    plot_extent = [
        center_x_mm - total_width_mm / 2.0, center_x_mm + total_width_mm / 2.0,
        center_y_mm - total_height_mm / 2.0, center_y_mm + total_height_mm / 2.0,
    ]

    im = ax.imshow(ppds_data_2d, cmap='viridis', origin='lower', extent=plot_extent)
    
    cbar = fig.colorbar(im, ax=ax, label="PPDS Value")
    cbar_formatter = ScalarFormatter(useMathText=True)
    cbar_formatter.set_powerlimits((0, 0))
    cbar.formatter = cbar_formatter
    cbar.update_ticks()

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    min_val = np.min(ppds_data_2d)
    max_val = np.max(ppds_data_2d)
    ax.set_title(f"{title_prefix}\nmax: {max_val:.3e}, min: {min_val:.3e}")

    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

# --- Main Processing ---
if __name__ == "__main__":
    # --- MODIFIED: Use command-line arguments and glob ---
    if len(sys.argv) != 3:
        print("\nUsage: python calculate_and_plot_ppds_multiplexing.py <properties_dir> <ppdfs_dir>")
        print("  - <properties_dir>: Directory containing 'beams_properties_*.hdf5' files.")
        print("  - <ppdfs_dir>: Directory containing original 'position_*_ppdfs.hdf5' files.\n")
        sys.exit(1)

    properties_input_dir = sys.argv[1]
    ppdfs_dataset_dir = sys.argv[2]

    if not os.path.isdir(properties_input_dir):
        raise NotADirectoryError(f"Beam properties input directory not found: {properties_input_dir}")
    if not os.path.isdir(ppdfs_dataset_dir):
        raise NotADirectoryError(f"PPDFs dataset directory not found: {ppdfs_dataset_dir}")

    # Discover beam properties files
    property_files = sorted(glob.glob(os.path.join(properties_input_dir, "beams_properties_*.hdf5")))
    if not property_files:
        print(f"Error: No 'beams_properties_*.hdf5' files found in {properties_input_dir}")
        sys.exit(1)

    # Dynamically extract unique ID
    try:
        base_filename = os.path.basename(property_files[0])
        layouts_unique_id = base_filename.split('_')[2]
        print(f"Detected unique ID: {layouts_unique_id}")
    except IndexError:
        print(f"Error: Could not parse unique ID from filename: {property_files[0]}")
        sys.exit(1)
        
    n_layouts = len(property_files)
    layout_sequence = range(n_layouts)
    
    # --- END MODIFIED SECTION ---

    # Output directories
    ppds_output_dir = "output/ppds_maps_multiplexing"
    os.makedirs(ppds_output_dir, exist_ok=True)
    cumulative_plots_dir = "output/ppds_cumulative_plots_multiplexing"
    os.makedirs(cumulative_plots_dir, exist_ok=True)

    all_layout_ppds_maps = [] # To store individual maps for cumulative calculation
    first_layout_ppds_map = None # To store layout 0 map

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(), BarColumn(), TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        refresh_per_second=2,
    ) as progress:

        task_ppds_layouts = progress.add_task("Processing PPDS for layouts...", total=n_layouts)

        for layout_idx in layout_sequence:
            progress.console.print(f"--- Processing Layout {layout_idx:04d} for PPDS ---")
            
            # --- MODIFIED: Use dynamic ID and correct padding ---
            beams_properties_hdf5_filename = f"beams_properties_{layouts_unique_id}_{layout_idx:04d}.hdf5"
            path_to_props = os.path.join(properties_input_dir, beams_properties_hdf5_filename)
            
            # 1. Load beam properties for the current layout
            if not os.path.exists(path_to_props):
                progress.console.print(f"Warning: Beam properties file not found, skipping layout {layout_idx}.")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                progress.update(task_ppds_layouts, advance=1)
                continue
            
            try:
                with h5py.File(path_to_props, "r") as f:
                    layout_beams_properties = torch.from_numpy(f["beam_properties"][:])
            except Exception as e:
                progress.console.print(f"Error loading beam properties for layout {layout_idx}: {e}")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                progress.update(task_ppds_layouts, advance=1)
                continue

            if layout_beams_properties.shape[0] == 0:
                progress.console.print(f"No beams found in properties file for layout {layout_idx}, skipping.")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                progress.update(task_ppds_layouts, advance=1)
                continue

            # 2. Pre-calculate Sum_b(V_i,b) for each detector i
            max_detector_idx_in_props = int(layout_beams_properties[:, DETECTOR_IDX_COL].max().item())
            num_detectors_for_sum_v = max_detector_idx_in_props + 1
            sum_V_i_b_for_detectors = torch.zeros(num_detectors_for_sum_v, dtype=torch.float32)

            for beam_prop_row in layout_beams_properties:
                detector_i = int(beam_prop_row[DETECTOR_IDX_COL].item())
                fwhm_tangential = beam_prop_row[FWHM_TANGENTIAL_COL].item()
                fwhm_radial = beam_prop_row[FWHM_RADIAL_COL].item()

                if np.isnan(fwhm_tangential) or np.isnan(fwhm_radial) or fwhm_tangential <= 0 or fwhm_radial <= 0:
                    continue
                V_i_b = fwhm_tangential * FWHM_VERTICAL * fwhm_radial
                if detector_i < num_detectors_for_sum_v: sum_V_i_b_for_detectors[detector_i] += V_i_b

            sum_V_i_b_for_detectors[sum_V_i_b_for_detectors <= 1e-9] = 1e-9

            # 3. Load original PPDFs for this layout
            # The original PPDFs are indexed with 3-digit padding
            ppdfs_hdf5_filename_original = f"position_{layout_idx:03d}_ppdfs.hdf5"
            try:
                original_ppdfs_for_layout = load_ppdfs_data_from_hdf5(
                    ppdfs_dataset_dir, ppdfs_hdf5_filename_original, fov_config
                )
            except Exception as e:
                progress.console.print(f"Error loading original PPDFs for layout {layout_idx}: {e}")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                progress.update(task_ppds_layouts, advance=1)
                continue
            
            if original_ppdfs_for_layout.ndim == 3:
                original_ppdfs_for_layout = original_ppdfs_for_layout.view(original_ppdfs_for_layout.shape[0], -1)
            num_detectors_in_ppdf_file = original_ppdfs_for_layout.shape[0]

            # 4. Calculate PPDS_j for all voxels j
            ppds_map_tensor = torch.zeros(n_voxels_flat, dtype=torch.float32)
            num_detectors_to_process = min(num_detectors_for_sum_v, num_detectors_in_ppdf_file)

            for detector_i in range(num_detectors_to_process):
                if sum_V_i_b_for_detectors[detector_i] <= 1e-9:
                    continue
                ppdf_i_j_values = original_ppdfs_for_layout[detector_i, :]
                relevant_voxels_mask = ppdf_i_j_values > 0
                term_to_add = ppdf_i_j_values[relevant_voxels_mask] / sum_V_i_b_for_detectors[detector_i]
                ppds_map_tensor[relevant_voxels_mask] += term_to_add
            
            # 5. Save and Plot individual map, and store for aggregation
            current_ppds_map_reshaped = ppds_map_tensor.view(fov_y_dim, fov_x_dim).cpu().numpy()
            
            # --- MODIFIED: Use correct padding for output filenames ---
            ppds_hdf5_output_filename = os.path.join(ppds_output_dir, f"ppds_map_layout_{layout_idx:04d}.hdf5")
            with h5py.File(ppds_hdf5_output_filename, "w") as f_out:
                f_out.create_dataset("ppds_map", data=current_ppds_map_reshaped)
            
            plot_title_prefix = f"PPDS Map, Layout {layout_idx:04d}"
            plot_filename = os.path.join(ppds_output_dir, f"ppds_plot_layout_{layout_idx:04d}.png")
            plot_ppds_map(current_ppds_map_reshaped, plot_title_prefix, plot_filename, fov_config)
            
            all_layout_ppds_maps.append(current_ppds_map_reshaped)
            if layout_idx == 0:
                first_layout_ppds_map = current_ppds_map_reshaped
            
            progress.update(task_ppds_layouts, advance=1)

    # --- After the loop: Cumulative and No-Rotation Plots ---
    if not all_layout_ppds_maps:
        progress.console.print("No PPDS maps were generated. Skipping cumulative plots.")
    else:
        # Cumulative PPDS map (sum of all layouts)
        if len(all_layout_ppds_maps) == len(layout_sequence):
            progress.console.print("Summing all layout PPDS maps for cumulative plot...")
            cumulative_ppds_map = np.sum(np.stack(all_layout_ppds_maps, axis=0), axis=0)
            
            # --- MODIFIED: Dynamic title and filename for cumulative plot ---
            plot_title_prefix_cum = f"PPDS Map, {len(layout_sequence)} Layouts (Cumulative)"
            plot_filename_cum = os.path.join(cumulative_plots_dir, f"ppds_map_{len(layout_sequence)}_layouts_cumulative.png")
            plot_ppds_map(cumulative_ppds_map, plot_title_prefix_cum, plot_filename_cum, fov_config)
            progress.console.print(f"Cumulative PPDS map (sum) plotted: {plot_filename_cum}")

            cumulative_hdf5_filename = os.path.join(cumulative_plots_dir, f"ppds_map_{len(layout_sequence)}_layouts_cumulative.hdf5")
            with h5py.File(cumulative_hdf5_filename, "w") as f_out:
                f_out.create_dataset("ppds_map_cumulative", data=cumulative_ppds_map)
            progress.console.print(f"Cumulative PPDS map (sum) saved: {cumulative_hdf5_filename}")
        else:
            progress.console.print(f"Warning: Expected {len(layout_sequence)} maps, got {len(all_layout_ppds_maps)}. Cumulative sum might be affected.")

        # "No Rotation" PPDS map (using the first layout's map)
        if first_layout_ppds_map is not None:
            plot_title_prefix_no_rot = "PPDS Map, Single Layout (Layout 0)"
            plot_filename_no_rot = os.path.join(cumulative_plots_dir, "ppds_map_layout_0.png")
            plot_ppds_map(first_layout_ppds_map, plot_title_prefix_no_rot, plot_filename_no_rot, fov_config)
            progress.console.print(f"Single layout PPDS map (Layout 0) plotted: {plot_filename_no_rot}")
        else:
            progress.console.print("Could not plot single layout map as Layout 0 data was not processed.")

    print("\nPPDS processing completed.")