import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
import time
import h5py
import torch
torch.set_num_threads(8)
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# Assuming these utility functions are available and correctly pathed
from geometry_2d_utils import fov_tensor_dict
from ppdf_io import load_ppdfs_data_from_hdf5

# --- Configuration ---
properties_input_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/ppdf-analysis/beam-analysis/output"  # Directory where beams_properties_...hdf5 files are

#FOR MULTIPLEXING, UNCOMMENT THIS
#ppds_output_dir = "output/ppds_maps" # Directory to save PPDS maps and plots
#os.makedirs(ppds_output_dir, exist_ok=True)
#cumulative_plots_dir = "output/ppds_cumulative_plots"
#os.makedirs(cumulative_plots_dir, exist_ok=True)

#FOR NON MULTIPLEXING, USE THIS
ppds_output_dir = "output/ppds_maps_non_multiplexing"
os.makedirs(ppds_output_dir, exist_ok=True)
cumulative_plots_dir = "output/ppds_cumulative_plots_non_multiplexing"
os.makedirs(cumulative_plots_dir, exist_ok=True)

# PPDF dataset paths (these need to be correct, as in your extraction scripts)
scanner_layouts_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/scanner_layouts" # ADJUST AS NEEDED
scanner_layouts_filename = "scanner_layouts_77faff53af5863ca146878c7c496c75e.tensor" # ADJUST
ppdfs_dataset_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/scanner_layouts_77faff53af5863ca146878c7c496c75e" # ADJUST

# Unique ID for file naming (should match your properties files)
layouts_unique_id = "77faff53af5863ca146878c7c496c75e" 
fov_config = fov_tensor_dict(
    n_pixels=(512, 512),
    mm_per_pixel=(0.25, 0.25),
    center_coordinates=(0.0, 0.0),
)
fov_y_dim = int(fov_config["n pixels"][0].item())
fov_x_dim = int(fov_config["n pixels"][1].item())
n_voxels_flat = fov_y_dim * fov_x_dim

DETECTOR_IDX_COL = 1
FWHM_TANGENTIAL_COL = 4
FWHM_RADIAL_COL = 5
FWHM_VERTICAL = 1.0
BEAM_IDX_COL = 2
EPSILON = 1e-12

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

    # Determine vmin and vmax for consistent color scaling if desired, or let imshow auto-scale
    # For individual plots, auto-scaling is fine. For comparing, you might want fixed vmin/vmax.
    im = ax.imshow(ppds_data_2d, cmap='viridis', origin='lower', extent=plot_extent)
    
    cbar = fig.colorbar(im, ax=ax, label="PPDS Value")
    # Optional: Format colorbar ticks if numbers get too large/small
    # cbar_formatter = ScalarFormatter(useMathText=True)
    # cbar_formatter.set_powerlimits((0, 0))
    # cbar.formatter = cbar_formatter
    # cbar.update_ticks()

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    min_val = np.min(ppds_data_2d)
    max_val = np.max(ppds_data_2d)
    ax.set_title(f"{title_prefix}, max: {max_val:.3e}, min: {min_val:.3e}") # Using scientific notation

    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

# --- Main Processing ---
if __name__ == "__main__":
    if not os.path.isdir(ppdfs_dataset_dir):
        print(f"ERROR: PPDFs dataset directory not found: {ppdfs_dataset_dir}")
        sys.exit(1)
    if not os.path.isdir(properties_input_dir):
        print(f"ERROR: Beam properties input directory not found: {properties_input_dir}")
        sys.exit(1)

    all_layout_ppds_maps = [] # To store individual maps for cumulative calculation
    first_layout_ppds_map = None # To store layout 0 map

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(), BarColumn(), TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        refresh_per_second=2,
    ) as progress:

        layout_sequence = range(24)
        task_ppds_layouts = progress.add_task("Processing PPDS for layouts...", total=len(list(layout_sequence)))

        for layout_idx in layout_sequence:
            # ... (Most of the existing loop for loading properties, Sum(V_i,b), loading PPDFs, and calculating ppds_map) ...
            # --- Start of existing loop content (abbreviated) ---
            progress.console.print(f"--- Processing Layout {layout_idx:02d} for PPDS ---")
            beams_properties_hdf5_filename = (
                f"beams_properties_{layouts_unique_id}_{layout_idx:02d}.hdf5"
            )
            path_to_props = os.path.join(properties_input_dir, beams_properties_hdf5_filename)
            if not os.path.exists(path_to_props): # ... (skip if not exists)
                progress.console.print(f"Warning: Beam properties file not found: {path_to_props}, skipping layout {layout_idx}.")
                progress.update(task_ppds_layouts, advance=1)
                continue
            try: # ... (load layout_beams_properties)
                with h5py.File(path_to_props, "r") as f:
                    layout_beams_properties = torch.from_numpy(f["beam_properties"][:])
            except Exception as e:
                progress.console.print(f"Error loading beam properties for layout {layout_idx} from '{path_to_props}': {e}")
                progress.update(task_ppds_layouts, advance=1)
                continue
            if layout_beams_properties.shape[0] == 0: # ... (skip if no beams)
                progress.console.print(f"No beams found in properties file for layout {layout_idx}, skipping PPDS.")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)) # Append zeros if skipping
                if layout_idx == 0:
                    first_layout_ppds_map = np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)
                progress.update(task_ppds_layouts, advance=1)
                continue
            if layout_beams_properties.shape[1] <= max(DETECTOR_IDX_COL, FWHM_TANGENTIAL_COL, FWHM_RADIAL_COL): # ... (column check)
                progress.console.print(f"Error: Beam properties file for layout {layout_idx} has too few columns. Skipping.")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)) # Append zeros
                if layout_idx == 0:
                    first_layout_ppds_map = np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)
                progress.update(task_ppds_layouts, advance=1)
                continue

            #this modification is for non mux ppdfs
            # --- MODIFICATION: Identify Non-Multiplexing Detectors ---
            detector_indices_in_layout = layout_beams_properties[:, DETECTOR_IDX_COL].int()
            unique_detector_ids, counts = torch.unique(detector_indices_in_layout, return_counts=True)
            
            non_multiplexing_detector_ids = unique_detector_ids[counts == 1]
            if non_multiplexing_detector_ids.numel() == 0:
                progress.console.print(f"No non-multiplexing detectors found in layout {layout_idx}. Skipping PPDS calculation for this layout.")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                if layout_idx == 0: first_layout_ppds_map = np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)
                progress.update(task_ppds_layouts, advance=1)
                continue
            
            # Create a boolean mask for beams belonging to non-multiplexing detectors
            non_multiplexing_beam_mask = torch.isin(detector_indices_in_layout, non_multiplexing_detector_ids)
            beams_from_non_multiplexing_detectors = layout_beams_properties[non_multiplexing_beam_mask]
            
            if beams_from_non_multiplexing_detectors.shape[0] == 0: # Should be caught by above, but good check
                progress.console.print(f"Logical error: No beams selected after non-multiplexing filter for layout {layout_idx}. Skipping.")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                if layout_idx == 0: first_layout_ppds_map = np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)
                progress.update(task_ppds_layouts, advance=1)
                continue
            # --- END OF MODIFICATION ---





            # Calculate Sum_b(V_i,b)
            max_detector_idx_overall = -1
            if layout_beams_properties[:, DETECTOR_IDX_COL].numel() > 0:
                 max_detector_idx_in_props = int(layout_beams_properties[:, DETECTOR_IDX_COL].max().item())
            if max_detector_idx_overall == -1 : # Should be caught earlier
                progress.console.print(f"No detector indices found in layout {layout_idx}. Skipping.")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                if layout_idx == 0: first_layout_ppds_map = np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)
                progress.update(task_ppds_layouts, advance=1)
                continue

            num_detectors_total_possible = max_detector_idx_overall + 1
            sum_V_i_b_for_detectors = torch.zeros(num_detectors_total_possible, dtype=torch.float32)
            sub_task_sum_v = progress.add_task(f"[L{layout_idx:02d}] Calc Sum(V_i,b) (NM)...", total=beams_from_non_multiplexing_detectors.shape[0], visible=False)
            # --- MODIFICATION: Iterate only over beams from non-multiplexing detectors ---
            for beam_prop_row in beams_from_non_multiplexing_detectors:
                detector_i = int(beam_prop_row[DETECTOR_IDX_COL].item())
                # Since it's non-multiplexing, there's only one beam 'b' for this detector 'i'.
                # The Sum_b V_i,b becomes just V_i,single_beam.
                fwhm_tangential = beam_prop_row[FWHM_TANGENTIAL_COL].item()
                fwhm_radial = beam_prop_row[FWHM_RADIAL_COL].item()

                if np.isnan(fwhm_tangential) or np.isnan(fwhm_radial) or fwhm_tangential <= EPSILON or fwhm_radial <= EPSILON:
                    progress.update(sub_task_sum_v, advance=1)
                    continue 

                V_i_single_beam = fwhm_tangential * FWHM_VERTICAL * fwhm_radial
                if detector_i < num_detectors_total_possible:
                    sum_V_i_b_for_detectors[detector_i] = V_i_single_beam # Assign, not accumulate
                progress.update(sub_task_sum_v, advance=1)
            progress.remove_task(sub_task_sum_v)
            
            sum_V_i_b_for_detectors[sum_V_i_b_for_detectors <= EPSILON] = EPSILON
            
            # Load original PPDFs
            ppdfs_hdf5_filename_original = f"position_{layout_idx:03d}_ppdfs.hdf5"
            path_to_original_ppdfs = os.path.join(ppdfs_dataset_dir, ppdfs_hdf5_filename_original)
            if not os.path.exists(path_to_original_ppdfs): # ... (skip if not exists)
                progress.console.print(f"Warning: Original PPDF file not found: {path_to_original_ppdfs}, skipping layout {layout_idx}.")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                if layout_idx == 0:
                    first_layout_ppds_map = np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)
                progress.update(task_ppds_layouts, advance=1)
                continue
            try: # ... (load original_ppdfs_for_layout)
                original_ppdfs_for_layout = load_ppdfs_data_from_hdf5(
                    ppdfs_dataset_dir, ppdfs_hdf5_filename_original, fov_config
                )
            except Exception as e:
                progress.console.print(f"Error loading original PPDFs for layout {layout_idx} from '{path_to_original_ppdfs}': {e}")
                all_layout_ppds_maps.append(np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32))
                if layout_idx == 0:
                    first_layout_ppds_map = np.zeros((fov_y_dim, fov_x_dim), dtype=np.float32)
                progress.update(task_ppds_layouts, advance=1)
                continue

            if original_ppdfs_for_layout.ndim == 3: original_ppdfs_for_layout = original_ppdfs_for_layout.view(original_ppdfs_for_layout.shape[0], -1)
            num_detectors_in_ppdf_file = original_ppdfs_for_layout.shape[0]

            # Calculate PPDS map for the current layout
            ppds_map_tensor = torch.zeros(n_voxels_flat, dtype=torch.float32) # Renamed from ppds_map
            valid_nm_detector_ids_for_ppds = [
                did.item() for did in non_multiplexing_detector_ids if did.item() < num_detectors_in_ppdf_file
            ]

            if not valid_nm_detector_ids_for_ppds:
                progress.console.print(f"No valid non-multiplexing detectors to process for PPDS in layout {layout_idx}.")
            else:
                sub_task_voxels = progress.add_task(f"[L{layout_idx:02d}] Calc PPDS/voxel (NM)...", total=len(valid_nm_detector_ids_for_ppds), visible=False)
                for detector_i in valid_nm_detector_ids_for_ppds:
                    if detector_i >= num_detectors_total_possible or sum_V_i_b_for_detectors[detector_i] <= EPSILON : # Check if Sum_V was calculated and is valid
                        progress.update(sub_task_voxels, advance=1)
                        continue

                    ppdf_i_j_values = original_ppdfs_for_layout[detector_i, :]
                    relevant_voxels_mask = ppdf_i_j_values > 0 # Or some other threshold if needed
                    
                    term_to_add = ppdf_i_j_values[relevant_voxels_mask] / sum_V_i_b_for_detectors[detector_i]
                    ppds_map_tensor[relevant_voxels_mask] += term_to_add
                    progress.update(sub_task_voxels, advance=1)
                progress.remove_task(sub_task_voxels)
            # --- END OF MODIFICATION ---
            
            current_ppds_map_reshaped = ppds_map_tensor.view(fov_y_dim, fov_x_dim).cpu().numpy()
            all_layout_ppds_maps.append(current_ppds_map_reshaped)
            if layout_idx == 0:
                first_layout_ppds_map = current_ppds_map_reshaped.copy()

            # Save individual HDF5
            ppds_hdf5_output_filename = os.path.join(ppds_output_dir, f"ppds_map_layout_{layout_idx:02d}.hdf5")
            with h5py.File(ppds_hdf5_output_filename, "w") as f_out:
                f_out.create_dataset("ppds_map", data=current_ppds_map_reshaped)

            # Plot individual map (using helper function)
            plot_title_prefix = f"PPDS Map, Layout {layout_idx:02d}"
            plot_filename = os.path.join(ppds_output_dir, f"ppds_plot_layout_{layout_idx:02d}.png")
            plot_ppds_map(current_ppds_map_reshaped, plot_title_prefix, plot_filename, fov_config)
            
            # Store for cumulative and first layout plots
            all_layout_ppds_maps.append(current_ppds_map_reshaped)
            if layout_idx == 0:
                first_layout_ppds_map = current_ppds_map_reshaped
            
            progress.console.print(f"Layout {layout_idx:02d}: PPDS map processed and saved.")
            progress.update(task_ppds_layouts, advance=1)

        # --- After the loop: Cumulative and No-Rotation Plots ---
        if not all_layout_ppds_maps or len(all_layout_ppds_maps) != len(layout_sequence):
             progress.console.print("Warning: Not all layouts processed or maps generated. Cumulative results might be incomplete.")
        if any(m.any() for m in all_layout_ppds_maps):
            cumulative_ppds_map = np.sum(np.stack(all_layout_ppds_maps, axis=0), axis=0)
            plot_title_prefix_cum = f"PPDS Map (Non-Multiplex), {len(layout_sequence)} Rotations Cumulative"
            plot_filename_cum = os.path.join(cumulative_plots_dir, "ppds_map_all_rotations_cumulative_nm.png")
            plot_ppds_map(cumulative_ppds_map, plot_title_prefix_cum, plot_filename_cum, fov_config)
            progress.console.print(f"Cumulative PPDS map (NM, sum) plotted: {plot_filename_cum}")
            cumulative_hdf5_filename = os.path.join(cumulative_plots_dir, "ppds_map_all_rotations_cumulative_nm.hdf5")
            with h5py.File(cumulative_hdf5_filename, "w") as f_out:
                f_out.create_dataset("ppds_map_cumulative_nm", data=cumulative_ppds_map)

            if first_layout_ppds_map is not None and first_layout_ppds_map.any():
                plot_title_prefix_no_rot = "PPDS Map (Non-Multiplex), No Rotation (Layout 0)"
                plot_filename_no_rot = os.path.join(cumulative_plots_dir, "ppds_map_no_rotation_nm.png")
                plot_ppds_map(first_layout_ppds_map, plot_title_prefix_no_rot, plot_filename_no_rot, fov_config)
            else:
                progress.console.print("No data for Layout 0 (NM), skipping 'No Rotation' plot.")
        else:
            progress.console.print("No valid PPDS data generated across all layouts. Skipping final cumulative plots.")

        progress.console.print(f"PPDS (Non-Multiplexing) processing completed.")
        progress.console.print(f"Cumulative/No-Rotation plots in: {cumulative_plots_dir}")