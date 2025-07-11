import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from rich.progress import track # Use track for simple loops

INPUT_BEAM_VOLUMES_DIR = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/ordered/beam_volumes_new_formula"
BEAM_DIR = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/ordered/npzs"
OUTPUT_DIR = "results/ppds_maps_new_formula" # Save maps in a dedicated subdirectory
ORIGINAL_PPDF_DIR = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/ordered/sysmats"

N_ROTATIONS = 24
FOV_X_PIXELS = 512
FOV_Y_PIXELS = 512
EXPECTED_DETECTORS_PER_FILE = 864 # Should match previous steps
EPSILON = 1e-12
PIXEL_SIZE_MM = 0.25 # Ensure this is correct

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Load Consolidated Beam Volumes (V_i,b_new_approx) ---
print("Loading consolidated Beam Volumes (V_i,b_new_approx)...")
volume_file = os.path.join(INPUT_BEAM_VOLUMES_DIR, "ppds_volumes_new_formula_allrots.npz")
try:
    volume_data = np.load(volume_file)
    # This V_i,b_approx is FWHM_tan_old * 1.0 * L_mm_proxy (length of segmented beam)
    beam_V_ib_approx_all = volume_data['beam_volumes_new'] # Shape (N_ROTATIONS, n_detectors, 15 max beams)
    
    n_loaded_rotations = beam_V_ib_approx_all.shape[0]
    n_detectors_from_vol_file = beam_V_ib_approx_all.shape[1]
    max_beams_per_detector = beam_V_ib_approx_all.shape[2]

    if n_loaded_rotations != N_ROTATIONS:
         print(f"Warning: Loaded volumes data has {n_loaded_rotations} rotations, expected {N_ROTATIONS}.")
    if n_detectors_from_vol_file != EXPECTED_DETECTORS_PER_FILE:
         print(f"Warning: Loaded volumes data has {n_detectors_from_vol_file} detectors, expected {EXPECTED_DETECTORS_PER_FILE}.")

    print(f"Loaded V_i,b_approx shape: {beam_V_ib_approx_all.shape}")

except FileNotFoundError:
    print(f"Error: Could not find input file {volume_file}. Exiting.")
    exit()
except Exception as e:
    print(f"Error loading consolidated beam volumes: {e}. Exiting.")
    exit()


# --- Initialize PPDS Map ---
cumulative_ppds_map_new = np.zeros((FOV_Y_PIXELS, FOV_X_PIXELS), dtype=np.float64)
all_individual_ppds_maps_new = [] # For storing individual layout maps
first_layout_ppds_map_new = None

# --- Loop Through Rotations and Calculate Cumulative PPDS using New Formula ---
print(f"Calculating cumulative PPDS map (new formula) across {n_loaded_rotations} rotations...")
for aid in track(range(n_loaded_rotations), description="Processing Rotations (New Formula)"):
    current_rotation_ppds_map = np.zeros((FOV_Y_PIXELS, FOV_X_PIXELS), dtype=np.float64)
    try:
        # --- Load Original PPDFs for this rotation ---
        original_ppdf_fname = os.path.join(ORIGINAL_PPDF_DIR, f"scanner_ppdfs_{aid:02d}.hdf5")
        if not os.path.exists(original_ppdf_fname):
            print(f"Warning: Original PPDF file {original_ppdf_fname} not found for rotation {aid}. Skipping.")
            all_individual_ppds_maps_new.append(np.zeros_like(current_rotation_ppds_map))
            if aid == 0: first_layout_ppds_map_new = np.zeros_like(current_rotation_ppds_map)
            continue
            
        with h5py.File(original_ppdf_fname, "r") as f:
            # ppdfs_flat shape: (n_detectors_in_this_file, FOV_X_PIXELS * FOV_Y_PIXELS)
            ppdfs_flat_rot = np.copy(f["ppdfs"][:]) 
            n_detectors_in_ppdf, _ = ppdfs_flat_rot.shape
            if n_detectors_in_ppdf != n_detectors_from_vol_file:
                print(f"Warning: Detector count mismatch for rot {aid}. PPDFs: {n_detectors_in_ppdf}, Volumes: {n_detectors_from_vol_file}. Using min.")
            
        num_detectors_to_process_this_rot = min(n_detectors_in_ppdf, n_detectors_from_vol_file)

        # --- Get V_i,b_approx for this Rotation ---
        V_ib_approx_rot = beam_V_ib_approx_all[aid, :num_detectors_to_process_this_rot, :] # Shape (n_dets_proc, 15)

        # --- Calculate Sum_b(V_i,b_approx) per detector ---
        # Sum over the 15 max beams for each detector
        Sum_V_i_approx = np.sum(V_ib_approx_rot, axis=1) # Shape (n_dets_proc,)
        # Ensure denominator is not zero
        denominator_Sum_V_i = np.where(Sum_V_i_approx > EPSILON, Sum_V_i_approx, EPSILON)

        # --- Accumulate PPDS based on New Formula ---
        # PPDS_j_new = Σ_i ( PPDF_i,j / Σ_b V_i,b_new_approx )
        for det_idx in range(num_detectors_to_process_this_rot):
            if denominator_Sum_V_i[det_idx] <= EPSILON: # Skip if sum of volumes is effectively zero
                continue

            # PPDF_i for current detector, reshaped to 2D (H, W)
            ppdf_i_2d = ppdfs_flat_rot[det_idx].reshape(FOV_Y_PIXELS, FOV_X_PIXELS)
            
            # PPDF_i,j are the values in ppdf_i_2d
            # Add contributions where PPDF_i,j > 0
            # The condition PPDF_i,j > 0 is implicitly handled if ppdf_i_2d values are positive
            term_to_add_for_detector_i = ppdf_i_2d / denominator_Sum_V_i[det_idx]
            current_rotation_ppds_map += term_to_add_for_detector_i
            
        all_individual_ppds_maps_new.append(current_rotation_ppds_map)
        if aid == 0:
            first_layout_ppds_map_new = current_rotation_ppds_map.copy()
        cumulative_ppds_map_new += current_rotation_ppds_map

    except FileNotFoundError:
        print(f"Error: Data file not found for rotation {aid}. Skipping rotation.")
        all_individual_ppds_maps_new.append(np.zeros_like(current_rotation_ppds_map))
        if aid == 0: first_layout_ppds_map_new = np.zeros_like(current_rotation_ppds_map)
        continue
    except Exception as e:
        print(f"Error processing rotation {aid}: {e}. Skipping rotation.")
        all_individual_ppds_maps_new.append(np.zeros_like(current_rotation_ppds_map))
        if aid == 0: first_layout_ppds_map_new = np.zeros_like(current_rotation_ppds_map)
        continue

# --- Plotting Helper Function (modified from your ASCI plotting) ---
def plot_map_with_stats(map_data, title_str, filename_str, extent_mm=(-64, 64, -64, 64), v_is_percent=False):
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    
    # FOV extent needs to be determined from PIXEL_SIZE_MM and FOV_X/Y_PIXELS
    # Assuming (0,0) is center of FOV for plotting extent
    half_width_mm = FOV_X_PIXELS * PIXEL_SIZE_MM / 2.0
    half_height_mm = FOV_Y_PIXELS * PIXEL_SIZE_MM / 2.0
    actual_extent = [-half_width_mm, half_width_mm, -half_height_mm, half_height_mm]

    im = ax.imshow(map_data.T, extent=actual_extent, origin="lower", cmap='viridis', interpolation='nearest') # Transpose for X,Y
    
    cbar_label = "PPDS Value (New Formula)"
    if v_is_percent: cbar_label = "Index Value"

    cbar = fig.colorbar(im, ax=ax, label=cbar_label)

    min_val = np.min(map_data)
    max_val = np.max(map_data)

    if v_is_percent:
        from matplotlib.ticker import PercentFormatter
        cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
        cbar.update_ticks()
        title = f"{title_str}, max: {max_val:.2%}, min: {min_val:.2%}"
    else:
        # For PPDS, scientific notation might be better if values are very small/large
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4)) # Adjust as needed
        cbar.formatter = formatter
        cbar.update_ticks()
        title = f"{title_str}, max: {max_val:.3e}, min: {min_val:.3e}"
        
    ax.set_xlabel(f"X (mm)")
    ax.set_ylabel(f"Y (mm)")
    ax.set_title(title)
    
    plt.savefig(filename_str, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {filename_str}")


# --- Save and Plot Final Cumulative PPDS Map (New Formula) ---
print("\nSaving final cumulative PPDS map (new formula)...")
final_map_npy_file_new = os.path.join(OUTPUT_DIR, 'ppds_map_new_formula_cumulative.npy')
np.save(final_map_npy_file_new, cumulative_ppds_map_new)
print(f"Final new PPDS map saved to: {final_map_npy_file_new}")

plot_map_with_stats(cumulative_ppds_map_new,
                    f"PPDS Map (New Formula), {n_loaded_rotations} Rotations Cumulative",
                    os.path.join(OUTPUT_DIR, 'ppds_map_new_formula_cumulative.png'))

# --- Save and Plot "No Rotation" PPDS Map (New Formula) ---
if first_layout_ppds_map_new is not None:
    no_rot_map_npy_file_new = os.path.join(OUTPUT_DIR, 'ppds_map_new_formula_no_rotation.npy')
    np.save(no_rot_map_npy_file_new, first_layout_ppds_map_new)
    print(f"'No Rotation' new PPDS map saved to: {no_rot_map_npy_file_new}")

    plot_map_with_stats(first_layout_ppds_map_new,
                        "PPDS Map (New Formula), No Rotation (Layout 0)",
                        os.path.join(OUTPUT_DIR, 'ppds_map_new_formula_no_rotation.png'))
else:
    print("Could not generate 'No Rotation' plot as data for layout 0 was not processed.")

print("\nPPDS map calculation (new formula) complete.")