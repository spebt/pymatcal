import numpy as np
import h5py
import os
import scipy as sp
import skimage as ski
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
)

INPUT_DIR = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/ordered/sysmats"
OUTPUT_DIR = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/ordered/beam_volumes_new_formula" # Saving final results here
BEAM_DIR = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/ordered/npzs"
N_ROTATIONS = 24
FOV_X_PIXELS = 512
FOV_Y_PIXELS = 512
PIXEL_SIZE_MM = 0.25
EXPECTED_DETECTORS_PER_FILE = 864 
PROFILE_HALF_LENGTH_TANGENTIAL = 15 # Half-length of line profile for FWHM in pixels (we can change this if we want)
PROFILE_INTERPOLATION_FACTOR = 4 # Interpolation factor for FWHM calculation
RADIAL_PROFILE_N_SAMPLES = 200

# --- Helper Functions ---

def get_intersections(slope, intercept, fov_x_max_idx, fov_y_max_idx):
    """
    Calculates intersection points of line x = slope*y + intercept
    with the FOV boundaries (0 to fov_max_idx for both x and y).
    fov_x_max_idx = FOV_X_PIXELS - 1
    fov_y_max_idx = FOV_Y_PIXELS - 1
    Returns a list of valid [x, y] intersection points.
    """
    points = []
    if np.isinf(slope): # Vertical line x = intercept
        if 0 <= intercept <= fov_x_max_idx:
            points.append([intercept, 0])
            points.append([intercept, fov_y_max_idx])
    else: # Non-vertical lines
        x_at_y0 = intercept
        x_at_y_max = slope * fov_y_max_idx + intercept
        if 0 <= x_at_y0 <= fov_x_max_idx:
            points.append([x_at_y0, 0])
        if 0 <= x_at_y_max <= fov_x_max_idx:
            points.append([x_at_y_max, fov_y_max_idx])

        if not np.isclose(slope, 0):
            y_at_x0 = -intercept / slope
            y_at_x_max = (fov_x_max_idx - intercept) / slope
            if 0 <= y_at_x0 <= fov_y_max_idx:
                points.append([0, y_at_x0])
            if 0 <= y_at_x_max <= fov_y_max_idx:
                points.append([fov_x_max_idx, y_at_x_max])
        # No special handling for slope=0 here, covered by y-intersections
        
    if len(points) > 1:
        # Get unique points, ensure they are sorted to define a segment
        unique_pts_arr = np.unique(np.array(points), axis=0)
        # Sort by primary direction to get consistent segment ends
        if np.abs(slope) <= 1 and not np.isinf(slope): # Closer to horizontal, sort by y
            unique_pts_arr = unique_pts_arr[unique_pts_arr[:,1].argsort()]
        else: # Closer to vertical, sort by x
            unique_pts_arr = unique_pts_arr[unique_pts_arr[:,0].argsort()]
        return unique_pts_arr.tolist()
    elif len(points) == 1: # Line grazes a corner
        return points 
    return []

def get_beam_center(slope, intercept, fov_x_max_idx, fov_y_max_idx):
    """Calculates the geometric center of the beam segment within the FOV."""
    intersections = get_intersections(slope, intercept, fov_x_max_idx, fov_y_max_idx)
    if len(intersections) < 2:
        # Fallback: if line is outside or grazes, this center might not be ideal.
        # The centerline itself (defined by slope/intercept) is more key for profile.
        # For tangential width, center is where profile is taken.
        # Using FOV center might be too far if beam is at edge.
        # A point on the line within FOV would be better if intersections fail.
        # This function might need more robust fallback for edge cases.
        # For now, keep original fallback for consistency if it worked before.
        # print(f"Warning: Beam with slope={slope}, intercept={intercept} has < 2 intersections. Using FOV center.")
        return [FOV_X_PIXELS / 2.0, FOV_Y_PIXELS / 2.0]
    center = np.mean(np.array(intersections), axis=0)
    return center.tolist() # [center_x, center_y]

def calculate_fwhm(profile, interpolation_factor=PROFILE_INTERPOLATION_FACTOR):
    """Calculates FWHM from an intensity profile."""
    if profile is None or len(profile) < 2 or np.all(np.isclose(profile,0)) or np.max(profile) <=0:
        return 0.0
    
    # Normalize profile to avoid issues with very small max values if they are close to zero
    profile_norm = profile / np.max(profile) if np.max(profile) > 1e-9 else profile

    profile_interpolated = sp.ndimage.zoom(profile_norm, interpolation_factor, order=1)
    max_val_interp = np.max(profile_interpolated) # Max is now 1.0 if normalization worked
    if max_val_interp <= 1e-9 : return 0.0 # Still zero after normalization and zoom

    half_max = max_val_interp / 2.0
    indices_above_half = np.where(profile_interpolated >= half_max)[0]

    if len(indices_above_half) == 0 : # Nothing is above half max (e.g. single point peak)
        return 0.0
    if len(indices_above_half) < 2: # Or if only one point is above half_max, FWHM is effectively sub-pixel
        # A very narrow peak, could return 1/interpolation_factor or 0
        return 1.0 / interpolation_factor if len(profile) > 0 else 0.0


    fwhm_interpolated_indices = indices_above_half[-1] - indices_above_half[0]
    fwhm_pixels = fwhm_interpolated_indices / interpolation_factor
    return fwhm_pixels


# --- Main Processing Function ---

def calculate_volumes_for_rotation_new(hdf5_fname, params_fname, pbar, task_detectors):
    """
    Loads data for one rotation, calculates tangential AND RADIAL FWHM,
    and then the new beam volume for each beam.
    Returns original PPDFs (for numerator in next step), and new beam volumes.
    """
    fov_pixels_flat = FOV_X_PIXELS * FOV_Y_PIXELS
    beam_volumes_new = None # V_i,b_new = FWHM_tan * 1.0 * FWHM_rad
    original_ppdfs_for_numerator = None # Will store the ppdfs_flat for this rotation

    fov_x_max_idx = FOV_X_PIXELS - 1
    fov_y_max_idx = FOV_Y_PIXELS - 1

    try:
        with h5py.File(hdf5_fname, "r") as f:
            ppdfs_flat_rot = np.copy(f["ppdfs"][:]) # Shape (n_detectors, fov_pixels_flat)
            original_ppdfs_for_numerator = ppdfs_flat_rot # Store for return
            n_detectors, flat_size = ppdfs_flat_rot.shape
            if flat_size != fov_pixels_flat:
                 raise ValueError(f"Flat PPDF size mismatch in {hdf5_fname}")

        params_data = np.load(params_fname)
        beam_params_from_file = params_data['beam params'] # (n_detectors, 15, 5) [s, i, r, L_pix, T_sum]
        # filtered_ppdfs_3d = params_data['filtered ppdfs'] # Not strictly needed for V_new calculation

        if beam_params_from_file.shape[0] != n_detectors:
             raise ValueError(f"Detector count mismatch between {hdf5_fname} ({n_detectors}) and {params_fname} ({beam_params_from_file.shape[0]})")
        
        beam_volumes_new = np.zeros((n_detectors, 15), dtype=np.float32) # Store new volumes
        
        # --- Temp storage for debugging/verification ---
        # all_fwhm_tan_pix = np.zeros((n_detectors, 15))
        # all_fwhm_rad_pix = np.zeros((n_detectors, 15))

        for det_idx in range(n_detectors):
            ppdf_2d = ppdfs_flat_rot[det_idx].reshape(FOV_Y_PIXELS, FOV_X_PIXELS) # Original PPDF for this detector

            for beam_idx in range(15): # Max 15 beams per detector from old script
                params = beam_params_from_file[det_idx, beam_idx]
                slope, intercept, r, length_pixels, t_sum = params

                if r == 0 : # Skip if not a valid beam from previous segmentation/fitting
                    continue

                fwhm_tan_pixels = 0.0
                fwhm_rad_pixels = 0.0

                # --- 1. Calculate Tangential FWHM (w_pixels from old script) ---
                center_x_geom, center_y_geom = get_beam_center(slope, intercept, fov_x_max_idx, fov_y_max_idx)
                if np.isinf(slope):
                    puv_x_tan, puv_y_tan = 0, 1 # Perpendicular to vertical is horizontal
                elif np.isclose(slope, 0):
                    puv_x_tan, puv_y_tan = 1, 0 # Perpendicular to horizontal is vertical
                else:
                    # Perpendicular direction to x = slope*y + c  (or y = (x-c)/slope )
                    # Original line direction vector (dy, dx) is (1, slope) if y is independent
                    # If x = slope*y + c, direction is (slope, 1) for (dx, dy)
                    # Perpendicular is (-1, slope) or (1, -slope)
                    norm_tan = np.sqrt(1**2 + slope**2) # using (-1, slope) as perp to (slope, 1)
                    puv_x_tan = -1.0 / norm_tan
                    puv_y_tan = slope / norm_tan
                
                r0_tan = np.clip(center_y_geom - PROFILE_HALF_LENGTH_TANGENTIAL * puv_y_tan, 0, fov_y_max_idx)
                c0_tan = np.clip(center_x_geom - PROFILE_HALF_LENGTH_TANGENTIAL * puv_x_tan, 0, fov_x_max_idx)
                r1_tan = np.clip(center_y_geom + PROFILE_HALF_LENGTH_TANGENTIAL * puv_y_tan, 0, fov_y_max_idx)
                c1_tan = np.clip(center_x_geom + PROFILE_HALF_LENGTH_TANGENTIAL * puv_x_tan, 0, fov_x_max_idx)
                
                try:
                    profile_tan = ski.measure.profile_line(ppdf_2d, (r0_tan, c0_tan), (r1_tan, c1_tan),
                                                           linewidth=1, mode="constant", cval=0, order=1)
                    fwhm_tan_pixels = calculate_fwhm(profile_tan)
                except Exception: # Catch any error during profile_line
                    profile_tan = None
                    fwhm_tan_pixels = 0.0
                # all_fwhm_tan_pix[det_idx, beam_idx] = fwhm_tan_pixels


                # --- 2. Calculate Radial FWHM ---
                # Define the radial line segment using intersection points with FOV
                intersections_rad = get_intersections(slope, intercept, fov_x_max_idx, fov_y_max_idx)
                if len(intersections_rad) < 2: # Beam doesn't properly cross FOV or grazes a corner
                    fwhm_rad_pixels = 0.0 # Or use length_pixels as a fallback if desired.
                else:
                    # Start and end points of the centerline segment within FOV
                    # Intersections are [[x0,y0], [x1,y1]...] sorted
                    start_pt_rad = intersections_rad[0]  # [x,y]
                    end_pt_rad = intersections_rad[-1] # [x,y]
                    
                    # Sample PPDF along this radial line segment
                    try:
                        profile_rad = ski.measure.profile_line(
                            ppdf_2d,
                            (start_pt_rad[1], start_pt_rad[0]), # (row, col) for start
                            (end_pt_rad[1], end_pt_rad[0]),   # (row, col) for end
                            linewidth=1, mode="constant", cval=0, order=1,
                            # num=RADIAL_PROFILE_N_SAMPLES # Optional: control number of samples
                        )
                        fwhm_rad_pixels = calculate_fwhm(profile_rad)
                    except Exception: # Catch any error during profile_line
                        profile_rad = None
                        fwhm_rad_pixels = 0.0 # Or use length_pixels as a fallback
                # all_fwhm_rad_pix[det_idx, beam_idx] = fwhm_rad_pixels


                # --- 3. Calculate New Volume ---
                # V_i,b_new = (FWHM_tan_mm) * (FWHM_vert_mm=1.0*PIXEL_SIZE_MM if interpreted as 1 pixel thick) * (FWHM_rad_mm)
                # Or, if FWHM_vert is unitless 1.0:
                fwhm_tan_mm = fwhm_tan_pixels * PIXEL_SIZE_MM
                fwhm_rad_mm = fwhm_rad_pixels * PIXEL_SIZE_MM
                
                # Assuming FWHM_vertical is a unitless 1.0 as per formula discussions
                V_new_mm_units = fwhm_tan_mm * 1.0 * fwhm_rad_mm 
                beam_volumes_new[det_idx, beam_idx] = V_new_mm_units
            
            pbar.update(task_detectors, advance=1)
        
        # print(f"Max FWHM_tan for {os.path.basename(hdf5_fname)}: {np.max(all_fwhm_tan_pix) * PIXEL_SIZE_MM:.2f} mm")
        # print(f"Max FWHM_rad for {os.path.basename(hdf5_fname)}: {np.max(all_fwhm_rad_pix) * PIXEL_SIZE_MM:.2f} mm")

    except Exception as e:
        print(f"\nFATAL Error processing rotation file {os.path.basename(hdf5_fname)}: {e}")
        return None, None # Return None for PPDFs too on fatal error

    return original_ppdfs_for_numerator, beam_volumes_new


# --- Main Execution Block ---
if __name__ == "__main__":
    # Store PPDFs for each rotation to pass to the next script
    # This might consume a lot of memory if all PPDFs are kept.
    # Alternative: final script loads them one by one.
    # For now, let's try to save them rotation by rotation to avoid huge memory.
    
    pbar = Progress(
        SpinnerColumn(), BarColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(), TextColumn("[{task.completed} of {task.total}]"),
        TimeElapsedColumn(), console=None,
    )
    task_rotations = pbar.add_task("[cyan]Processing rotations...", total=N_ROTATIONS)
    task_detectors = pbar.add_task("[green]Processing PPDFs.....", total=EXPECTED_DETECTORS_PER_FILE)

    # We will save volumes rotation by rotation into a new consolidated file
    # And the final PPDS script will load original PPDFs itself.
    
    all_new_volumes_list = [] # To gather volumes from each rotation

    with pbar:
        for i in range(N_ROTATIONS):
            pbar.reset(task_detectors, description="[green]Processing PPDFs.....")
            hdf5_fname = os.path.join(INPUT_DIR, f"scanner_ppdfs_{i:02d}.hdf5")
            params_fname = os.path.join(BEAM_DIR, f"scanner_ppdfs_{i:02d}_beam_params.npz")

            if not os.path.exists(hdf5_fname) or not os.path.exists(params_fname):
                print(f"\nInput files missing for rotation {i}, skipping.")
                pbar.update(task_detectors, advance=EXPECTED_DETECTORS_PER_FILE, description="[red]Input Missing! Skip")
                all_new_volumes_list.append(None) # Placeholder for skipped rotation
                pbar.update(task_rotations, advance=1)
                continue

            print(f"\nProcessing rotation {i} for new volumes...")
            _, volumes_new_rot = calculate_volumes_for_rotation_new( # We don't need to return PPDFs from here
                hdf5_fname, params_fname, pbar, task_detectors
            )
            all_new_volumes_list.append(volumes_new_rot)
            pbar.update(task_rotations, advance=1)

    print("\nCombining new volume results...")
    valid_new_volumes = [v for v in all_new_volumes_list if v is not None]

    if not valid_new_volumes:
        print("Error: No valid rotation data processed for new volumes. Cannot save final results.")
    else:
        final_new_volumes = np.stack(valid_new_volumes, axis=0) # Shape (n_valid_rots, n_detectors, 15)
        print(f"Final New Volumes shape: {final_new_volumes.shape}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        new_volume_outfile = os.path.join(OUTPUT_DIR, "ppds_volumes_new_formula_allrots.npz")
        np.savez_compressed(new_volume_outfile, beam_volumes_new=final_new_volumes)
        print(f"Saved New Volumes (FWHM_tan * FWHM_rad based) to: {new_volume_outfile}")

    print("\nNew beam volume calculation complete.")