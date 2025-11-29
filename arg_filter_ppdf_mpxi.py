import torch
import h5py
import os
import numpy as np
from rich.progress import track
import sys

def filter_all_layouts_by_mpxi():
    # --- 1. Configuration ---
    # Adjust these paths and values as needed.

    # Base directory where your data is stored
    base_dir = "../data/mph_hourglass_single_position_base_2mm_20pinholes_rotated_18custom_rot"
    
    # Directory containing the original 'outputs' (properties, masks, ppdfs)
    input_dir = os.path.join(base_dir, "outputs")

    # !! New directory to save the filtered PPDFs !!
    # This script will create this directory if it doesn't exist.
    filtered_output_dir = os.path.join(base_dir, "filtered_outputs/mpxi_2")
    
    # --- NEW: Multiplexing Filter Configuration ---
    # !! CHECK THIS VALUE !! Look at the 'Header' attribute in your beams_properties file
    MULTIPLEX_COLUMN_INDEX = 10
    DESIRED_MULTIPLEX_VALUE = 2

    # Total number of layouts to process
    TOTAL_LAYOUTS = 18

    # --- 2. Main Processing ---
    os.makedirs(filtered_output_dir, exist_ok=True)
    print(f"Filtered PPDFs will be saved to: {filtered_output_dir}")
    print(f"Keeping beams with mpxi == {DESIRED_MULTIPLEX_VALUE}.")

    # Loop over all scanner positions (layouts)
    for layout_idx in track(range(TOTAL_LAYOUTS), description="Filtering PPDFs..."):
        try:
            # --- Load Data for the current Layout ---
            # 1. Load Beam Properties and Header
            props_fname = os.path.join(input_dir, f"beams_properties_configuration_{layout_idx:02d}.hdf5")
            with h5py.File(props_fname, "r") as f:
                properties = torch.from_numpy(f["beam_properties"][:])
                # Robustly handle header stored as bytes or strings
                header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in f["beam_properties"].attrs["Header"]]

            # 2. Load Beam Masks
            masks_fname = os.path.join(input_dir, f"beams_masks_configuration_{layout_idx:02d}.hdf5")
            with h5py.File(masks_fname, "r") as f:
                masks = torch.from_numpy(f["beam_mask"][:])

            # 3. Load Original PPDFs
            ppdfs_fname = os.path.join(input_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
            with h5py.File(ppdfs_fname, "r") as f:
                original_ppdfs = torch.from_numpy(f["ppdfs"][:])

        except FileNotFoundError as e:
            print(f"\n⚠️ Skipping layout {layout_idx}, file not found: {e.filename}")
            continue

            
        det_idx_col = header.index('detector unit id')
        beam_idx_col = header.index('beam id')

        mpxi_values = properties[:, MULTIPLEX_COLUMN_INDEX]
        criteria_mask = (mpxi_values == DESIRED_MULTIPLEX_VALUE)
        
        good_beams_properties = properties[criteria_mask]
        
        if good_beams_properties.shape[0] == 0:
            # print(f"\nℹ️ No beams found for layout {layout_idx} with MPXI = {DESIRED_MULTIPLEX_VALUE}. Skipping.")
            # We still need to save an empty PPDF file to be consistent
            pass

        # --- Create a New Combined Mask from "Good" Beams Only ---
        n_detectors = masks.shape[0]
        final_filtered_mask = torch.zeros_like(masks, dtype=torch.bool)

        for i in range(n_detectors):
            # Find which of the "good" beams belong to the current detector unit
            beams_for_this_detector = good_beams_properties[good_beams_properties[:, det_idx_col] == i]
            
            if len(beams_for_this_detector) > 0:
                # Get the original beam IDs for the good beams
                good_beam_indices = beams_for_this_detector[:, beam_idx_col].long()
                
                # Create a boolean mask for this detector that is True only for pixels
                # that belong to one of the good beams.
                detector_mask = torch.isin(masks[i], good_beam_indices)
                final_filtered_mask[i] = detector_mask

        # --- Apply the Final Mask and Save the New PPDF ---
        # Element-wise multiplication zeros out the pixels from rejected beams
        filtered_ppdfs = original_ppdfs * final_filtered_mask.float()
        
        # Save the new filtered PPDF file
        output_fname = os.path.join(filtered_output_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
        with h5py.File(output_fname, "w") as f:
            f.create_dataset("ppdfs", data=filtered_ppdfs.numpy())

    print("\n🚀 Filtering complete for all layouts!")

if __name__ == "__main__":
    filter_all_layouts_by_mpxi()