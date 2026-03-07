import torch
import h5py
import os
import numpy as np
import gc
from rich.progress import track
import sys

# --- NEW: Auto-detecting System Matrix Loader ---
def load_system_matrix_dict(h5_path: str) -> dict:
    """Bypasses PyTorch beta sparse tensors to prevent memory leaks."""
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            return {
                "is_sparse": True,
                "indptr": torch.tensor(h5f["indptr"][:], dtype=torch.int64),
                "indices": torch.tensor(h5f["indices"][:], dtype=torch.int64),
                "data": torch.tensor(h5f["data"][:], dtype=torch.float32),
                "shape": tuple(h5f.attrs["shape"])
            }
        elif "ppdfs" in h5f:
            return {
                "is_sparse": False,
                "dense_tensor": torch.tensor(h5f["ppdfs"][:], dtype=torch.float32)
            }
        else:
            raise ValueError(f"Unknown matrix format in {h5_path}")

# --- NEW: Auto-detecting Mask Matrix Loader ---
def load_mask_matrix_dict(h5_path: str) -> dict:
    """Bypasses PyTorch beta sparse tensors for integer masks."""
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            return {
                "is_sparse": True,
                "indptr": torch.tensor(h5f["indptr"][:], dtype=torch.int64),
                "indices": torch.tensor(h5f["indices"][:], dtype=torch.int64),
                "data": torch.tensor(h5f["data"][:], dtype=torch.int32),
                "shape": tuple(h5f.attrs["shape"])
            }
        elif "beam_mask" in h5f:
            return {
                "is_sparse": False,
                "dense_tensor": torch.tensor(h5f["beam_mask"][:], dtype=torch.int32)
            }
        else:
            raise ValueError(f"Unknown mask format in {h5_path}")


def filter_all_layouts_by_mpxi():
    # --- 1. Configuration ---
    # Adjust these paths and values as needed.

    # Base directory where your data is stored
    base_dir = "../data/mph_hourglass_single_position_base_2mm_20pinholes_rotated_18custom_rot"
    
    # Directory containing the original 'outputs' (properties, masks, ppdfs)
    input_dir = os.path.join(base_dir, "outputs")

    # !! New directory to save the filtered PPDFs !!
    filtered_output_dir = os.path.join(base_dir, "filtered_outputs/mpxi_2")
    
    # --- NEW: Multiplexing Filter Configuration ---
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
                header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in f["beam_properties"].attrs["Header"]]

            # 2. Load Beam Masks (Using Dict bypass)
            masks_fname = os.path.join(input_dir, f"beams_masks_configuration_{layout_idx:02d}.hdf5")
            masks_dict = load_mask_matrix_dict(masks_fname)

            # 3. Load Original PPDFs (Using Dict bypass)
            ppdfs_fname = os.path.join(input_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
            ppdfs_dict = load_system_matrix_dict(ppdfs_fname)

        except FileNotFoundError as e:
            print(f"\n⚠️ Skipping layout {layout_idx}, file not found: {e.filename}")
            continue
            
        det_idx_col = header.index('detector unit id')
        beam_idx_col = header.index('beam id')

        mpxi_values = properties[:, MULTIPLEX_COLUMN_INDEX]
        criteria_mask = (mpxi_values == DESIRED_MULTIPLEX_VALUE)
        
        good_beams_properties = properties[criteria_mask]
        
        # Determine matrix dimensions
        is_sparse_output = ppdfs_dict["is_sparse"]
        if is_sparse_output:
            n_detectors, n_voxels = ppdfs_dict["shape"]
        else:
            n_detectors, n_voxels = ppdfs_dict["dense_tensor"].shape

        output_fname = os.path.join(filtered_output_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")

        # --- Process Row-by-Row to prevent Memory Leaks ---
        with torch.no_grad():
            
            # Setup output structures based on desired format
            if is_sparse_output:
                out_data, out_indices, out_indptr = [], [], [0]
            else:
                out_h5 = h5py.File(output_fname, "w")
                out_ds = out_h5.create_dataset("ppdfs", shape=(n_detectors, n_voxels), dtype="float32", chunks=(1, n_voxels), compression="gzip")

            for i in range(n_detectors):
                # 1. Extract Mask Row
                if masks_dict["is_sparse"]:
                    mask_row = torch.zeros(n_voxels, dtype=torch.int32)
                    m_start, m_end = int(masks_dict["indptr"][i]), int(masks_dict["indptr"][i+1])
                    if m_start < m_end:
                        mask_row[masks_dict["indices"][m_start:m_end]] = masks_dict["data"][m_start:m_end]
                else:
                    mask_row = masks_dict["dense_tensor"][i]

                # 2. Extract PPDF Row
                if ppdfs_dict["is_sparse"]:
                    ppdf_row = torch.zeros(n_voxels, dtype=torch.float32)
                    p_start, p_end = int(ppdfs_dict["indptr"][i]), int(ppdfs_dict["indptr"][i+1])
                    if p_start < p_end:
                        ppdf_row[ppdfs_dict["indices"][p_start:p_end]] = ppdfs_dict["data"][p_start:p_end]
                else:
                    ppdf_row = ppdfs_dict["dense_tensor"][i]

                # 3. Apply the filtering logic
                beams_for_this_detector = good_beams_properties[good_beams_properties[:, det_idx_col] == i]
                
                if len(beams_for_this_detector) > 0:
                    good_beam_indices = beams_for_this_detector[:, beam_idx_col].long()
                    detector_mask = torch.isin(mask_row, good_beam_indices)
                    filtered_row = ppdf_row * detector_mask.float()
                else:
                    filtered_row = torch.zeros(n_voxels, dtype=torch.float32)

                # 4. Save the Row
                if is_sparse_output:
                    nz_idx = torch.nonzero(filtered_row, as_tuple=False).squeeze(1)
                    if nz_idx.numel() > 0:
                        out_data.append(filtered_row[nz_idx])
                        out_indices.append(nz_idx)
                    out_indptr.append(out_indptr[-1] + nz_idx.numel())
                else:
                    out_ds[i, :] = filtered_row.numpy()

            # 5. Finalize and write sparse format to disk
            if is_sparse_output:
                with h5py.File(output_fname, "w") as f_out:
                    if len(out_data) > 0:
                        f_out.create_dataset("data", data=torch.cat(out_data).numpy(), compression="gzip")
                        f_out.create_dataset("indices", data=torch.cat(out_indices).int().numpy(), compression="gzip")
                    else:
                        f_out.create_dataset("data", data=np.array([], dtype=np.float32))
                        f_out.create_dataset("indices", data=np.array([], dtype=np.int32))
                    
                    f_out.create_dataset("indptr", data=np.array(out_indptr, dtype=np.int32))
                    f_out.attrs["shape"] = [n_detectors, n_voxels]
            else:
                out_h5.close()
                
            gc.collect()

    print("\n🚀 Filtering complete for all layouts!")

if __name__ == "__main__":
    filter_all_layouts_by_mpxi()