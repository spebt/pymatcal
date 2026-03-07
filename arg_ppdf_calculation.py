import os
import sys
import time
import h5py
import yaml
import torch
import argparse
import numpy as np
import scipy.sparse as sp
from torch import device, arange, tensor, get_num_threads

from scanner_modeling._raytracer_2d._local_functions import (
    ppdf_2d_local,
    reduced_edges_2d_local,
    sfov_properties,
    subdivision_grid_rectangle,
)
from scanner_modeling.geometry_2d import (
    fov_tensor_dict,
    load_scanner_geometry_from_layout,
    load_scanner_layouts,
)

def load_config(config_path: str) -> dict:
    """
    Loads experimental parameters from a YAML file.
    
    Args:
        config_path: Path to the .yml or .yaml configuration file.
    Returns:
        A dictionary containing the configuration.
    Raises:
        FileNotFoundError: If the configuration file is missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_ppdf_for_layout(layout_idx: int, config_path: str, save_sparse: bool = True):
    """
    Calculates the PPDF for a specific layout index.
    """
    start_time = time.time()
    
    # 1. Load Configuration
    config = load_config(config_path)
    print(f"--- Starting PPDF calculation for Layout: {layout_idx} ---")
    print(f"--- Using Configuration: {config_path} ---")
    print(f"--- Output Format: {'Sparse (Optimized)' if save_sparse else 'Dense (Legacy)'} ---")

    # Sync PyTorch threads with SLURM allocation to prevent overhead
    num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    torch.set_num_threads(num_threads)

    # 2. Setup Parameters from Config
    default_device = device("cpu")
    
    input_tensor_path = config['paths']['input_tensor']
    output_dir = config['paths']['output_dir']
    
    scanner_layout_dir = os.path.dirname(input_tensor_path)
    scanner_layout_filename = os.path.basename(input_tensor_path)

    # Geometry Loading
    scanner_layouts, _ = load_scanner_layouts(scanner_layout_dir, scanner_layout_filename)
    
    if not (0 <= layout_idx < len(scanner_layouts)):
        print(f"Error: Layout index {layout_idx} is out of bounds for the loaded file.")
        sys.exit(1)

    # Physics and FOV Geometry
    mu_dict = tensor(config['scanner']['mu_values'], device=default_device)
    
    fov_dict = fov_tensor_dict(
        tuple(config['fov']['pixels']),
        tuple(config['fov']['size_mm']),
        tuple(config['fov']['center_mm']),
        tuple(config['fov']['subdivisions'])
    )
    
    crystal_n_subs = tuple(config['scanner']['crystal_subdivisions'])
    
    # Pre-calculate SFov (Sub-FOV) properties
    sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
    fov_n_pxs = int(fov_dict["n pixels"].prod())
    n_sfov = int(fov_dict["n subdivisions"].prod())
    
    # Map pixel IDs for 1D HDF5 storage
    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )
    subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

    print(f"System Matrix: {config['fov']['pixels'][0]}x{config['fov']['pixels'][1]} px")
    print(f"Hardware: PyTorch using {get_num_threads()} threads.")

    # 3. Geometry Preparation
    (
        plate_objects_vertices, crystal_objects_vertices,
        plate_objects_edges, crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    n_crystals = crystal_objects_vertices.shape[0]
    crystal_idx_tensor = arange(n_crystals)

    # 4. Iterative Calculation
    os.makedirs(output_dir, exist_ok=True)
    matrix_rows = []

    for dataset_idx, crystal_idx_tensor_val in enumerate(crystal_idx_tensor):
        crystal_idx = int(crystal_idx_tensor_val.item())
        dense_row = np.zeros(fov_n_pxs, dtype=np.float32)
        
        reduced_crystal_edges_sfovs = []
        reduced_plate_edges_sfovs = []
        for sfov_idx in range(n_sfov):
            red_plate, red_crys = reduced_edges_2d_local(
                sfov_idx, crystal_idx, sfov_corners_batch,
                plate_objects_vertices, plate_objects_edges,
                crystal_objects_vertices, crystal_objects_edges,
                default_device,
            )
            reduced_crystal_edges_sfovs.append(red_crys)
            reduced_plate_edges_sfovs.append(red_plate)

        for sfov_idx in range(n_sfov):
            ppdf_slice = ppdf_2d_local(
                sfov_idx, crystal_idx, sfov_pixels_batch,
                crystal_objects_vertices, reduced_plate_edges_sfovs[sfov_idx],
                reduced_crystal_edges_sfovs[sfov_idx], subdivision_grid,
                mu_dict, default_device,
            )
            dense_row[sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()
            
        # Append to our list in either sparse or dense format
        if save_sparse:
            dense_row[dense_row < 1e-9] = 0.0 
            matrix_rows.append(sp.csr_matrix(dense_row))
        else:
            matrix_rows.append(dense_row)

    # 5. Single I/O Write Step
    h5_file_path = os.path.join(output_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
    
    with h5py.File(h5_file_path, "w") as h5file:
        if save_sparse:
            full_sparse_matrix = sp.vstack(matrix_rows)
            h5file.create_dataset("data", data=full_sparse_matrix.data, compression="gzip")
            h5file.create_dataset("indices", data=full_sparse_matrix.indices, compression="gzip")
            h5file.create_dataset("indptr", data=full_sparse_matrix.indptr, compression="gzip")
            h5file.attrs["shape"] = full_sparse_matrix.shape
        else:
            full_dense_matrix = np.vstack(matrix_rows)
            h5file.create_dataset("ppdfs", data=full_dense_matrix)

    elapsed = time.time() - start_time
    print(f"--- Finished Layout {layout_idx} in {elapsed:.2f}s ---")
    print(f"Output saved to: {h5_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PPDF for a specific scanner layout.")
    
    # Required positional argument
    parser.add_argument("layout_idx", type=int, help="The integer index of the layout to process.")
    
    # Optional named argument with default path
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base_config.yml", 
        help="Path to the configuration YAML file."
    )
    # NEW FLAG: Use --dense to force legacy saving behavior
    parser.add_argument(
        "--dense", 
        action="store_true", 
        help="Save output in the legacy dense HDF5 format instead of sparse."
    )
    
    args = parser.parse_args()

    try:
        # If --dense is provided, save_sparse becomes False
        calculate_ppdf_for_layout(args.layout_idx, args.config, save_sparse=not args.dense)
    except Exception as e:
        print(f"Execution Error: {e}")
        sys.exit(1)