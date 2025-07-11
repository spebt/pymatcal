# calculate_ppdf_for_system.py

import torch
import time
import h5py
import os
from tqdm import tqdm  # NEW: progress bar
from torch import empty as empty_tensor, tensor, zeros, Tensor

# Import the necessary functions from your raytracer library
from raytracer_2d import (
    get_fov_pixel_centers_2d,
    get_convex_hull_2d,
    get_edges_from_verts_2d,
    get_cuts_rays_on_polygons_2d,
    get_angular_terms_2d
)

def calculate_ppdf_for_detector(detector_idx, sim_params):
    fov_voxels = sim_params['fov_voxels']
    detector_centers = sim_params['detector_centers']
    detector_verts = sim_params['detector_verts']
    collimator_verts = sim_params['collimator_verts']
    mu_detector = sim_params['mu_detector']
    mu_collimator = sim_params['mu_collimator']

    rays = torch.stack([fov_voxels, detector_centers[detector_idx].expand_as(fov_voxels)], dim=1)
    rays_lengths = torch.norm(rays[:, 1] - rays[:, 0], dim=1)

    fov_corners = get_convex_hull_2d(fov_voxels)
    cone_of_view = get_convex_hull_2d(torch.vstack((detector_centers[detector_idx].unsqueeze(0), fov_corners)))

    hull_min, _ = torch.min(cone_of_view, dim=0)
    hull_max, _ = torch.max(cone_of_view, dim=0)
    coll_min, _ = torch.min(collimator_verts, dim=1)
    coll_max, _ = torch.max(collimator_verts, dim=1)

    candidate_indices = torch.where(
        (coll_max[:, 0] >= hull_min[0]) & (coll_min[:, 0] <= hull_max[0]) &
        (coll_max[:, 1] >= hull_min[1]) & (coll_min[:, 1] <= hull_max[1])
    )[0]

    reduced_collimator_verts = collimator_verts[candidate_indices]

    target_detector_edges = get_edges_from_verts_2d(detector_verts[detector_idx].unsqueeze(0))
    t_absorb_pairs = get_cuts_rays_on_polygons_2d(rays, target_detector_edges)
    path_length_absorb = torch.abs(t_absorb_pairs[:, 0, 1] - t_absorb_pairs[:, 0, 0]) * rays_lengths
    absorption_term = 1.0 - torch.exp(-path_length_absorb * mu_detector)

    attenuation_term = torch.ones_like(rays_lengths, dtype=torch.float64)
    if reduced_collimator_verts.shape[0] > 0:
        reduced_collimator_edges = get_edges_from_verts_2d(reduced_collimator_verts)
        t_attenu_pairs = get_cuts_rays_on_polygons_2d(rays, reduced_collimator_edges)
        path_lengths_attenu = torch.abs(t_attenu_pairs[:, :, 1] - t_attenu_pairs[:, :, 0]) * rays_lengths.unsqueeze(1)
        dl_mu_attenu = torch.sum(path_lengths_attenu * mu_collimator, dim=1)
        attenuation_term = torch.exp(-dl_mu_attenu)

    geometric_term = get_angular_terms_2d(rays, rays_lengths, target_detector_edges.squeeze(0)[:2])

    return (absorption_term * attenuation_term * geometric_term).float()

if __name__ == "__main__":
    fov_config = {
        "n_pixels": tensor([256, 256]),
        "mm_per_pixel": tensor([0.5, 0.5]),
        "center": tensor([0.0, 0.0]),
    }

    mu_config = {
        "detector": 0.475,
        "collimator": 3.5,
    }

    input_filename = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/nonmux/full_system_4cecbf37567bd8514932255ca6f35d00.tensor"
    output_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/nonmux/outputs_new/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading system geometry from: {input_filename}")
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"File {input_filename} does not exist.")

    system_data = torch.load(input_filename)
    detector_verts = system_data['detector units'].to("cpu")
    collimator_verts = system_data['collimator plates'].to("cpu")

    n_detectors = detector_verts.shape[0]
    print(f"Loaded {n_detectors} detectors and {collimator_verts.shape[0]} collimator plates.")

    sim_params = {
        'fov_voxels': get_fov_pixel_centers_2d(
            fov_config['n_pixels'], fov_config['mm_per_pixel'], fov_config['center']
        ).view(-1, 2),
        'detector_verts': detector_verts,
        'detector_centers': detector_verts.mean(dim=1),
        'collimator_verts': collimator_verts,
        'mu_detector': mu_config['detector'],
        'mu_collimator': mu_config['collimator']
    }

    fov_n_pixels = sim_params['fov_voxels'].shape[0]
    output_hdf5_filename = os.path.join(output_dir, "ppdf_results.hdf5")

    print(f"\nStarting PPDF calculation for {n_detectors} detectors.")
    print(f"Output will be saved to: {output_hdf5_filename}")

    with h5py.File(output_hdf5_filename, "w") as out_file:
        ppdf_dataset = out_file.create_dataset(
            "ppdfs", 
            shape=(n_detectors, fov_n_pixels), 
            dtype="f4",
            chunks=(1, fov_n_pixels)
        )

        start_time_total = time.time()
        for i in tqdm(range(n_detectors), desc="Calculating PPDFs"):
            ppdf_row = calculate_ppdf_for_detector(i, sim_params)
            ppdf_dataset[i, :] = ppdf_row.numpy()

    elapsed_total = time.time() - start_time_total
    print("\nCalculation complete.")
    print(f"Total time: {elapsed_total:.2f} seconds")
    print(f"Average time per detector: {elapsed_total / n_detectors:.4f} seconds")
