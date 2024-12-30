import sys

import h5py
import numpy as np
import torch
import yaml
from mpi4py import MPI

from ray3d_torch import (
    get_cuboids,
    get_fov_voxel_centers,
    get_rays,
    raytrace_torch,
    read_cuboids,
)

time_start = MPI.Wtime()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

d_cpu = torch.device("cpu")
nrots = 6
rots_interval_rad = np.pi / nrots * 2
device = d_cpu
config = yaml.safe_load(open("shifted_layer_3x3_114x114.yaml"))
geoms = torch.tensor(
    config["detector"]["detector geometry"], device=device, dtype=torch.float32
)

rshift = 93.0

device = torch.device("cpu")
nrots = 6

fname = "detector_cuboids.npz"
(
    plate_cuboids,
    crystal_cuboids,
) = read_cuboids(fname)

all_crystal_cuboids = crystal_cuboids.view(-1, 4, 3)

n_crystal_per_rank = 2
crystal_indices = torch.arange(all_crystal_cuboids.shape[0]).view(
    -1, n_crystal_per_rank
)


cuboids = torch.cat([plate_cuboids, crystal_cuboids], dim=1)
cuboids = cuboids.view(-1, 4, 3)

fov_n_voxels_xyz = torch.tensor([8, 8, 1])
mmpvx_xyz = torch.tensor([1.0, 1.0, 1.0])
voxel_centers = (
    get_fov_voxel_centers(fov_n_voxels_xyz, mmpvx_xyz, device)
    - fov_n_voxels_xyz * mmpvx_xyz * 0.5
)

pAs = voxel_centers.clone().view(-1, 3)

# partial_matrix = torch.empty(
#     (128, 0, n_crystal_per_rank), device=device, dtype=torch.float32
# )
pBs = all_crystal_cuboids[crystal_indices[rank], 0].clone().view(-1, 3)

n_pAs = pAs.shape[0]
n_pBs = pBs.shape[0]

n_cuboids = cuboids.shape[0]
f = h5py.File(f"shifted_layer_3x3_{fov_n_voxels_xyz[0]}x{fov_n_voxels_xyz[1]}.hdf5", "w", driver="mpio", comm=MPI.COMM_WORLD)
dset = f.create_dataset(
    "system matrix",
    (all_crystal_cuboids.shape[0], fov_n_voxels_xyz[0], fov_n_voxels_xyz[1]),
    dtype="f",
)


ts = raytrace_torch(pAs, pBs, cuboids, device=device)

mask_ts_0 = ts[:, :, :, 0] > 0
mask_ts_1 = ts[:, :, :, 1] > 0
absorb_mask = torch.logical_and(mask_ts_0, torch.logical_not(mask_ts_1))
attenu_mask = torch.logical_and(mask_ts_0, mask_ts_1)
diff_ts_absorb = 1 - ts[:, :, :, 0]
diff_ts_attenu = ts[:, :, :, 0] - ts[:, :, :, 1]
diff_ts_absorb[~absorb_mask] = 0
diff_ts_attenu[~attenu_mask] = 0
rays_pAs = pAs.view(-1, 1, 3).expand(-1, n_pBs, -1)
rays_pBs = pBs.view(1, -1, 3).expand(n_pAs, -1, -1)
rays_vectors = rays_pBs - rays_pAs
rays_lengths = torch.norm(rays_vectors, dim=-1)
rays_lengths_expanded = rays_lengths.unsqueeze(2).expand(-1, -1, ts.shape[-2])

config = yaml.safe_load(open("shifted_layer_3x3_114x114.yaml"))
geoms = torch.tensor(
    config["detector"]["detector geometry"], device=device, dtype=torch.float32
)
mu_tensor_plate = geoms[geoms[:, 6] == 0, 7].unsqueeze(0).expand(nrots, -1)
mu_tensor_detector = geoms[geoms[:, 6] != 0, 7].unsqueeze(0).expand(nrots, -1)

pb_cuboids_vectors = crystal_cuboids[:, :, 1:, :]
pb_cuboids_norms = (
    pb_cuboids_vectors / torch.norm(pb_cuboids_vectors, dim=-1).unsqueeze(-1)
).reshape(-1, 3, 3)
pb_cuboids_areas = torch.norm(
    torch.cross(
        pb_cuboids_vectors[:, :, [1, 2, 0], :],
        pb_cuboids_vectors[:, :, [2, 0, 1], :],
        dim=-1,
    ),
    dim=-1,
).reshape(-1, 3)
mu_tensor = (
    torch.cat([mu_tensor_plate, mu_tensor_detector], dim=1)
    .view(1, 1, -1)
    .expand(ts.shape[0], ts.shape[1], -1)
)
absorb_exponents = torch.sum(rays_lengths_expanded * diff_ts_absorb * mu_tensor, dim=-1)
attenu_exponents = torch.sum(rays_lengths_expanded * diff_ts_attenu * mu_tensor, dim=-1)
solid_angels = (
    torch.sum(
        torch.sum(
            torch.abs(
                rays_vectors.unsqueeze(2).expand(-1, -1, 3, -1)
                * pb_cuboids_norms[crystal_indices[rank]]
                .unsqueeze(0)
                .expand(rays_vectors.shape[0], -1, -1, -1),
            ),
            dim=-1,
        )
        * pb_cuboids_areas[crystal_indices[rank]]
        .unsqueeze(0)
        .expand(ts.shape[0], -1, -1),
        dim=-1,
    )
    / rays_lengths**3
)
partial_matrix = torch.movedim(
    (
        torch.exp(-attenu_exponents)
        * (1 - torch.exp(-absorb_exponents))
        * solid_angels
        / 4
        / np.pi
    ).view(fov_n_voxels_xyz[0], fov_n_voxels_xyz[1], n_crystal_per_rank),
    2,
    0,
)
print(f"Shape of partial_matrix: {partial_matrix.shape}")
# Free up memory
del (
    mask_ts_0,
    mask_ts_1,
    absorb_mask,
    attenu_mask,
    diff_ts_absorb,
    diff_ts_attenu,
    rays_vectors,
    rays_lengths,
    rays_lengths_expanded,
    mu_tensor_plate,
    mu_tensor_detector,
    pb_cuboids_vectors,
    pb_cuboids_norms,
    pb_cuboids_areas,
    mu_tensor,
    absorb_exponents,
    attenu_exponents,
    solid_angels,
)

dset[crystal_indices[rank]] = partial_matrix.cpu().numpy()
f.close()
time_end = MPI.Wtime()
print(
    f"{'Rank':16s}: {rank:04d}, {'elapsed time':16s}: {str(time_end - time_start):32s}"
)
