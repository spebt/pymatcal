import numpy as np
import torch
import yaml
import pandas as pd

from ray3d_torch import (
    get_cuboids,
)

d_cpu = torch.device("cpu")
nrots = 6
rots_interval_rad = np.pi / nrots * 2
compute_device = d_cpu
config = yaml.safe_load(open("shifted_layer_3x3_114x114.yaml"))
geoms = torch.tensor(
    config["detector"]["detector geometry"], device=compute_device, dtype=torch.float32
)

rshift = 93.0

plate_geoms = geoms[geoms[:, 6] == 0]
plate_cuboids_rots_xyz = torch.arange(
    0, nrots, device=compute_device, dtype=torch.float32
).unsqueeze(-1).unsqueeze(-1).repeat(1, plate_geoms.shape[0], 3) * torch.tensor(
    [0, 0, rots_interval_rad], device=compute_device, dtype=torch.float32
)

plate_cuboids_centers_xyz, plate_cuboids_vectors_xyz = get_cuboids(
    plate_geoms, plate_cuboids_rots_xyz, rshift, compute_device
)

# Detector units

detector_geoms = geoms[geoms[:, 6] != 0]
detector_cuboids_rots_xyz = torch.arange(
    0, nrots, device=compute_device, dtype=torch.float32
).unsqueeze(-1).unsqueeze(-1).repeat(1, detector_geoms.shape[0], 3) * torch.tensor(
    [0, 0, rots_interval_rad], device=compute_device, dtype=torch.float32
)

detector_cuboids_centers_xyz, detector_cuboids_vectors_xyz = get_cuboids(
    detector_geoms, detector_cuboids_rots_xyz, rshift, compute_device
)

plate_cuboids = torch.cat(
    (plate_cuboids_centers_xyz.unsqueeze(2), plate_cuboids_vectors_xyz),
    dim=2,
)

detector_cuboids = torch.cat(
    (
        detector_cuboids_centers_xyz.unsqueeze(2),
        detector_cuboids_vectors_xyz,
    ),
    dim=2,
)


output = {
    "plate cuboids": plate_cuboids.cpu().numpy().astype(np.float32),
    "crystal cuboids": detector_cuboids.cpu().numpy().astype(np.float32),
}

np.savez_compressed("detector_cuboids.npz", **output)
