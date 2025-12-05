import torch
from torch import Tensor
from .._config import DTYPE

def ray_obb_intersection_local(
    o_local: Tensor,     # (..., 3) mm
    d_local: Tensor,     # (..., 3) mm
    half_sizes: Tensor,  # (..., 3) mm
):
    """
    Exact slab intersection with an oriented bounding box in *local* coordinates.

    Computes (t_enter, t_exit, hit_mask).

    hit_mask = ray actually intersects a finite segment.

    Returns
    -------
    t_enter, t_exit : Tensor
        Ray parameters along d_local, not physical distances.
        Physical path length inside the OBB is (t_exit - t_enter) * ||d_local||.
    hit_mask : Tensor
        Boolean mask where intersection is valid.
    """

    # Avoid division-by-zero by replacing 0 with tiny epsilon
    device = o_local.device
    eps = torch.finfo(DTYPE).eps
    inv_d = 1.0 / torch.where(d_local.abs() < eps, torch.full_like(d_local, eps), d_local)

    t1 = (-half_sizes - o_local) * inv_d   # (..., 3)
    t2 = ( half_sizes - o_local) * inv_d   # (..., 3)

    t_min = torch.minimum(t1, t2)          # per-axis entry
    t_max = torch.maximum(t1, t2)          # per-axis exit

    # Ray enters after the latest per-axis entry
    t_enter = t_min.max(dim=-1).values
    # Ray exits before the earliest per-axis exit
    t_exit  = t_max.min(dim=-1).values

    # VALID iff exit after enter and exit >= 0
    zero = torch.zeros((), dtype=DTYPE, device=device)
    hit_mask = t_exit >= torch.maximum(t_enter, zero)

    return t_enter, t_exit, hit_mask
