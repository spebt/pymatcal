from torch import Tensor, stack
from .._config import DTYPE



def rays_3d_batch(pa_batch: Tensor, pb_batch: Tensor) -> Tensor:
    """
    3D analogue of rays_2d_batch.

    Form rays from two sets of points a and b in 3D.

    Inputs
    ------
    pa_batch: Tensor, shape (N_pa, 3) or (B, N_pa, 3)
        FOV voxel centers, for example.

    pb_batch: Tensor, shape (N_pb, 3) or (B, N_pb, 3)
        Detector voxel centers, for example.

    Returns
    -------
    Tensor of shape (N_pa, N_pb, 2, 3) for unbatched inputs, or
    (B, N_pa, N_pb, 2, 3) for batched inputs, where the last dimension is:
        [..., 0, :] = origins (points a)
        [..., 1, :] = endpoints (points b)

    NOTE:
    - All coordinates are converted to DTYPE and assumed to be in mm.
    - If one input is batched and the other isn’t, the output is batched.
    """
    pa_batch = pa_batch.to(dtype=DTYPE)
    pb_batch = pb_batch.to(dtype=DTYPE)

    if pa_batch.ndim == 2 and pb_batch.ndim == 2:
        # Unbatched case
        N_pa = pa_batch.shape[0]
        N_pb = pb_batch.shape[0]
        pa_expanded = pa_batch.unsqueeze(1).expand(N_pa, N_pb, 3)
        pb_expanded = pb_batch.unsqueeze(0).expand(N_pa, N_pb, 3)
        return stack((pa_expanded, pb_expanded), dim=2)  # (N_pa, N_pb, 2, 3)

    elif pa_batch.ndim == 3 and pb_batch.ndim == 2:
        # pa batched, pb unbatched
        B, N_pa, _ = pa_batch.shape
        N_pb = pb_batch.shape[0]
        pb_batched = pb_batch.unsqueeze(0).expand(B, N_pb, 3)
        pa_expanded = pa_batch.unsqueeze(2).expand(B, N_pa, N_pb, 3)
        pb_expanded = pb_batched.unsqueeze(1).expand(B, N_pa, N_pb, 3)
        return stack((pa_expanded, pb_expanded), dim=3)  # (B, N_pa, N_pb, 2, 3)

    elif pa_batch.ndim == 2 and pb_batch.ndim == 3:
        # pb batched, pa unbatched
        B, N_pb, _ = pb_batch.shape
        N_pa = pa_batch.shape[0]
        pa_batched = pa_batch.unsqueeze(0).expand(B, N_pa, 3)
        pa_expanded = pa_batched.unsqueeze(2).expand(B, N_pa, N_pb, 3)
        pb_expanded = pb_batch.unsqueeze(1).expand(B, N_pa, N_pb, 3)
        return stack((pa_expanded, pb_expanded), dim=3)  # (B, N_pa, N_pb, 2, 3)

    elif pa_batch.ndim == 3 and pb_batch.ndim == 3:
        # Both batched
        if pa_batch.shape[0] != pb_batch.shape[0]:
            raise ValueError("Batch sizes of pa_batch and pb_batch must match")
        B, N_pa, _ = pa_batch.shape
        _, N_pb, _ = pb_batch.shape
        pa_expanded = pa_batch.unsqueeze(2).expand(B, N_pa, N_pb, 3)
        pb_expanded = pb_batch.unsqueeze(1).expand(B, N_pa, N_pb, 3)
        return stack((pa_expanded, pb_expanded), dim=3)

    else:
        raise ValueError(
            f"pa_batch and pb_batch must have ndim 2 or 3; got {pa_batch.ndim}, {pb_batch.ndim}"
        )


def ray_directions_from_points(pa_batch: Tensor, pb_batch: Tensor) -> Tensor:
    """
    Compute ray directions from two sets of 3D points.

    Parameters
    ----------
    pa_batch : Tensor, shape (Na, 3)
        Ray start points.

    pb_batch : Tensor, shape (Nb, 3)
        Ray end points.

    Returns
    -------
    directions : Tensor, shape (Na, Nb, 3)
        directions[i, j, :] = pb_batch[j, :] - pa_batch[i, :]

    Notes
    -----
    Directions are NOT normalized; normalization can be done by the caller
    if needed.
    """
    if pa_batch.ndim != 2 or pa_batch.shape[1] != 3:
        raise ValueError(
            f"pa_batch must have shape (Na, 3), got {tuple(pa_batch.shape)}"
        )
    if pb_batch.ndim != 2 or pb_batch.shape[1] != 3:
        raise ValueError(
            f"pb_batch must have shape (Nb, 3), got {tuple(pb_batch.shape)}"
        )

    pa_batch = pa_batch.to(dtype=DTYPE)
    pb_batch = pb_batch.to(dtype=DTYPE)

    # (Na, 1, 3) and (1, Nb, 3) broadcast to (Na, Nb, 3)
    return pb_batch.unsqueeze(0) - pa_batch.unsqueeze(1)
