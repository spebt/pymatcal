from typing import Dict
import torch
from torch import Tensor, arange, argwhere, atan2, cat, cos, diff
from torch import empty as empty_tensor, uint16 as uint16_tensor
from torch import (
    float32,
    gradient,
    linspace,
    logical_and,
    meshgrid,
    norm,
    pi,
    sin,
    stack,
    tensor,
    where,
    zeros,
    zeros_like,
)
from torch.nn import functional as F

from geometry_2d_utils import local_max_1d


def get_arc_nodes_2d(
    convex_hull: Tensor,
    crystal_center: Tensor,
):
    """
    Get the arc nodes for the spiral sampling.
    The second node controls the curvature of the spiral.
    """

    # find the closest point in the convex hull (excluding the crystal center) to the center of the crystal

    distances = norm(convex_hull - crystal_center, dim=1)
    indices = distances.argsort()
    nodes = convex_hull[indices][1:3]

    rads = atan2(
        nodes[:, 1] - crystal_center[1],
        nodes[:, 0] - crystal_center[0],
    )
    rads = rads + 2 * pi * (rads < 0)
    distances = distances[indices][1:3]
    return stack([distances, rads], dim=1)


def get_sampling_arc_coordinates_2d(nodes, n=100):
    """Get spiral sampling coordinates in 2D."""
    ends = nodes[nodes[:, 1].argsort()]
    if ends[1, 1] - ends[0, 1] > pi:
        ends[1, 1] = ends[1, 1] - 2 * pi
        angles = linspace(ends[1, 1], ends[0, 1], n)
        radius = linspace(ends[1, 0], ends[0, 0], n)
    else:
        angles = linspace(ends[0, 1], ends[1, 1], n)
        radius = linspace(ends[0, 0], ends[1, 0], n)
    x = radius * cos(angles)
    y = radius * sin(angles)
    return stack((x, y), dim=-1), angles


def sample_ppdf_on_arc_2d_local(
    ppdf_data_2d: Tensor,
    detector_center: Tensor,
    hull_2d: Tensor,
    fov_dict: dict,
    n_samples: int = 1024,
):
    arc_nodes = get_arc_nodes_2d(
        hull_2d,
        detector_center,
    )
    # Get the sampling points and angles
    sampling_points, sampling_rads = get_sampling_arc_coordinates_2d(
        arc_nodes,
        n=n_samples,
    )
    sampling_points = sampling_points + detector_center

    # Prepare the input and grid for sampling

    # 2D PPDF field
    ppdf_field = ppdf_data_2d.T.view(
        1, 1, int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
    ).to(float32)

    # The grid contains the coordinates of the points where the input scalar
    # field is sampled. The grid is in the range [-1, 1] for both x and y.

    grid = (
        sampling_points.unsqueeze(0).unsqueeze(0).to(float32)
        / fov_dict["size in mm"]
        * 2
    )

    sampled_ppdf = F.grid_sample(
        ppdf_field, grid, mode="bilinear", align_corners=True
    ).squeeze()
    return sampled_ppdf, sampling_rads, sampling_points


def get_beams_angle_radian(
    beams_centers: Tensor,
    reference_point: Tensor,
) -> Tensor:
    n_beams = beams_centers.shape[0]
    reference_point_expanded = reference_point.unsqueeze(0).expand(n_beams, -1)
    angles = atan2(
        reference_point_expanded[:, 1] - beams_centers[:, 1],
        reference_point_expanded[:, 0] - beams_centers[:, 0],
    )
    angles = angles + 2 * pi * (angles < 0).float()
    return angles


def get_beams_weighted_center(
    beams_masks: Tensor,
    pixels_xys: Tensor,
    ppdf_data: Tensor,
) -> Tensor:
    """
    Compute the weighted center of the beams based on the pixel data and beam masks.

    Parameters
    ----------
    beam_masks: Tensor
      - The masks for the beams.
      - `shape`: (n_pixels, n_beams)
      - `dtype`: `bool`


    pixels_xys: Tensor
      The field of view points.

    ppdf_data: Tensor
      The pixel data probability distribution function.

    beams_relative_sensitivities: Tensor
      The means of the beams.
    """
    n_beams = beams_masks.shape[0]
    pixels_xys_expanded = pixels_xys.unsqueeze(0).expand(n_beams, -1, -1)
    ppdf_expanded = (
        ppdf_data.view(-1).unsqueeze(0).unsqueeze(2).expand(n_beams, -1, 2)
    )

    xy_weighted_sum = (
        (pixels_xys_expanded * ppdf_expanded)
        .masked_fill_(~beams_masks.unsqueeze(2).expand(-1, -1, 2), 0)
        .sum(dim=1)
    )
    total_weights = (
        (ppdf_data.view(-1).unsqueeze(0).expand(n_beams, -1))
        .clone()
        .masked_fill_(~beams_masks, 0)
        .sum(dim=1)
    )
    return xy_weighted_sum / total_weights.unsqueeze(1)


def full_width_half_maximum_1d_batch(
    data_x_batch: Tensor,
    data_y_batch: Tensor,
):
    """
    Calculate the full width at half maximum (FWHM) for each curve in the batch.
    The FWHM is defined as the distance between the two points where the curve
    crosses half of the maximum value.

    The function returns the x-coordinates of the two points and the FWHM value.

    Parameters
    ----------

    data_x_batch: Tensor
      The x-coordinates of the data points.
      `shape`: (n_batch, n_points)
      `dtype`: `float32`

    data_y_batch: Tensor
      The y-coordinates of the data points.
      `shape`: (n_batch, n_points)
      `dtype`: `float32`

    Returns
    ----------

    fwhm_batch: Tensor
      The FWHM values for each curve in the batch.
      `shape`: (n_batch,)
      `dtype`: `float32`

    x_bounds_batch: Tensor
        The x-coordinates of the two points where the curve crosses half of the maximum.
        `shape`: (n_batch, 2)
        `dtype`: `float32`
    """

    n_batch = data_x_batch.shape[0]
    n_points = data_x_batch.shape[1]

    # Find the indices of the points where the curve crosses half of the maximum
    half_max = (
        (data_y_batch.max(dim=1).values * 0.5).unsqueeze(1).expand(-1, n_points)
    )
    x_above_half_max_batch = (
        where(data_y_batch >= half_max, data_x_batch, 0).sort(dim=1).values
    )
    x_upper_lower_batch = x_above_half_max_batch[:, [0, -1]]
    fwhm_batch = x_upper_lower_batch[:, 1] - x_upper_lower_batch[:, 0]
    return fwhm_batch, x_upper_lower_batch


def beam_sampling_line_batch(
    detector_unit_center: Tensor,
    beam_center_batch: Tensor,
    n_samples: int = 1024,
    length: float = 32.0,
):
    """
    Get the sampling line for the beams in batch.
    The sampling line is perpendicular to the beam axis and passes through the
    beam center.

    Parameters
    ----------

    detector_unit_center: Tensor
      The center of the detector unit.
      `shape`: (2,)
      `dtype`: `float32`

    beam_center_batch: Tensor
      The weighted centers of the beams.
      `shape`: (n_beams, 2)
      `dtype`: `float32`

    n_samples: int
      The number of samples to take along the sampling line.

    length: float
      The length of the sampling line.

    Returns
    -------
    sampling_points_batch: Tensor
      The sampling points along the sampling line.
      `shape`: (n_beams, n_samples, 2)
      `dtype`: `float32`

    sampling_distance: Tensor
      The distances along the sampling line.
      `shape`: (n_samples,)
      `dtype`: `float32`
    """
    n_beams = beam_center_batch.shape[0]
    beam_axis_rad = atan2(
        beam_center_batch[:, 1] - detector_unit_center[1],
        beam_center_batch[:, 0] - detector_unit_center[0],
    )
    beam_axis_rad = beam_axis_rad + 2 * pi * (beam_axis_rad < 0)
    sampling_line_rad = beam_axis_rad + pi * 0.5
    kx = cos(sampling_line_rad)
    ky = sin(sampling_line_rad)
    sampling_distance = linspace(-length / 2, length / 2, n_samples)
    sampling_points_batch = stack((kx, ky), dim=1).unsqueeze(1).expand(
        -1, n_samples, -1
    ) * sampling_distance.view(1, n_samples, 1).expand(
        n_beams, -1, 2
    ) + beam_center_batch.unsqueeze(
        1
    ).expand(
        -1, n_samples, -1
    )
    return sampling_points_batch, sampling_distance


def beam_samples_on_points_batch(
    beam_data_2d_batch: Tensor,
    beam_sampling_points_batch: Tensor,
    fov_dict: dict,
):
    """
    Sample the beams data on the sampling points in batch.

    Parameters
    ----------
    beam_data_2d_batch: Tensor
      The PPDFs data in 2D.
      `shape`: (n_beams, n_pixels_x, n_pixels_y)
      `dtype`: `float32`

    beam_sampling_points_batch: Tensor
        The sampling points for the beams.
        `shape`: (n_beams, n_samples, 2)
        `dtype`: `float32`

    fov_dict: Dict, (optional)
        The field of view dictionary.
        `shape`: (n_pixels_x, n_pixels_y)
        `dtype`: `float32`

    n_samples: int
        The number of samples to take along the sampling line.

    Returns
    -------
    sampled_beams_batch: Tensor
      The sampled beams data.
      `shape`: (n_beams, n_samples)
      `dtype`: `float32`
    """
    # Prepare the input and grid for sampling
    # 2D PPDF field

    beam_field = beam_data_2d_batch.view(
        beam_data_2d_batch.shape[0],
        1,
        beam_data_2d_batch.shape[1],
        beam_data_2d_batch.shape[2],
    ).swapaxes(
        -1, -2
    )  # (n_beams, 1, n_pixels_x, n_pixels_y)

    # The grid contains the coordinates of the points where the input scalar
    # field is sampled. The grid is in the range [-1, 1] for both x and y.

    grid = (
        beam_sampling_points_batch.unsqueeze(1) / fov_dict["size in mm"] * 2
    )  # (n_beams, 1, n_samples, 2)

    sampled_beams_batch = F.grid_sample(
        beam_field, grid, mode="bilinear", align_corners=True
    ).squeeze()
    return sampled_beams_batch


def angular_edges_on_arc(
    sampled_data: Tensor,
    sampling_rads: Tensor,
    threshold: float = 0.01,
) -> Tensor:

    relative_sampled_ppdf = sampled_data / sampled_data.max()
    threshold = 0.01
    thresholded_relative_sampled_ppdf = zeros_like(
        relative_sampled_ppdf
    ).masked_fill_(relative_sampled_ppdf > threshold, 1)

    forward_diff_abs = diff(
        thresholded_relative_sampled_ppdf, prepend=tensor([0.0])
    ).abs()
    radian_edges = sampling_rads[forward_diff_abs > 0.5]
    return radian_edges


def beams_boundaries_radians(
    arc_sampled_ppdf: Tensor, arc_rads: Tensor, threshold: float = 0.01
) -> Tensor:
    arc_rads_step = arc_rads[1] - arc_rads[0]
    n_samples = arc_rads.shape[0]

    appened_rads = cat(
        (
            arc_rads,
            arc_rads[-1:] + arc_rads_step,
        ),
        dim=0,
    )
    relative_sampled_ppdf = arc_sampled_ppdf / arc_sampled_ppdf.max()
    threshold = 0.01
    thresholded_relative_sampled_ppdf = zeros_like(
        relative_sampled_ppdf
    ).masked_fill_(relative_sampled_ppdf > threshold, 1)

    forward_diff_abs = diff(
        thresholded_relative_sampled_ppdf,
        prepend=tensor([0.0]),
        append=tensor([0.0]),
    ).abs()

    radian_edges_indices = argwhere(forward_diff_abs > 0.5).squeeze()

    # Get the mean value of the relative sampled ppdf in each interval
    # between the radian edges
    n_intervals = radian_edges_indices.shape[0] - 1
    indices_expanded = arange(n_samples).view(1, -1).expand(n_intervals, -1)
    interval_boundaries = stack(
        (
            radian_edges_indices[:-1],
            radian_edges_indices[1:],
        ),
        dim=1,
    )
    interval_boundaries_expanded = interval_boundaries.unsqueeze(1).expand(
        -1, n_samples, -1
    )

    interval_masks = logical_and(
        indices_expanded >= interval_boundaries_expanded[:, :, 0],
        indices_expanded < interval_boundaries_expanded[:, :, 1],
    )

    interval_means = relative_sampled_ppdf.unsqueeze(0).expand(
        n_intervals, -1
    ).clone().masked_fill_(~interval_masks, 0).sum(dim=1) / interval_masks.sum(
        dim=1
    )

    beams_boundaries_indices = interval_boundaries[interval_means > threshold]
    beams_boundaries_angles = appened_rads[beams_boundaries_indices]
    return beams_boundaries_angles


def get_beams_masks(
    pixels_rads: Tensor,
    beam_bounds: Tensor,
) -> Tensor:
    n_beams = beam_bounds.shape[0]
    pixels_rads_expanded = pixels_rads.unsqueeze(0).expand(n_beams, -1)
    # Get beams masks
    return logical_and(
        pixels_rads_expanded >= beam_bounds[:, 0].view(-1, 1),
        pixels_rads_expanded < beam_bounds[:, 1].view(-1, 1),
    )


def get_beams_combined_mask(
    beams_masks: Tensor,
) -> Tensor:
    """
    Get the combined mask for the beams.
    The combined mask is uint16 tensor with the same shape as the beams masks.
    The value of the combined mask is the beam id.
    """
    n_beams = beams_masks.shape[0]
    beams_masks_valued = (
        arange(1, n_beams + 1).view(-1, 1).expand(-1, beams_masks.shape[1])
    )
    return (
        beams_masks_valued.clone()
        .masked_fill_(~beams_masks, 0)
        .to(dtype=uint16_tensor)
    ).sum(dim=0).unsqueeze(0)


def beams_line_properties(
    beams_basic_properties: Dict[str, Tensor],
    detector_unit_center: Tensor,
    ppdf_2d: Tensor,
    fov_dict: Dict,
    line_n_samples: int = 4096,
):
    # Get the beam sampling lines
    (beam_sp_ptx_batch, beam_sp_distance) = beam_sampling_line_batch(
        detector_unit_center,
        beams_basic_properties["weighted centers"],
        n_samples=line_n_samples,
        length=64.0,
    )

    n_beams = int(beams_basic_properties["number of beams"].item())
    beams_masks = beams_basic_properties["masks"]

    beams_data_2d_batch = (
        ppdf_2d.view(-1)
        .unsqueeze(0)
        .expand(n_beams, -1)
        .clone()
        .masked_fill_(~beams_masks, 0)
    ).view(
        n_beams,
        int(fov_dict["n pixels"][0]),
        int(fov_dict["n pixels"][1]),
    )

    sampled_beams_batch = beam_samples_on_points_batch(
        beams_data_2d_batch, beam_sp_ptx_batch, fov_dict
    )
    fwhm_batch, x_bounds_batch = full_width_half_maximum_1d_batch(
        beam_sp_distance.view(1, -1).expand(-1, line_n_samples),
        sampled_beams_batch,
    )
    return {
        "masks": beams_masks,
        "fwhm": fwhm_batch,
        "fwhm distance bounds": x_bounds_batch,
        "number of beams": tensor([n_beams]),
        "sampling points": beam_sp_ptx_batch,
        "sampling distance": beam_sp_distance,
        "sampled data": sampled_beams_batch,
    }


def get_beams_basic_properties(
    beams_masks: Tensor,
    ppdf_2d: Tensor,
    pixels_xys: Tensor,
):
    n_beams = beams_masks.shape[0]
    relative_ppdf = ppdf_2d / ppdf_2d.max()
    n_fov_points = pixels_xys.shape[0]
    # Calculate the relative sensitivity of each beam
    relative_ppdf_expanded = (
        relative_ppdf.view(-1).unsqueeze(0).expand(n_beams, -1)
    )
    # Calculate the relative sensitivity of each beam
    relative_sensitivity = relative_ppdf_expanded.clone().masked_fill_(
        ~beams_masks, 0
    ).sum(dim=1) / beams_masks.sum(dim=1)

    beams_sizes = beams_masks.sum(dim=1)

    absolute_sensitivity = relative_sensitivity * ppdf_2d.max()

    return beams_sizes, relative_sensitivity, absolute_sensitivity


def get_beam_width(
    beams_centers: Tensor,
    detector_unit_center: Tensor,
    beams_masks: Tensor,
    ppdf_2d: Tensor,
    fov_dict: Dict,
    line_n_samples: int = 4096,
):
    """
    Compute the Full Width at Half Maximum (FWHM) of beams and their sampled data
    along specified sampling lines.

    Parameters
    ----------
    beams_centers : Tensor
        A tensor of shape `(n_beams, 2)` representing the coordinates of the beam centers.
    detector_unit_center : Tensor
        A tensor of shape `(2,)` representing the center of the detector unit.
    beams_masks : Tensor
        A boolean tensor of shape `(n_beams, n_pixels_x * n_pixels_y)` indicating
        the valid regions for each beam.
    ppdf_2d : Tensor
        A 2D tensor of shape `(n_pixels_x, n_pixels_y)` representing the pixelated
        probability density function (PPDF).
    fov_dict : Dict
        A dictionary containing field-of-view (FOV) metadata, including the number
        of pixels in each dimension (`"n pixels"`).
    line_n_samples : int, optional
        The number of samples to take along each sampling line. Default is 4096.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        - `beams_fwhm` : Tensor
            A tensor of shape `(n_beams,)` containing the Full Width at Half Maximum
            (FWHM) for each beam.
        - `x_bounds_batch` : Tensor
            A tensor of shape `(n_beams, 2)` containing the x-coordinate bounds for
            each beam's FWHM.
        - `sampled_beams_data` : Tensor
            A tensor of shape `(n_beams, line_n_samples)` containing the sampled beam
            data along the sampling lines.
        - `beam_sp_distance` : Tensor
            A tensor of shape `(line_n_samples,)` representing the distances along
            the sampling lines.
    """

    # Get the beam sampling lines
    (beam_sp_ptx_batch, beam_sp_distance) = beam_sampling_line_batch(
        detector_unit_center,
        beams_centers,
        n_samples=line_n_samples,
        length=64.0,
    )

    n_beams = beams_centers.shape[0]

    beams_data_2d_batch = (
        ppdf_2d.view(-1)
        .unsqueeze(0)
        .expand(n_beams, -1)
        .clone()
        .masked_fill_(~beams_masks, 0)
    ).view(
        n_beams,
        int(fov_dict["n pixels"][0]),
        int(fov_dict["n pixels"][1]),
    )

    sampled_beams_data = beam_samples_on_points_batch(
        beams_data_2d_batch, beam_sp_ptx_batch, fov_dict
    )
    beams_fwhm, x_bounds_batch = full_width_half_maximum_1d_batch(
        beam_sp_distance.view(1, -1).expand(-1, line_n_samples),
        sampled_beams_data,
    )
    return beams_fwhm, x_bounds_batch, sampled_beams_data, beam_sp_distance


def beams_properties_2d(
    ppdf_2d: Tensor,
    pixels_coordinates: Tensor,
    pixels_rads: Tensor,
    rads_edges: Tensor,
    detector_unit_center: Tensor,
    fov_dict: dict,
    threshold: float = 0.01,
    line_n_samples: int = 4096,
) -> Dict[str, Tensor]:

    # Basic properties of the beams
    relative_ppdf = ppdf_2d / ppdf_2d.max()
    n_fov_points = pixels_coordinates.shape[0]
    n_rad_intervals = rads_edges.shape[0] - 1
    pixels_rads_expanded = pixels_rads.unsqueeze(1).expand(-1, n_rad_intervals)

    rad_interval_edges = stack(
        [
            rads_edges[:-1],
            rads_edges[1:],
        ],
        dim=1,
    )  # (n_rad_intervals, 2)
    rads_edges_expanded = rads_edges.unsqueeze(0).expand(n_fov_points, -1)

    # Get the radian intervals masks
    rad_interval_masks = (
        pixels_rads_expanded >= rads_edges_expanded[:, :-1]
    ) & (
        pixels_rads_expanded < rads_edges_expanded[:, 1:]
    )  # (n_fov_points, n_rad_intervals)

    relative_ppdf_expanded = (
        relative_ppdf.view(-1).unsqueeze(1).expand(-1, n_rad_intervals)
    )
    sum_template = relative_ppdf_expanded.clone().masked_fill_(
        ~rad_interval_masks, 0
    )
    intervals_relative_sensitivities = sum_template.sum(
        dim=0
    ) / rad_interval_masks.sum(dim=0)

    # Discard the range with mean < 0.01
    beams_masks = rad_interval_masks[
        :, intervals_relative_sensitivities > threshold
    ]
    beams_relative_sensitivities = intervals_relative_sensitivities[
        intervals_relative_sensitivities > threshold
    ]
    beam_n_pixels = beams_masks.sum(dim=0)

    # Get the weighted center of the beams
    beams_weighted_centers = beams_weighted_center(
        beams_masks,
        pixels_coordinates,
        relative_ppdf,
        beams_relative_sensitivities,
    )
    beams_sensitivities = beams_relative_sensitivities * ppdf_2d.max()

    # Calculate the beam regions radian edges
    beam_rads_edges = rad_interval_edges[
        intervals_relative_sensitivities > threshold
    ]

    # Get the beam sampling lines
    (beam_sp_ptx_batch, beam_sp_distance) = beam_sampling_line_batch(
        detector_unit_center,
        beams_weighted_centers,
        n_samples=line_n_samples,
        length=64.0,
    )

    # Swap the axes of the beam masks
    beams_masks = beams_masks.swapaxes(0, 1)
    # Number of beams
    n_beams = beams_masks.shape[0]

    beams_data_2d_batch = (
        ppdf_2d.view(-1)
        .unsqueeze(0)
        .expand(n_beams, -1)
        .clone()
        .masked_fill_(~beams_masks, 0)
    ).view(
        n_beams,
        int(fov_dict["n pixels"][0]),
        int(fov_dict["n pixels"][1]),
    )

    sampled_beams_batch = beam_samples_on_points_batch(
        beams_data_2d_batch, beam_sp_ptx_batch, fov_dict
    )
    fwhm_batch, x_bounds_batch = full_width_half_maximum_1d_batch(
        beam_sp_distance.view(1, -1).expand(-1, line_n_samples),
        sampled_beams_batch,
    )
    # Calculate the beam axis angle in radian
    beam_axis_rad = atan2(
        beams_weighted_centers[:, 1] - detector_unit_center[1],
        beams_weighted_centers[:, 0] - detector_unit_center[0],
    )
    beam_axis_rad = beam_axis_rad + 2 * pi * (beam_axis_rad < 0)
    return {
        "masks": beams_masks,
        "weighted centers": beams_weighted_centers,
        "rads edges": beam_rads_edges,
        "axial angle": beam_axis_rad,
        "sensitivities": beams_sensitivities,
        "relative sensitivities": beams_relative_sensitivities,
        "n pixels": beam_n_pixels,
        "fwhm": fwhm_batch,
        "fwhm distance bounds": x_bounds_batch,
        "number of beams": tensor([n_beams]),
        "sampling points": beam_sp_ptx_batch,
        "sampling distance": beam_sp_distance,
        "sampled data": sampled_beams_batch,
    }


def beam_radial_sampling_line_batch(
    detector_unit_center: Tensor, # (2,)
    beam_center_batch: Tensor,    # (n_beams, 2)
    n_samples: int = 1024,
    max_length_mm: float = 100.0, # Max sampling length from detector center
    start_offset_mm: float = 0.1  # Small offset from detector to avoid singularity
) -> tuple[Tensor, Tensor]:
    """
    Get the radial sampling lines for the beams in batch.
    The sampling line is along the beam axis (detector center to beam center).

    Parameters
    ----------
    detector_unit_center: Tensor
      The center of the detector unit.
    beam_center_batch: Tensor
      The weighted centers of the beams.
    n_samples: int
      The number of samples to take along the sampling line.
    max_length_mm: float
      The maximum distance from the detector center to sample along the radial line.
    start_offset_mm: float
      Small distance from detector center to start sampling.

    Returns
    -------
    sampling_points_batch: Tensor (n_beams, n_samples, 2)
      The sampling points along the radial line.
    sampling_distances_from_detector: Tensor (n_samples,)
      The distances from the detector center along the sampling line.
    """
    n_beams = beam_center_batch.shape[0]
    device = detector_unit_center.device

    # Vector from detector to each beam center
    direction_vectors = beam_center_batch - detector_unit_center.unsqueeze(0) # (n_beams, 2)
    norm_direction_vectors = norm(direction_vectors, dim=1, keepdim=True) # (n_beams, 1)
    
    # Avoid division by zero if beam center is at detector center
    norm_direction_vectors = torch.where(norm_direction_vectors < 1e-6, 
                                         torch.ones_like(norm_direction_vectors), 
                                         norm_direction_vectors)
    
    unit_direction_vectors = direction_vectors / norm_direction_vectors # (n_beams, 2)

    # Define distances along the radial line from the detector center
    sampling_distances_from_detector = linspace(start_offset_mm, max_length_mm, n_samples, device=device) # (n_samples,)

    # Calculate sampling points: detector_center + distance * unit_vector
    # Need to expand dimensions for broadcasting:
    # detector_unit_center: (1, 1, 2)
    # sampling_distances_from_detector: (1, n_samples, 1)
    # unit_direction_vectors: (n_beams, 1, 2)
    
    sampling_points_batch = (
        detector_unit_center.unsqueeze(0).unsqueeze(0) +
        sampling_distances_from_detector.unsqueeze(0).unsqueeze(-1) * unit_direction_vectors.unsqueeze(1)
    ) # Shape: (n_beams, n_samples, 2)

    return sampling_points_batch, sampling_distances_from_detector
    
    
    
def get_beam_radial_width(
    beams_centers: Tensor,      # (n_beams, 2)
    detector_unit_center: Tensor, # (2,)
    beams_masks: Tensor,        # (n_beams, n_pixels_flat) bool
    ppdf_2d: Tensor,            # (H, W) PPDF for the current detector
    fov_dict: Dict,
    radial_line_n_samples: int = 1024,
    radial_line_max_length_mm: float = 150.0, # Should cover the FOV extent
    radial_line_start_offset_mm: float = 0.1
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the Radial Full Width at Half Maximum (FWHM) of beams.
    Samples along lines from detector center through beam centers.
    """
    # 1. Get the radial beam sampling lines and distances from detector
    (beam_radial_sp_ptx_batch, beam_radial_sp_distance) = beam_radial_sampling_line_batch(
        detector_unit_center,
        beams_centers,
        n_samples=radial_line_n_samples,
        max_length_mm=radial_line_max_length_mm,
        start_offset_mm=radial_line_start_offset_mm
    ) # beam_radial_sp_distance is distances from detector

    n_beams = beams_centers.shape[0]

    # 2. Prepare PPDF data for sampling (masked by beam masks)
    # Ensure ppdf_2d is (H, W)
    # Ensure beams_masks is (n_beams, H*W)
    
    # The PPDF data used for sampling should ideally be only from within the beam's mask.
    # However, F.grid_sample samples from the continuous field.
    # The `beams_masks` are more to define *which pixels contribute to the beam's identity*,
    # not necessarily to zero out PPDF values outside the mask *before* FWHM profile sampling.
    # For radial FWHM, we sample the original `ppdf_2d` along the radial line.
    # The concept of the beam's own mask is less critical here than for tangential,
    # as the radial profile is defined by the PPDF intensity along that specific line.
    
    # We need ppdf_2d in the shape (n_beams, H, W) for beam_samples_on_points_batch
    # if we want to process it similarly. But `beam_samples_on_points_batch`
    # expects `beam_data_2d_batch` to be (n_beams, n_pixels_x, n_pixels_y).
    # For radial FWHM, we are sampling the *same* `ppdf_2d` for all beams, just along different lines.
    
    # Let's adapt sampling logic slightly if needed, or reuse beam_samples_on_points_batch carefully.
    # `beam_samples_on_points_batch` expects `beam_data_2d_batch` of shape (n_beams, H, W)
    # So, we can expand the single ppdf_2d:
    ppdf_2d_expanded_for_sampling = ppdf_2d.unsqueeze(0).expand(
        n_beams, ppdf_2d.shape[0], ppdf_2d.shape[1]
    )

    # 3. Sample PPDF data along these radial lines
    sampled_beams_radial_data = beam_samples_on_points_batch(
        ppdf_2d_expanded_for_sampling, # (n_beams, H, W)
        beam_radial_sp_ptx_batch,      # (n_beams, n_samples, 2)
        fov_dict
    ) # Output: (n_beams, n_samples)

    # 4. Calculate FWHM from these 1D radial profiles
    # The 'x-axis' for FWHM is `beam_radial_sp_distance` (distances from detector)
    beams_fwhm_radial, x_bounds_radial_batch = full_width_half_maximum_1d_batch(
        beam_radial_sp_distance.view(1, -1).expand(n_beams, radial_line_n_samples), # X-values (distances)
        sampled_beams_radial_data  # Y-values (sampled PPDF)
    )
    
    return beams_fwhm_radial, x_bounds_radial_batch, sampled_beams_radial_data, beam_radial_sp_distance