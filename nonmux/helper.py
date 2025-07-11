from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes
import torch

from typing import Tuple, Dict, Any
from torch import Tensor
import hashlib


def plot_polygons_from_vertices_2d_mpl(
    vertices: torch.Tensor, ax: Axes, **kwargs
):
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p


def plate_random_aperture_dev(input: Tensor) -> Dict[str, Tensor]:
    """
    Generate a random aperture for a plate.
    The apertures is defined by a polygons each with 4 vertices.

    Parameters
    ----------
    input : Tensor
        A tensor of shape (5,) containing the following values:

        - n_segments: Number of segments of the plate
        - inner_radius: Inner radius of the plate
        - thickness: Thickness of the plate
        - aperture_unit: Unit size of the aperture
        - ratio: Ratio of the aperture to the inner radius

    Returns
    -------
    Dict[str, Tensor]
        A dictionary containing the following:

        - "all cell polygons": `rank-3` tensor, `shape`: (n_cells, 4, 2), containing the vertices of all cells
        - "merged polygons": `rank-3` `tensor, `shape`: (n_merged_polygons, 4, 2), containing the vertices of the merged polygons
    """

    n_segments = input[0]
    inner_radius = input[1]
    thickness = input[2]
    aperture_unit = input[3]
    ratio = input[4]

    rad_tan = torch.tan(torch.pi / n_segments)
    y_half_front = rad_tan * inner_radius

    m = int(2 * ratio * y_half_front / aperture_unit)
    biased_ratios = (
        torch.tensor([m, m + 1]) * aperture_unit / y_half_front * 0.5
    )

    ratio_diff_abs = torch.abs(biased_ratios - ratio)
    m = m + 1 if ratio_diff_abs[1] < ratio_diff_abs[0] else m

    actual_ratio = m * aperture_unit / y_half_front * 0.5
    min_offset = thickness * rad_tan
    y_half_back = y_half_front + min_offset
    n_middle_polygons = int(2 * (y_half_front - min_offset) / aperture_unit)

    m = m - 1 if m >= n_middle_polygons else m
    offset = y_half_front - n_middle_polygons * aperture_unit * 0.5
    y_start = -y_half_front + offset
    y_end = y_half_front - offset
    x_start = inner_radius
    x_end = inner_radius + thickness

    low_corner_polygon = torch.tensor(
        [
            [x_start, -y_half_front],
            [x_end, -y_half_back],
            [x_end, y_start],
            [x_start, y_start],
        ]
    ).unsqueeze(0)

    up_corner_polygon = torch.tensor(
        [
            [x_start, y_end],
            [x_end, y_end],
            [x_end, y_half_back],
            [x_start, y_half_front],
        ]
    ).unsqueeze(0)

    middle_polygons_vertices = torch.empty((n_middle_polygons, 4, 2))
    middle_polygons_vertices[:, [0, 3], 0] = (
        (torch.ones(n_middle_polygons) * x_start).unsqueeze(1).expand(-1, 2)
    )
    middle_polygons_vertices[:, [1, 2], 0] = (
        (torch.ones(n_middle_polygons) * x_end).unsqueeze(1).expand(-1, 2)
    )
    middle_polygons_vertices[:, [0, 1], 1] = (
        (torch.arange(n_middle_polygons) * aperture_unit + y_start)
        .unsqueeze(1)
        .expand(-1, 2)
    )
    middle_polygons_vertices[:, [2, 3], 1] = (
        ((torch.arange(n_middle_polygons) + 1) * aperture_unit + y_start)
        .unsqueeze(1)
        .expand(-1, 2)
    )

    mask = torch.ones((n_middle_polygons), dtype=torch.bool)
    mask[torch.sort(torch.randperm(n_middle_polygons)[:m])[0]] = False
    mask = torch.cat(
        (
            torch.ones((1), dtype=torch.bool),
            mask,
            torch.ones((1), dtype=torch.bool),
        )
    )
    all_cells = torch.cat(
        (low_corner_polygon, middle_polygons_vertices, up_corner_polygon),
        dim=0,
    )
    indices = torch.argwhere(mask).squeeze()
    data = torch.stack(
        (
            indices,
            torch.cat(
                (torch.tensor([100]), torch.diff(indices, dim=0))
            ).squeeze(),
            torch.cat(
                (torch.diff(indices, dim=0), torch.tensor([100]))
            ).squeeze(),
        ),
        dim=1,
    ).squeeze()
    partitions = torch.stack(
        (data[data[:, 1] > 1, 0], data[data[:, 2] > 1, 0]), dim=1
    )
    merged_polygons = torch.cat(
        (
            all_cells[partitions[:, 0], :2, :],
            all_cells[partitions[:, 1], 2:, :],
        ),
        dim=1,
    )

    print(
        f"{'Number of aperture cell':28s}:{m}\n{'Actual aperture ratio':28s}:{actual_ratio}\n"
    )
    return {
        "all cell polygons": all_cells,
        "merged polygons": merged_polygons,
        "partitions": partitions,
        "mask": mask,
    }


def plate_random_apertures(input: Tensor) -> tuple[Tensor, Tensor]:
    """
    Generate a plate with random apertures.
    The apertures is defined by a polygons each with 4 vertices.

    Parameters
    ----------
    input : torch.Tensor
        A tensor of shape (5,) containing the following values:

        - n_segments: Number of segments of the plate
        - inner_radius: Inner radius of the plate
        - thickness: Thickness of the plate
        - aperture_unit: Unit size of the aperture
        - ratio: Ratio of the aperture to the inner radius

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_polygons, 4, 2) containing the vertices of the plate polygons.
        Each cell is defined by 4 vertices.
    """

    n_segments = input[0]
    inner_radius = input[1]
    thickness = input[2]
    aperture_unit = input[3]
    ratio = input[4]

    rad_tan = torch.tan(torch.pi / n_segments)
    y_half_front = rad_tan * inner_radius

    m = int(2 * ratio * y_half_front / aperture_unit)
    biased_ratios = (
        torch.tensor([m, m + 1]) * aperture_unit / y_half_front * 0.5
    )

    ratio_diff_abs = torch.abs(biased_ratios - ratio)
    m = m + 1 if ratio_diff_abs[1] < ratio_diff_abs[0] else m

    min_offset = thickness * rad_tan
    y_half_back = y_half_front + min_offset
    n_middle_polygons = int(2 * (y_half_front - min_offset) / aperture_unit)

    m = m - 1 if m >= n_middle_polygons else m
    offset = y_half_front - n_middle_polygons * aperture_unit * 0.5
    y_start = -y_half_front + offset
    y_end = y_half_front - offset
    x_start = inner_radius
    x_end = inner_radius + thickness

    # Calculate the actual ratio
    actual_ratio = m * aperture_unit / y_half_front * 0.5

    low_corner_polygon = torch.tensor(
        [
            [x_start, -y_half_front],
            [x_end, -y_half_back],
            [x_end, y_start],
            [x_start, y_start],
        ]
    ).unsqueeze(0)

    up_corner_polygon = torch.tensor(
        [
            [x_start, y_end],
            [x_end, y_end],
            [x_end, y_half_back],
            [x_start, y_half_front],
        ]
    ).unsqueeze(0)

    middle_polygons_vertices = torch.empty((n_middle_polygons, 4, 2))
    middle_polygons_vertices[:, [0, 3], 0] = (
        (torch.ones(n_middle_polygons) * x_start).unsqueeze(1).expand(-1, 2)
    )
    middle_polygons_vertices[:, [1, 2], 0] = (
        (torch.ones(n_middle_polygons) * x_end).unsqueeze(1).expand(-1, 2)
    )
    middle_polygons_vertices[:, [0, 1], 1] = (
        (torch.arange(n_middle_polygons) * aperture_unit + y_start)
        .unsqueeze(1)
        .expand(-1, 2)
    )
    middle_polygons_vertices[:, [2, 3], 1] = (
        ((torch.arange(n_middle_polygons) + 1) * aperture_unit + y_start)
        .unsqueeze(1)
        .expand(-1, 2)
    )

    mask = torch.ones((n_middle_polygons), dtype=torch.bool)
    mask[torch.sort(torch.randperm(n_middle_polygons)[:m])[0]] = False
    mask = torch.cat(
        (
            torch.ones((1), dtype=torch.bool),
            mask,
            torch.ones((1), dtype=torch.bool),
        )
    )
    all_cells = torch.cat(
        (low_corner_polygon, middle_polygons_vertices, up_corner_polygon),
        dim=0,
    )
    indices = torch.argwhere(mask).squeeze()
    data = torch.stack(
        (
            indices,
            torch.cat(
                (torch.tensor([100]), torch.diff(indices, dim=0))
            ).squeeze(),
            torch.cat(
                (torch.diff(indices, dim=0), torch.tensor([100]))
            ).squeeze(),
        ),
        dim=1,
    ).squeeze()
    partitions = torch.stack(
        (data[data[:, 1] > 1, 0], data[data[:, 2] > 1, 0]), dim=1
    )
    merged_polygons = torch.cat(
        (
            all_cells[partitions[:, 0], :2, :],
            all_cells[partitions[:, 1], 2:, :],
        ),
        dim=1,
    )
    return merged_polygons, actual_ratio


def plates_random_apertures(
    input: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    plates_random_apertures(input, n, step) -> Tensor
    Generate a full circle of plates with random apertures.

    Args
    ----------
    input : torch.Tensor
        A tensor of shape (5,) containing the following values:

        - n_segments: Number of segments of the plate
        - inner_radius: Inner radius of the plate
        - thickness: Thickness of the plate
        - aperture_unit: Unit size of the aperture
        - ratio: Ratio of the aperture to the inner radius

    Returns
    -------
    plates_vertices: torch.Tensor, shape (n * m, 4, 2)
        A tensor containing the vertices of the plate segments
    """

    n = int(input[0])  # number of plate segments
    # rotation angles
    rotations = torch.arange(0, n) * 2 * torch.pi / n
    # rotation matrix
    rotation_matrices = torch.stack(
        (
            torch.cos(rotations),
            -torch.sin(rotations),
            torch.sin(rotations),
            torch.cos(rotations),
        ),
        dim=1,
    ).reshape(-1, 2, 2)

    # Preallocate the plates_vertices tensor
    # shape: (0, 4, 2)
    plates_vertices = torch.empty((0, 4, 2), dtype=torch.float32)
    ratios = torch.empty((0), dtype=torch.float32)
    for idx in range(n):
        # Generate random apertures for each segment independently
        plate_vertices, ratio = plate_random_apertures(input)
        ratios = torch.cat((ratios, ratio.unsqueeze(0)), dim=0)
        # Rotate the vertices for each segment
        rotated_vertices = torch.bmm(
            rotation_matrices[idx]
            .unsqueeze(0)
            .expand(plate_vertices.shape[0] * 4, -1, -1),
            plate_vertices.view(-1, 2).unsqueeze(2),
        ).reshape(-1, 4, 2)

        # Concatenate the rotated vertices to the plates_vertices tensor
        plates_vertices = torch.cat(
            (
                plates_vertices,
                rotated_vertices,
            ),
            dim=0,
        )

    return plates_vertices, ratios.mean()


def rotate_and_repeat_4gon(
    input: torch.Tensor, n: int, step: float | Tensor
) -> torch.Tensor:
    """
    rotate_and_repeat_4gon(input, n, step) -> Tensor


    Rotate and repeat the vertices of the quadrilateral (polygons
    with 4 vertices).

    Args
    ----------
    input: torch.Tensor, shape (m, 4, 2)
      A tensor containing the vertices of the quadrilaterals.

    n: int
      The number of segments to repeat the vertices.
        - Example: 6

    step:
      The step size for the rotation. Unit is radian.
        - Example: `2 * torch.pi / n`


    Returns
    -------
    rotated_vertices: torch.Tensor, shape (n * m, 4, 2)

        A tensor containing the rotated and repeated vertices.
    """
    # Rotate the vertices for each segment

    # rotation angles
    rotations = (torch.arange(0, n) * step).repeat_interleave(
        4 * input.shape[0]
    )

    # rotation matrix
    rotation_matrix = torch.stack(
        (
            torch.cos(rotations),
            -torch.sin(rotations),
            torch.sin(rotations),
            torch.cos(rotations),
        ),
        dim=1,
    ).reshape(-1, 2, 2)

    # rotate and repeat
    output = torch.bmm(
        rotation_matrix,
        (input.tile(n, 1, 1)).view(-1, 2).unsqueeze(2),
    ).view(-1, 4, 2)
    return output


def cell_grid_2d(input: Tensor) -> Tensor:
    """
    Generate a grid of cells in 2D space. Each cell is a rectangle defined by
    4 vertices.

    Parameters
    ----------
    `input` : `Tensor`

      `rank-1` `Tensor` of shape `(5, )`
      - `input[0]` : `float` : radial distance to the FOV center in mm
      - `input[1]` : `float` : x size of a cell in mm
      - `input[2]` : `float` : y size of a cell in mm
      - `input[3]` : `float` : number of cells in x direction
      - `input[4]` : `float` : number of cells in y direction

    Returns
    -------

    `Tensor`:
      `rank-4` tensor of shape `(M, N, 4, 2)` containing the vertices of the grid cells.
    """
    x_borders = torch.arange(int(input[3]) + 1) * input[1] + input[0]
    y_borders = (
        torch.arange(int(input[4]) + 1) * input[2] - input[2] * input[4] * 0.5
    )

    x_grid, y_grid = torch.meshgrid(x_borders, y_borders, indexing="ij")
    return torch.stack(
        [
            torch.stack([x_grid[:-1, :-1], y_grid[:-1, :-1]], dim=-1),
            torch.stack([x_grid[1:, :-1], y_grid[:-1, :-1]], dim=-1),
            torch.stack([x_grid[1:, 1:], y_grid[1:, 1:]], dim=-1),
            torch.stack([x_grid[:-1, 1:], y_grid[1:, 1:]], dim=-1),
        ],
        dim=-2,
    )


def grid_cells_random_mask_batch(input: Tensor) -> Tensor:
    """
    Generate a randomized mask for the grid cells with given occupancy.

    Parameters
    ----------
    `input` : `Tensor`

      `rank-1` `Tensor` of shape `(4, )`
      - `input[0]` : `float` : ratio, number of occupied cells over total
      number of cells of a panel
      - `input[1]` : `float` : number of cells in x direction
      - `input[2]` : `float` : number of cells in y direction
      - `input[3]` : `float` : batch size

    Returns
    -------

    `Tensor`:
      `rank-2` tensor of shape `(input[3], input[1] * input[2])` containing the mask for the grid cells.
    """
    n_total = int(input[1] * input[2])
    shape = (int(input[3]), int(input[1:3].prod()))
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[
        torch.arange(int(input[3])).unsqueeze(-1),
        torch.multinomial(
            torch.ones(shape, dtype=torch.float),
            int(torch.prod(input[:-1])),
            replacement=False,
        ),
    ] = True
    return mask


def grid_cells_random_full_mask_batch(input: Tensor) -> Tensor:
    """
    Generate a randomized mask for the grid cells with given occupancy. \\
    The last column of the mask is fully occupied.

    Parameters
    ----------
    `input` : `Tensor`

      `rank-1` `Tensor` of shape `(4, )`
      - `input[0]` : `float` : ratio, number of occupied cells over total
      number of cells of a panel
      - `input[1]` : `float` : number of cells in x direction
      - `input[2]` : `float` : number of cells in y direction
      - `input[3]` : `float` : batch size

    Returns
    -------

    `Tensor`:
      `rank-2` tensor of shape `(input[3], input[1] * input[2])` containing the mask for the grid cells.
    """

    cells_mask = grid_cells_random_mask_batch(input).view(
        int(input[3]), int(input[1]), int(input[2])
    )
    full_column_mask = torch.ones(
        (int(input[3]), 1, int(input[2])), dtype=torch.bool
    )
    concatenated_mask = torch.cat(
        [
            cells_mask,
            full_column_mask,
        ],
        dim=1,
    )
    return concatenated_mask


def single_crystal(input: Tensor) -> Tensor:
    """
    Generate a single crystal polygon based on the size.

    Parameters
    ----------
    input : Tensor
        `rank-1` tensor of shape (2, ) representing the
        size definition of the crystal.

    Returns
    -------
    output : Tensor
        `rank-3` tensor of shape (1, 4, 2) representing the vertices of the crystal.
    """

    return torch.tensor(
        [
            [-input[0] / 2, -input[1] / 2],
            [input[0] / 2, -input[1] / 2],
            [input[0] / 2, input[1] / 2],
            [-input[0] / 2, input[1] / 2],
        ]
    ).unsqueeze(0)


def detector_units_panels(
    cell_grid: Tensor,
    inner_cell: Tensor,
    outer_cell: Tensor,
    **kwargs,
) -> Tensor:
    """
    Generates a randomized scanner layout based on the input parameters.

    Parameters
    ----------
    cell_grid : Tensor, shape (N_r, N_c, 4, 2)
        `rank-4` tensor containing the vertices of the cell grid of a panel.\\
        `N_r` is the number of rows of cells in a grid \\
        `N_c` is the number of columns of cells in a grid \\

    inner_cell : Tensor, shape (M, 4, 2)
        `rank-3` tensor containing the vertices of detector units in a single
        cell in the inner layers.\\
        `M` is the number of detector units

    outer_cell : Tensor, shape (L, 4, 2)
        `rank-3` tensor containing the vertices of the outer detector units 
        in a single cell in the outermost layer.\\
        `L` is the number of outer detector units. `Default` is `1`.

    Returns
    -------
    out: Tensor
        A tensor containing the vertices of the detector units.

    """
    n_detector_panels = kwargs.get("n_detector_panels", 6)
    inner_cell_columns = kwargs.get("inner_cell_columns", [0, 1, 2, 3, 4, 5, 6])
    outer_cell_columns = kwargs.get("outer_cell_columns", [7])

    inner_cells_array = cell_grid[inner_cell_columns, :, :, :].view(-1, 4, 2)
    outer_cells_array = cell_grid[outer_cell_columns, :, :, :].view(-1, 4, 2)
    inner_cell_centers = inner_cells_array.mean(dim=-2)
    outer_cell_centers = outer_cells_array.mean(dim=-2)
    inner_units = inner_cell.repeat(
        inner_cell_centers.shape[0], 1, 1
    ) + inner_cell_centers.repeat_interleave(
        inner_cell.shape[0], dim=0
    ).unsqueeze(
        1
    ).expand(
        -1, 4, -1
    )
    outer_units = outer_cell.repeat(
        outer_cell_centers.shape[0], 1, 1
    ) + outer_cell_centers.repeat_interleave(
        outer_cell.shape[0], dim=0
    ).unsqueeze(
        1
    ).expand(
        -1, 4, -1
    )
    panel_detector_units = torch.cat(
        (
            inner_units,
            outer_units,
        ),
        dim=0,
    ).view(-1, 4, 2)
    out = rotate_and_repeat_4gon(
        panel_detector_units.view(-1, 4, 2),
        n=n_detector_panels,
        step=torch.pi / n_detector_panels * 2,
    )
    return out


def cell_detector_units(crystal_size: Tensor, centers: Tensor) -> Tensor:
    """
    Generates a randomized scanner layout based on the input parameters.

    Parameters
    ----------
    crystal_size : Tensor, shape (6,)
        A tensor containing the parameters for the scanner layout.

        The parameters are as follows:
        - `crystal_size[0]`: detector unit size in x direction
        - `crystal_size[1]`: detector unit size in y direction

    centers : Tensor, shape (N, 2)
        A tensor containing the center coordinates of the detector units.
    Returns
    -------
    out: Tensor
        A tensor containing the vertices of the detector units.

    """

    out = single_crystal(crystal_size).repeat(
        centers.size(0), 1, 1
    ) + centers.unsqueeze(1).repeat_interleave(4, dim=1)
    return out


def scanner_layout_random_last_full(
    input: Tensor, **kwargs
) -> Tuple[Tensor, Tensor]:
    """
    Generate a randomized scanner layout based on the input parameters.

    Parameters
    ----------
    input : Tensor
        `rank-1` tensor of shape (5,) containing the following values:
        - `input[0]`: ratio of the aperture to plate tangential length
        - `input[1]`: ratio of the occupied cells over total number of cells of a panel
        - `input[2]`: inner radius of the plate
        - `input[3]`: detector array to plate distance

    kwargs : Keyword Arguments, optional

        Additional parameters for the function. The following parameters can be provided:

        | Property                     | Description                                          | Type           | Default Value     |
        |:-----------------------------|:-----------------------------------------------------|:---------------|:------------------|
        | `inner_unit_size`            | Size of the inner layer detector unit in mm          |`Tuple`         |`(2.4, 2.4)`       |
        | `outer_unit_size`            | Size of the outer layer detector unit in mm          |`Tuple`         |`(3.0, 3.0)`       |
        | `inner_unit_centers_cell`    | Center coordinates of the inner layer detector units |`Tensor (N, 2)` |`tensor([[0, 0]])` |
        | `outer_unit_centers_cell`    | Center coordinates of the outer layer detector units |`Tensor (L, 2)` |`tensor([[0, 0]])` |
        | `cell_size`                  | Size of the cells in mm                              |`Tuple`         |`(3.36, 3.36)`     |
        | `n_cells`                    | Number of cells in x and y direction                 |`Tuple`         |`(8, 32)`          |
        | `n_plate_segments`           | Number of plate segments                             |`int`           |`6`                |
        | `n_detector_panels`          | Number of detector panels                            |`int`           |`6`                |
        | `plate_thickness`            | Thickness of the plate in mm                         |`float`         |`2.0`              |
        | `aperture_unit_size`         | Size of the aperture unit in mm                      |`float`         |`2.0`              |

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing:
        - `rank-3` tensor of shape (M, 4, 2) containing the vertices of plate segments
        - `rank-3` tensor of shape (N, 4, 2) containing the vertices of the detector units
    """
    inner_unit_size = kwargs.get("inner_unit_size", (2.4, 2.4))
    inner_unit_centers_cell = kwargs.get(
        "inner_unit_centers_cell", torch.tensor([[0, 0]]).view(-1, 2)
    )
    outer_unit_size = kwargs.get("outer_unit_size", (3.0, 3.0))
    outer_unit_centers_cell = kwargs.get(
        "outer_unit_centers_cell", torch.tensor([[0, 0]]).view(-1, 2)
    )
    cell_size = kwargs.get("cell_size", (3.36, 3.36))
    n_cells = kwargs.get("n_cells", (8, 32))
    n_plate_segments = kwargs.get("n_plate_segments", 6)
    n_detector_panels = kwargs.get("n_detector_panels", 6)
    plate_thickness = kwargs.get("plate_thickness", 2.0)
    aperture_unit_size = kwargs.get("aperture_unit_size", 2.0)

    plate_segments, aperture_ratio = plates_random_apertures(
        torch.tensor(
            [
                n_plate_segments,
                input[2],
                plate_thickness,
                aperture_unit_size,
                input[0],
            ]
        )
    )
    inner_units_cell_array = cell_detector_units(
        torch.tensor(inner_unit_size),
        inner_unit_centers_cell,
    )
    outer_units_cell_array = cell_detector_units(
        torch.tensor(outer_unit_size),
        outer_unit_centers_cell,
    )

    panel_cell_grid = cell_grid_2d(
        torch.tensor(
            [
                float(input[2] + input[3] + plate_thickness),
                cell_size[0],
                cell_size[1],
                n_cells[0],
                n_cells[1],
            ]
        )
    )

    detector_units = detector_units_panels(
        panel_cell_grid,
        inner_units_cell_array,
        outer_units_cell_array,
        **kwargs,
    )
    scanner_cells_mask = grid_cells_random_full_mask_batch(
        torch.tensor([input[1], n_cells[0] - 1, n_cells[1], n_detector_panels])
    )

    return plate_segments, detector_units[scanner_cells_mask.view(-1)]


def rotate_polygon_batch(angles: Tensor, polygons: Tensor) -> Tensor:
    """
    Rotate a batch of polygons by given angles.

    Parameters
    -----------

        angles : Tensor, shape (`N_Batch`, 1)

            The angles in radians to rotate the polygons.

        polygons : Tensor

            The polygons to rotate, shape (`N_Batch`, 4, 2).

    Returns
    -------

        Tensor: The rotated polygons, shape (`N_Batch`, 4, 2).
    """
    rotation_matrix = (
        torch.stack(
            (
                torch.cos(angles),
                -torch.sin(angles),
                torch.sin(angles),
                torch.cos(angles),
            ),
            dim=1,
        )
        .view(-1, 2, 2)
        .repeat(polygons.size(1), 1, 1)
    )
    return torch.bmm(
        rotation_matrix.view(-1, 2, 2),
        polygons.view(-1, 2, 1),
    ).view(
        polygons.size(0), polygons.size(1), 2
    )  # (N, 4, 2)


def rotate_vertices_2d_batch(angle: Tensor, vertices_batch: Tensor) -> Tensor:
    """
    Rotate a batch of 2D vertices by given angles.

    Parameters
    ----------
    angle : Tensor
        The angles in radians to rotate the vertices.

    vertices_batch : Tensor
        The vertices to rotate, shape (..., 2).

    Returns
    -------
    Tensor
        The rotated vertices, shape (..., 2).
    """
    # Get number of vertices
    n_vertices = int(torch.prod(torch.tensor(vertices_batch.size()[:-1])))
    rotation_matrix = (
        torch.stack(
            (
                torch.cos(angle),
                -torch.sin(angle),
                torch.sin(angle),
                torch.cos(angle),
            ),
            dim=0,
        )
        .view(-1, 2, 2)
        .repeat(n_vertices, 1, 1)
    )  # (N, 2, 2)
    return torch.bmm(rotation_matrix, vertices_batch.view(-1, 2, 1)).view(
        vertices_batch.shape
    )


def generate_sha256_from_tensors(*tensors):
    hash_obj = hashlib.sha256()
    for tensor in tensors:
        hash_obj.update(tensor.numpy().tobytes())
    return hash_obj.hexdigest()


def generate_md5_from_tensors(*tensors):
    hash_obj = hashlib.md5()
    for tensor in tensors:
        hash_obj.update(tensor.numpy().tobytes())
    return hash_obj.hexdigest()