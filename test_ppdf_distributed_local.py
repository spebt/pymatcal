import torch.distributed as dist
from torch import (
    Tensor,
    tensor,
    arange,
    zeros as zeros_tensor,
    save as torch_save,
    float64 as torch_float64,
)

from scanner_modeling.geometry_2d import (
    fov_tensor_dict,
    load_scanner_geometry_from_layout,
    load_scanner_layouts,
)
from scanner_modeling.raytracer_2d import (
    ppdf_2d_local,
    sfov_properties,
    subdivision_grid_rectangle,
)


def setup_distributed():
    print("Setting up distributed environment...")
    dist.init_process_group(backend="gloo", init_method="env://")
    print("Done setup distributed environment...")


def cleanup_distributed():
    dist.destroy_process_group()


def get_local_sfov_ids(sfov_ids: Tensor, rank: int, size: int) -> Tensor:
    return sfov_ids[rank::size]


def run_sfov_crystal(
    local_sfov_ids: Tensor,
    sfov_px_ids: Tensor,
    crystal_id: int,
    fov_npx: Tensor,
    *args,
):
    ppdf = zeros_tensor(int(fov_npx.prod()), dtype=torch_float64)
    for sfov_id in local_sfov_ids:
        ppdf[
            sfov_px_ids[sfov_id, :, 0] * fov_npx[0] + sfov_px_ids[sfov_id, :, 1]
        ] = ppdf_2d_local(int(sfov_id), crystal_id, *args)

    dist.all_reduce(ppdf, op=dist.ReduceOp.SUM)
    return ppdf


def main(
    layouts_dir: str = "scanner_layouts",
    layouts_filename: str = "layouts.tensor",
):
    """
    Main function to run the distributed PPDF calculation for crystals in a scanner layout.
    This function initializes the distributed environment, loads the scanner layouts,
    sets up the field of view (FOV) properties, and runs the PPDF calculation for specified crystals.
    It saves the results to files named `ppdf_<crystal_id>.tensor` for each crystal processed.

    Parameters
    ----------
    layouts_dir : str
        Directory containing the scanner layouts.

    layouts_filename : str
        Filename of the scanner layouts.

    Returns
    -------
    None

    """

    scanner_layouts, layouts_md5 = load_scanner_layouts(
        layouts_dir,
        layouts_filename,
    )

    mu_dict = {"plate": 3.5, "crystal": 0.475}  # mm^-1

    fov_dict = fov_tensor_dict((512, 512), (128, 128), (0.0, 0.0), (8, 8))

    sfov_pxs_ids, sfov_pxs_coords, sfov_corners_batch = sfov_properties(
        fov_dict
    )
    n_sfov = int(fov_dict["n subdivisions"].prod())

    crystal_n_subs = (5, 5)

    subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

    layout_idx = 0

    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    n_crystals = int(crystal_objects_vertices.shape[0])
    # crystal_id = 400

    setup_distributed()

    rank = dist.get_rank()
    size = dist.get_world_size()
    args = (
        sfov_pxs_coords,
        sfov_corners_batch,
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
        subdivision_grid,
        mu_dict,
    )
    sfov_ids_global = arange(n_sfov)

    local_sfov_ids = get_local_sfov_ids(sfov_ids_global, rank, size)

    crystal_ids = tensor([0])

    ppdf = zeros_tensor(int(fov_dict["n pixels"].prod()), dtype=torch_float64)
    for crystal_id in crystal_ids:
        ppdf = run_sfov_crystal(
            local_sfov_ids,
            sfov_pxs_ids,
            int(crystal_id),
            fov_dict["n pixels"],
            *args,
        )
        dist.barrier()
        if crystal_id % size == rank:
            torch_save(
                ppdf,
                f"ppdf_{crystal_id:03}.tensor",
            )

    cleanup_distributed()


if __name__ == "__main__":
    import sys, os

    if len(sys.argv) != 2:
        print(
            "Usage: python test_ppdf_distributed_local.py <layouts_dir>/<layouts_filename>"
        )
        sys.exit(1)
    layouts_dir = sys.argv[1].split("/")[0]
    layouts_filename = sys.argv[1].split("/")[-1]

    # Ensure the layouts file exists
    if not os.path.exists(os.path.join(layouts_dir, layouts_filename)):
        print(
            f"File {layouts_filename} does not exist in directory {layouts_dir}."
        )
        sys.exit(1)

    main(layouts_dir, layouts_filename)
