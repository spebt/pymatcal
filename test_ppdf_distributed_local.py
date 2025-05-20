import torch.distributed as dist
from torch import (
    Tensor,
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


def main():

    scanner_layouts, layouts_md5 = load_scanner_layouts(
        "scanner_layouts",
        "scanner_layouts_77faff53af5863ca146878c7c496c75e.tensor",
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

    ppdf = zeros_tensor(int(fov_dict["n pixels"].prod()), dtype=torch_float64)
    for crystal_id in range(10):
        ppdf = run_sfov_crystal(
            local_sfov_ids,
            sfov_pxs_ids,
            crystal_id,
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
    main()
