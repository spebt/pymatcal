import torch
import time
import h5py
from rich.progress import Progress, BarColumn
from rich.console import Console

from raytracer_2d import (
    load_scanner_geometry_npz,
    set_default_device_as_cpu,
    get_geom_dict,
    get_ppdf,
)


if __name__ == "__main__":
    import sys

    # try:
    #     idx_start = int(sys.argv[1])
    #     idx_end = int(sys.argv[2])
    # except Exception as e:
    #     print("Requires two arguments: idx_start and idx_end")
    #     sys.exit(1)

    set_default_device_as_cpu()

    fov_dict = {
        "n_pixels": torch.tensor([512, 512]),
        "mm_per_pixel": torch.tensor([0.25, 0.25]),
        "center": torch.tensor([0.0, 0.0]),
    }
    scanner_geometry_dir = "scanner_geometry"
    plate_verts_2d, xtal_verts_2d = load_scanner_geometry_npz(
        scanner_geometry_dir + "/detector_cuboids.npz"
    )
    n_xtals = xtal_verts_2d.shape[0]
    geom_dict = get_geom_dict(plate_verts_2d, xtal_verts_2d, fov_dict)

    fov_n_pixels = int(torch.prod(fov_dict["n_pixels"]))
    ppdf = torch.empty(0, fov_n_pixels)

    elapsed_times = torch.zeros(n_xtals)
    progress = Progress(
        "{task.description}",
        BarColumn(),
        "{task.completed:03d}/{task.total:03d}",
        "[progress.percentage]{task.percentage:>3.0f}% Completed",
        transient=True,
        console=Console(),
    )
    out_h5file = h5py.File("scanner_ppdfs.hdf5", "w")
    ppdf = out_h5file.create_dataset(
        "ppdfs", shape=(n_xtals, fov_n_pixels), dtype="f"
    )
    task = progress.add_task("Computing PPDF", total=n_xtals)
    with progress:
        for idx in range(n_xtals):
            # progress.console.print(f"Current idx: {idx}")

            start_time = time.time()
            try:
                ppdf[idx] = (
                    get_ppdf(idx, geom_dict=geom_dict).unsqueeze(0).numpy()
                )

            except Exception as e:
                print("ID", idx, "\nError:", e)
                sys.exit(1)
            end_time = time.time()
            elapsed_times[idx] = end_time - start_time
            progress.update(task, advance=1)
        out_h5file.close()
        progress.console.print("PPDF calculation completed")
        progress.console.print(
            f"Average time per iteration: {elapsed_times.mean():.4f} seconds"
        )
        # progress.refresh()
