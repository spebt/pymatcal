import torch
import time
import h5py
from rich.progress import Progress, BarColumn

from raytracer_2d import (
    load_scanner_geometry_csv,
    get_geom_dict,
    get_ppdf,
)


if __name__ == "__main__":
    import sys

    torch.set_default_device("cpu")
    torch.set_num_threads(4)

    fov_dict = {
        "n_pixels": torch.tensor([256, 256]),
        "mm_per_pixel": torch.tensor([0.5, 0.5]),
        "center": torch.tensor([0.0, 0.0]),
    }
    fov_n_pixels = int(torch.prod(fov_dict["n_pixels"]))

    aperture_w_list = torch.arange(1, 5, 0.5)
    scanner_geometry_dir = "scanner_geometry"
    progress = Progress(
        "{task.description}",
        BarColumn(),
        "{task.completed:03d}/{task.total:03d}",
        "[progress.percentage]{task.percentage:>3.0f}% Completed",
        # transient=True,
        # console=Console(),
    )

    with progress:

        task1 = progress.add_task(
            "Processing...", total=aperture_w_list.shape[0]
        )
        for aperture_w in aperture_w_list:
            out_h5file = h5py.File(
                "ppdf_{}_mm_aperture.hdf5".format(aperture_w), "w"
            )

            # ppdf = torch.empty(0, fov_n_pixels)

            plate_vertices_2d, xtal_vertices_2d = load_scanner_geometry_csv(
                scanner_geometry_dir
                + "/"
                + "scanner_{}_mm_aperture.csv".format(aperture_w)
            )
            n_xtals = xtal_vertices_2d.shape[0]
            ppdf = out_h5file.create_dataset(
                "ppdfs", shape=(n_xtals, fov_n_pixels), dtype="f"
            )

            geom_dict = get_geom_dict(plate_vertices_2d, xtal_vertices_2d, fov_dict)

            idx_end = xtal_vertices_2d.shape[0]
            elapsed_times = torch.zeros(n_xtals)
            task2 = progress.add_task("Computing PPDF", total=n_xtals)
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
                progress.update(task2, advance=1)
            out_h5file.close()
            progress.console.print("PPDF calculation completed")
            progress.console.print(
                f"Average time per iteration: {elapsed_times.mean():.4f} seconds"
            )
            progress.update(task1, advance=1)
        # progress.refresh()
