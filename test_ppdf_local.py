if __name__ == "__main__":
    import torch.profiler
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from torch import float64 as torch_float64
    from torch import save as torch_save
    from torch import zeros as zeros_tensor

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

    scanner_layouts, layouts_md5 = load_scanner_layouts(
        "scanner_layouts",
        "scanner_layouts_77faff53af5863ca146878c7c496c75e.tensor",
    )

    mu_dict = {"plate": 3.5, "crystal": 0.475}  # mm^-1

    fov_dict = fov_tensor_dict((512, 512), (128, 128), (0.0, 0.0), (16, 16))

    sfov_pxs_ids, sfov_pxs_coords, sfov_corners_batch = sfov_properties(
        fov_dict
    )
    n_sfov = int(fov_dict["n subdivisions"].prod())
    print(f"Number of FOV Blocks: {n_sfov}")
    print(f"FOV Block Size: {int(sfov_pxs_coords.shape[1])}")
    crystal_n_subs = (5, 5)

    subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

    layout_idx = 0

    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(0, scanner_layouts)

    crystal_idx = 6

    progress = Progress(
        SpinnerColumn(),
        BarColumn(),
        " | ",
        "[progress.description]{task.description}",
        "[progress.percentage]{task.completed}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    ppdf = zeros_tensor(int(fov_dict["n pixels"].prod()), dtype=torch_float64)
    print(sfov_pxs_ids.shape)
    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,  # Optional: records tensor shapes
        profile_memory=True,  # Optional: profiles memory usage
        with_stack=True,  # Optional: records call stacks for better source attribution
        # (can add overhead)
    ) as prof:
        with progress:
            task = progress.add_task(
                description="Calculating PPDFs", total=n_sfov
            )
            progress.update(task, advance=0)
            for sfov_idx in range(n_sfov):
                # --- Your tensor-heavy code section STARTS here ---
                with torch.profiler.record_function(f"sfov_ppdf_{sfov_idx}"):
                    ppdf[sfov_pxs_ids_1d[sfov_idx]] = ppdf_2d_local(
                        sfov_idx,
                        crystal_idx,
                        sfov_pxs_coords,
                        sfov_corners_batch,
                        plate_objects_vertices,
                        crystal_objects_vertices,
                        plate_objects_edges,
                        crystal_objects_edges,
                        subdivision_grid,
                        mu_dict,
                    )
                progress.update(task, advance=1)
    # --- Your tensor-heavy code section ENDS here ---

    # print("CPU time usage:")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    # (Recommended) Export for detailed analysis in TensorBoard or Chrome Tracing
    prof.export_chrome_trace("tensor_ops_trace.json")
    torch_save(ppdf, f"ppdf_{crystal_idx:03}_loop.tensor")
