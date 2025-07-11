if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from helper import (
        plot_polygons_from_vertices_2d_mpl,
    )
    import sys
    import os
    from torch import (
        load as torch_load,
        empty as empty_tensor,
        Tensor,
        pi,
    )

    # Load the scanner layouts
    filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        raise FileNotFoundError(f"File {filename} does not exist.")

    filename_unique_id = filename.split(".")[0].split("_")[-1]
    scanner_layouts_data = torch_load(filename)["layouts"]

    try:
        position_indices = list(map(int, sys.argv[2].split(",")))
    except Exception as e:
        print(
            "Usage: python plot_scanner_layout.py <filename> <layout_position_indices_comma_separated>"
        )
        sys.exit(1)

    # Plot the detector units
    fig, ax = plt.subplots(layout="constrained", figsize=(10, 10))
    ax.set_title(f"Scanner layout {position_indices[0]:03d}")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_aspect("equal")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)

    # Plot the detector units
    detector_units_polygons_collection = plot_polygons_from_vertices_2d_mpl(
        empty_tensor((1, 4, 2)),
        ax=ax,
        fc="C0",
        ec="none",
    )

    # Plot the plate segments
    plate_segments_polygons_collection = plot_polygons_from_vertices_2d_mpl(
        empty_tensor((1, 4, 2)),
        ax=ax,
        fc="C1",
        ec="none",
    )
    # Add a legend
    ax.legend(
        [
            detector_units_polygons_collection,
            plate_segments_polygons_collection,
        ],
        ["Detector units", "Plate segments"],
        loc="upper right",
    )
    for i in position_indices:

        # Extract the detector units and plate segments
        detector_units = scanner_layouts_data[f"position {i:03d}"][
            "detector units"
        ]
        plate_segments = scanner_layouts_data[f"position {i:03d}"][
            "plate segments"
        ]
        position = scanner_layouts_data[f"position {i:03d}"][
            "position"
        ].tolist()
        angle = float(position[0] * 180 / pi)

        print(
            f"Plotting scanner layout {i:03d} with {detector_units.shape[0]} detector units and {plate_segments.shape[0]} plate segments"
        )

        # Remove the previous detector units and plate segments
        if "detector_units_polygons_collection" in locals():
            detector_units_polygons_collection.remove()
        if "plate_segments_polygons_collection" in locals():
            plate_segments_polygons_collection.remove()

        ax.set_title(
            f"Scanner Position ID: {i:03d}, Transformation: (rotation: {angle:.2f}°, x: {position[1]:.2f} mm, y: {position[2]:.2f} mm)"
        )

        # Plot the detector units
        detector_units_polygons_collection = plot_polygons_from_vertices_2d_mpl(
            detector_units,
            ax=ax,
            fc="C0",
            ec="none",
        )
        # Plot the plate segments
        plate_segments_polygons_collection = plot_polygons_from_vertices_2d_mpl(
            plate_segments,
            ax=ax,
            fc="C1",
            ec="none",
        )

        fig.savefig(
            f"scanner_layout_{i:03d}_{filename_unique_id}.png",
            dpi=150,
            bbox_inches="tight",
        )
