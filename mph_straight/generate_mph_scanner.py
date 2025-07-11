import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from torch import save as torch_save

from helper import (
    generate_transaxial_spect_geometry,
    plot_polygons_from_vertices_2d_mpl,
)

if __name__ == "__main__":
    # --- 1. Define Configuration Parameters ---
    # Using the correct physical dimensions from the paper
    cfg = {
        # FINAL CORRECTION: Use the physical diameter, not the FoV angle
        "pinhole_diameter_mm": 3.0,
        "n_pinholes": 18,
        "collimator_ring_radius_mm": 215.0,
        "collimator_thickness_mm": 20.0,
        "detector_ring_radius_mm": 215.0 + 542.0,
        "detector_thickness_mm": 9.5,
        "detector_crystal_width_mm": 3.5,
    }
    simulation_fov_mm = (32.0, 32.0)

    print("Generating physically accurate transaxial geometry...")

    # --- 2. Generate the Scanner Layout ---
    detector_units, collimator_segments = generate_transaxial_spect_geometry(**cfg)

    print(f"\nGenerated detector units: {detector_units.shape}")
    print(f"Generated solid collimator segments: {collimator_segments.shape}")

    # --- 3. Prepare Data and Save to .tensor File ---

    layout_entry = {
        "position": torch.tensor([0.0, 0.0, 0.0]),
        "detector units": detector_units,
        "plate segments": collimator_segments
    }
    output_data = {
        "configuration": cfg,
        "layouts": { "position 000": layout_entry }
    }
    output_tensor_filename = "final_accurate_pinhole_layout.tensor"
    torch_save(output_data, output_tensor_filename)
    print(f"\nSuccessfully saved configuration to: {output_tensor_filename}")

    # --- 4. Visualize for Confirmation ---
    fig, ax = plt.subplots(figsize=(12, 12))
    
    plot_polygons_from_vertices_2d_mpl(detector_units, ax, facecolor='lightblue', edgecolor='blue', alpha=0.8, label="Detector Crystals")
    plot_polygons_from_vertices_2d_mpl(collimator_segments, ax, facecolor='gray', edgecolor='black', label="Collimator Segments")

    fov_width, fov_height = simulation_fov_mm
    fov_patch = Rectangle(
        (-fov_width / 2, -fov_height / 2),
        fov_width, fov_height,
        edgecolor='red', facecolor='none', linestyle='--', linewidth=2,
        label=f'Simulation FOV ({fov_width}x{fov_height}mm)'
    )
    ax.add_patch(fov_patch)

    ax.set_aspect('equal', adjustable='box')
    plot_limit = cfg["detector_ring_radius_mm"] * 1.1
    ax.set_xlim([-plot_limit, plot_limit])
    ax.set_ylim([-plot_limit, plot_limit])
    
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Final Accurate Transaxial SPECT Geometry")
    ax.legend()
    plt.grid(True)

    output_viz_filename = "final_accurate_layout_viz.png"
    plt.savefig(output_viz_filename)
    print(f"Saved final accurate visualization to: {output_viz_filename}")