# main_script.py (Updated to Match Target Format)
import torch
from torch import save as torch_save, cat, tensor

# Assuming helper.py is in the same directory
from helper import (
    generate_transaxial_spect_geometry,
    plot_polygons_from_vertices_2d_mpl,
)

if __name__ == "__main__":
    # --- 1. Define Configuration Parameters (Unchanged) ---
    cfg = {
        "pinhole_diameter_mm": 3.0,
        "pinhole_opening_angle_deg": 27.0,
        "n_pinholes": 18,
        "collimator_ring_radius_mm": 215.0, # This is the junction radius
        "collimator_thickness_mm": 20.0,    # Total thickness (10mm inner, 10mm outer)
        "detector_ring_radius_mm": 215.0 + 542.0,
        "detector_thickness_mm": 9.5,
        "detector_crystal_width_mm": 3.5,
    }

    print("Generating bi-conical (double-tapered) pinhole geometry...")

    # --- 2. Generate the Scanner Layout (Unchanged) ---
    detector_units, inner_collimator, outer_collimator = generate_transaxial_spect_geometry(**cfg)

    # --- 3. Combine Collimator Parts to Match Target Format ---
    # The target format expects a single "plate segments" tensor.
    # We will concatenate the inner and outer rings.
    combined_plate_segments = cat((inner_collimator, outer_collimator), dim=0)
    print(f"\nCombined inner and outer collimator rings into a single tensor.")
    print(f"  Shape of final 'plate segments': {combined_plate_segments.shape}")

    # --- 4. Prepare Data for Saving in the Target Format ---
    # Create a layout dictionary for a single, untransformed position
    layout_entry = {
        "position": tensor([0.0, 0.0, 0.0]), # Position for the base layout
        "detector units": detector_units,
        "plate segments": combined_plate_segments, # Use the combined tensor
    }
    
    # Create the top-level output dictionary matching your script's structure
    # We add placeholders for metadata that your transformation script would generate.
    output_data = {
        "scanner MD5": "placeholder_md5_for_biconical_base_scanner",
        "motion_parameters": {
            "n_rotational_steps_defined": 1,
            "n_translational_shifts_grid": [1, 1],
            "translational_step_size_mm": [0.0, 0.0],
            "generated_n_positions": 1
        },
        "layouts": {
            "position 000": layout_entry # Store the single base layout
        }
    }

    # --- 5. Save the Layout ---
    out_file_name = "../data/scanner_layouts/mph_hourglass_configuration.tensor"
    
    print(f"\nSaving base bi-conical SPECT layout in the target format to:\n  {out_file_name}")
    torch_save(output_data, out_file_name)
    print("\nLayout saved successfully.")

    # --- Optional: Visualization (Slightly modified to show combined collimator) ---
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the combined collimator segments
    plot_polygons_from_vertices_2d_mpl(detector_units, ax, facecolor='lightblue', edgecolor='blue', alpha=0.8, label="Detector Crystals")
    plot_polygons_from_vertices_2d_mpl(combined_plate_segments, ax, facecolor='gray', edgecolor='black', label="Collimator (Combined)")
    
    ax.set_aspect('equal', adjustable='box')
    plot_limit = (cfg["detector_ring_radius_mm"] + 50) * 1.05
    ax.set_xlim([-plot_limit, plot_limit])
    ax.set_ylim([-plot_limit, plot_limit])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Base 2D Bi-Conical SPECT Geometry")
    ax.legend()
    plt.grid(True)
    plt.savefig("../data/plots/mph_hourglass_configuration.png")
    print("\nSaved visualization of the base layout to mph_hourglass_configuration.png")