# visualize_scanner.py
import sys
import os
import torch
import matplotlib.pyplot as plt
from helper import plot_polygons_from_vertices_2d_mpl

def visualize_scanner_layout(input_filename: str):
    """
    Loads a scanner .tensor file and plots the detector units and plate segments.
    """
    # --- 1. Load Data ---
    print(f"Loading scanner data from: {input_filename}")
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"File {input_filename} does not exist.")
    
    scanner_layout_data = torch.load(input_filename)
    
    # Extract the tensors. Use .clone() to avoid any potential memory issues with matplotlib.
    detector_units = scanner_layout_data["detector units"].clone()
    plate_segments = scanner_layout_data["plate segments"].clone()
    
    print(f"-> Loaded {detector_units.shape[0]} detector units.")
    print(f"-> Loaded {plate_segments.shape[0]} plate segments.")

    # --- 2. Create Plot ---
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot detector units (the individual crystals)
    # We use the function from your helper.py file
    plot_polygons_from_vertices_2d_mpl(
        detector_units, 
        ax, 
        facecolor='#1f77b4',  # A nice blue color
        edgecolor='#1f77b4',
        linewidth=0.3,
        label='Detector Units'
    )
    
    # Plot plate segments (the panel outlines)
    '''
    plot_polygons_from_vertices_2d_mpl(
        plate_segments, 
        ax,
        facecolor='none',     # No fill
        edgecolor='#17becf',  # A nice cyan color
        linewidth=2.0,
        label='Plate Segments'
    )
    '''

    # Add a dashed hexagon to represent the approximate Field of View (FOV) for context
    # This mimics the orange hexagon in your original image.
    fov_radius = 75  # Approximate radius in mm, adjust as needed
    fov_angles = torch.linspace(0, 2 * torch.pi, 7) + (torch.pi / 6) # 7 points to close the loop, rotated
    fov_x = fov_radius * torch.cos(fov_angles)
    fov_y = fov_radius * torch.sin(fov_angles)
    ax.plot(fov_x, fov_y, color='orange', linestyle='-.', linewidth=1.5, label='Approx. FOV')

    # --- 3. Finalize and Show Plot ---
    ax.set_aspect('equal', 'box')  # CRITICAL: Ensures the geometry is not distorted
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Scanner Layout Visualization\n({os.path.basename(input_filename)})")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.splitext(input_filename)[0] + "_layout.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    
    # Display the plot
    print("\nDisplaying plot...")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python visualize_scanner.py <path_to_scanner_file.tensor>\n")
        sys.exit(1)
    
    visualize_scanner_layout(sys.argv[1])