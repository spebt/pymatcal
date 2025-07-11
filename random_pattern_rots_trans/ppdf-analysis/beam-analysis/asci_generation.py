import sys
import glob
import h5py
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import PercentFormatter

if __name__ == "__main__":
    # --- MODIFIED: Use command-line arguments and glob to find files ---
    if len(sys.argv) != 2:
        print("\nUsage: python asci_generation.py <path_to_input_directory>")
        print("  - <path_to_input_directory>: Directory containing 'asci_histogram_*.hdf5' files.\n")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
        
    # Find all the asci_histogram files. The '*' is a wildcard.
    # Sorting ensures they are processed in a predictable order, though for summation it's not strictly necessary.
    asci_files = sorted(glob.glob(os.path.join(input_dir, "asci_histogram_*.hdf5")))

    if not asci_files:
        print(f"Error: No 'asci_histogram_*.hdf5' files found in {input_dir}")
        sys.exit(1)
        
    n_layouts_found = len(asci_files)
    print(f"Found {n_layouts_found} ASCI histogram files to aggregate.")
    # --- END MODIFIED SECTION ---

    # Get FOV and binning parameters (assuming they are fixed)
    fov_pixels_y, fov_pixels_x = 512, 512
    n_bins = 360

    # Initialize a single, empty histogram to accumulate all data.
    # This is now done only ONCE, before the loop.
    asci_histogram = torch.zeros(
        (fov_pixels_y * fov_pixels_x, n_bins),
        dtype=torch.int32,
    )

    # --- MODIFIED: Loop over all discovered files ---
    print("Aggregating histograms...")
    for asci_histogram_filename in asci_files:
        try:
            with h5py.File(asci_histogram_filename, "r") as f:
                layout_asci_histogram = torch.from_numpy(f["asci_histogram"][:])
            # The += operation is the core of the aggregation.
            asci_histogram += layout_asci_histogram
        except Exception as e:
            print(f"Warning: Could not read or process file {os.path.basename(asci_histogram_filename)}. Error: {e}. Skipping.")
    print("Aggregation complete.")

    # Calculate the final ASCI map from the single aggregated histogram.
    # This calculation is now done only ONCE, after the loop.
    asci_map = torch.count_nonzero(asci_histogram, dim=1) / float(n_bins)
    print(f"Final ASCI map shape: {asci_map.shape}")

    # --- Plotting the final aggregated map ---
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(
        figsize=(10, 8),
        layout="constrained",
    )
    
    # Define FOV extent in mm (assuming fixed values)
    fov_extent = [-64, 64, -64, 64]

    im = ax.imshow(
        asci_map.view(fov_pixels_y, fov_pixels_x).T,
        extent=fov_extent,
        origin="lower",
        cmap='viridis' # A common, perceptually uniform colormap
    )
    
    cbar = fig.colorbar(im, ax=ax, label="ASC Index")
    
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    # --- MODIFIED: Dynamic title and filename ---
    ax.set_title(
        f"ASC Index Map, {n_layouts_found} Layouts\nmax: {asci_map.max().item():.2%}, min: {asci_map.min().item():.2%}"
    )
    
    # Use PercentFormatter for the color bar for clear labeling
    cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
    cbar.update_ticks()
    
    out_figure_filename = os.path.join(plot_dir, f"asci_map_{n_layouts_found}_layouts.png")
    # --- END MODIFIED SECTION ---
    
    plt.savefig(out_figure_filename, dpi=300)

    print(f"\nFinal ASCI map saved to: {out_figure_filename}")

    # --- MODIFIED: The second, redundant block for "no rotation" has been removed ---
    # To generate a single-layout plot, you would simply ensure the input directory
    # contains only the single 'asci_histogram_....hdf5' file you want to plot.