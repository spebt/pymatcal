import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn, Console
from matplotlib.colors import LogNorm

# Assuming these utility functions are available and correctly pathed
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import fov_tensor_dict

# --- Configuration ---
PROPERTIES_INPUT_DIR = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/ppdf-analysis/beam-analysis/output"
SCANNER_LAYOUTS_DIR = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/scanner_layouts"
SCANNER_LAYOUTS_FILENAME = "scanner_layouts_77faff53af5863ca146878c7c496c75e.tensor"
LAYOUTS_UNIQUE_ID = "77faff53af5863ca146878c7c496c75e" 

OUTPUT_DIR = "output/multiplexing_analysis_by_layer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_LAYOUTS = 24

# --- Layer Definition Strategy ---
# Set to False to use precise manual definition based on helper.py logic
DEFINE_LAYERS_ADAPTIVELY = False # RECOMMENDED TO SET TO FALSE to use precise definition

# Parameters from your scanner_layout_random_last_full and its defaults in helper.py
# Input to scanner_layout_random_last_full from your generate_base_scanner.py
BASE_SCANNER_GENERATOR_INPUT = torch.tensor([0.5, 0.4, 93.0, 38.0]) 
# Default kwargs from scanner_layout_random_last_full relevant to layer geometry
KWARGS_PLATE_THICKNESS = 2.0
KWARGS_CELL_SIZE_RADIAL = 3.36  # This is cell_size[0] in helper.py
KWARGS_N_CELLS_RADIAL = 8       # This is n_cells[0] in helper.py (defines N_EXPECTED_LAYERS)
N_EXPECTED_LAYERS = KWARGS_N_CELLS_RADIAL

# Calculate precise layer boundaries if using manual definition
if not DEFINE_LAYERS_ADAPTIVELY:
    # Radial start of the first cell layer in the detector panel grid:
    # input[2] (inner_radius) + input[3] (detector_array_to_plate_distance) + plate_thickness
    RADIAL_START_OF_CELL_GRID = BASE_SCANNER_GENERATOR_INPUT[2].item() + \
                                BASE_SCANNER_GENERATOR_INPUT[3].item() + \
                                KWARGS_PLATE_THICKNESS
    
    # Radial boundaries of the N_EXPECTED_LAYERS cell layers
    # These are N_EXPECTED_LAYERS + 1 edges defining N_EXPECTED_LAYERS bins
    MANUAL_RADIAL_LAYER_BINS = (torch.arange(float(KWARGS_N_CELLS_RADIAL + 1)) * KWARGS_CELL_SIZE_RADIAL + \
                               RADIAL_START_OF_CELL_GRID).cpu().numpy()
    
    MANUAL_LAYER_LABELS = [f"L{i+1} ({MANUAL_RADIAL_LAYER_BINS[i]:.2f}-{MANUAL_RADIAL_LAYER_BINS[i+1]:.2f}mm)"
                           for i in range(KWARGS_N_CELLS_RADIAL)]

FOV_CONFIG = fov_tensor_dict(
    n_pixels=(512, 512),
    mm_per_pixel=(0.25, 0.25),
    center_coordinates=(0.0, 0.0),
)
FOV_CENTER = FOV_CONFIG["center coordinates in mm"] 

BEAM_PROPS_DETECTOR_IDX_COL = 1 # Column index for detector_unit_idx in beam_properties

# --- Main Processing ---
if __name__ == "__main__":
    console = Console()
    try:
        scanner_layouts_data, _ = load_scanner_layouts(
            SCANNER_LAYOUTS_DIR, SCANNER_LAYOUTS_FILENAME
        )
    except FileNotFoundError:
        console.print(f"[bold red]Error: Scanner layouts file not found at {os.path.join(SCANNER_LAYOUTS_DIR, SCANNER_LAYOUTS_FILENAME)}[/bold red]")
        console.print("Please check SCANNER_LAYOUTS_DIR and SCANNER_LAYOUTS_FILENAME paths.")
        exit()
    except Exception as e:
        console.print(f"[bold red]Error loading scanner layouts: {e}[/bold red]")
        exit()


    radial_layer_bins = None
    layer_labels = None

    if DEFINE_LAYERS_ADAPTIVELY:
        console.rule("[bold cyan]Layer Definition: Adaptive Quantile-based[/bold cyan]")
        all_detector_radial_distances = []
        with Progress(console=console) as progress_bar:
            pass_task = progress_bar.add_task("Collecting radial distances for layer definition...", total=N_LAYOUTS)
            for layout_idx in range(N_LAYOUTS):
                _, detector_units_vertices = load_scanner_layout_geometries(
                    int(layout_idx), scanner_layouts_data
                )
                if detector_units_vertices is None or detector_units_vertices.shape[0] == 0:
                    progress_bar.update(pass_task, advance=1)
                    continue
                
                detector_centers = detector_units_vertices.mean(dim=1)
                radial_distances = torch.norm(detector_centers - FOV_CENTER.unsqueeze(0), dim=1)
                all_detector_radial_distances.extend(radial_distances.cpu().numpy())
                progress_bar.update(pass_task, advance=1)

        if not all_detector_radial_distances:
            console.print("[bold red]Error: No detector radial distances collected. Cannot define layers adaptively.[/bold red]")
            console.print("Consider using manual layer definition or check your scanner geometry files.")
            exit()

        unique_distances = np.unique(all_detector_radial_distances)
        console.print(f"Collected {len(all_detector_radial_distances)} radial distances. {len(unique_distances)} unique values.")
        console.print(f"Min/Max radial distance: {np.min(all_detector_radial_distances):.2f} mm / {np.max(all_detector_radial_distances):.2f} mm")

        quantiles = np.linspace(0, 1, N_EXPECTED_LAYERS + 1)
        radial_layer_bins = np.quantile(np.array(all_detector_radial_distances), quantiles)
        radial_layer_bins = np.unique(radial_layer_bins) 

        if len(radial_layer_bins) < 2: # Need at least 2 edges to make 1 bin
            console.print(f"[bold red]Error: Could not define sufficient unique adaptive bins (found {len(radial_layer_bins)} edges). Using fallback.[/bold red]")
            min_rad, max_rad = np.min(all_detector_radial_distances), np.max(all_detector_radial_distances)
            # Ensure min_rad and max_rad are distinct enough to form a bin
            if np.isclose(min_rad, max_rad): max_rad = min_rad + 1.0 # Ensure some width
            radial_layer_bins = np.array([min_rad - 0.5, max_rad + 0.5]) 
        elif len(radial_layer_bins) -1 != N_EXPECTED_LAYERS :
             console.print(f"[bold yellow]Warning: Adaptive method resulted in {len(radial_layer_bins)-1} layers, but expected {N_EXPECTED_LAYERS}.[/bold yellow]")
        
        num_defined_layers = len(radial_layer_bins) - 1
        if num_defined_layers <=0:
            console.print("[bold red]Error: No layers could be defined adaptively. Exiting.[/bold red]")
            exit()
        layer_labels = [f"L{i+1} ({radial_layer_bins[i]:.2f}-{radial_layer_bins[i+1]:.2f}mm)" for i in range(num_defined_layers)]
        console.print(f"Defined {num_defined_layers} adaptive radial layer bins: {np.round(radial_layer_bins, 2)}")

    else: # Manual Definition (Precise)
        console.rule("[bold cyan]Layer Definition: Manual (Precise from Generator Code)[/bold cyan]")
        radial_layer_bins = MANUAL_RADIAL_LAYER_BINS
        layer_labels = MANUAL_LAYER_LABELS
        console.print(f"Using {len(layer_labels)} manually defined layers with {len(radial_layer_bins)} bins (edges): {np.round(radial_layer_bins, 2)}")

    if radial_layer_bins is None or layer_labels is None or len(radial_layer_bins) < 2:
        console.print("[bold red]Error: Layer bins could not be established. Exiting.[/bold red]")
        exit()
    if len(layer_labels) != (len(radial_layer_bins)-1): # Basic sanity check
        console.print(f"[bold red]Error: Mismatch between number of layer labels ({len(layer_labels)}) and number of bins ({len(radial_layer_bins)-1}). Exiting.[/bold red]")
        exit()


    all_detector_layer_beam_counts_list = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
        ) as progress_bar:
        task_layouts = progress_bar.add_task("Processing layouts...", total=N_LAYOUTS)
        for layout_idx in range(N_LAYOUTS):
            progress_bar.update(task_layouts, description=f"Layout {layout_idx:02d}: Loading props")
            beams_properties_hdf5_filename = f"beams_properties_{LAYOUTS_UNIQUE_ID}_{layout_idx:02d}.hdf5"
            path_to_props = os.path.join(PROPERTIES_INPUT_DIR, beams_properties_hdf5_filename)

            if not os.path.exists(path_to_props):
                progress_bar.update(task_layouts, advance=1)
                continue
            try:
                with h5py.File(path_to_props, "r") as f:
                    layout_beams_properties = torch.from_numpy(f["beam_properties"][:])
            except Exception: # Catch broad exception for file I/O issues
                progress_bar.update(task_layouts, advance=1)
                continue

            if layout_beams_properties.shape[0] == 0:
                progress_bar.update(task_layouts, advance=1)
                continue

            progress_bar.update(task_layouts, description=f"Layout {layout_idx:02d}: Loading geom")
            _, detector_units_vertices = load_scanner_layout_geometries(int(layout_idx), scanner_layouts_data)
            
            if detector_units_vertices is None or detector_units_vertices.shape[0] == 0:
                progress_bar.update(task_layouts, advance=1)
                continue
            detector_centers = detector_units_vertices.mean(dim=1)

            progress_bar.update(task_layouts, description=f"Layout {layout_idx:02d}: Assigning layers")
            current_radial_distances = torch.norm(detector_centers - FOV_CENTER.unsqueeze(0), dim=1).cpu().numpy()
            
            # np.digitize returns 1-based indices for bins. Subtract 1 for 0-based layer_labels.
            detector_layer_numeric_indices = np.digitize(current_radial_distances, radial_layer_bins, right=False) - 1
            # Clip to ensure indices are within [0, len(layer_labels)-1]
            detector_layer_numeric_indices = np.clip(detector_layer_numeric_indices, 0, len(layer_labels) - 1)

            progress_bar.update(task_layouts, description=f"Layout {layout_idx:02d}: Counting beams")
            try:
                beam_props_df = pd.DataFrame(layout_beams_properties.numpy(),
                                            columns=[f'col_{i}' for i in range(layout_beams_properties.shape[1])])
            except Exception as e: # Should not happen if layout_beams_properties is a tensor
                 console.print(f"[bold yellow]Warning: Could not create DataFrame for layout {layout_idx}: {e}[/bold yellow]")
                 progress_bar.update(task_layouts, advance=1)
                 continue
            
            # Ensure the detector index column exists
            if f'col_{BEAM_PROPS_DETECTOR_IDX_COL}' not in beam_props_df.columns:
                console.print(f"[bold yellow]Warning: Detector index column 'col_{BEAM_PROPS_DETECTOR_IDX_COL}' not found in beam properties for layout {layout_idx}. Skipping beam counting.[/bold yellow]")
                progress_bar.update(task_layouts, advance=1)
                continue

            detector_beam_counts = beam_props_df.groupby(f'col_{BEAM_PROPS_DETECTOR_IDX_COL}').size().rename('num_beams')
            
            for det_idx_in_layout, num_beams in detector_beam_counts.items():
                det_idx_in_layout = int(det_idx_in_layout)
                # Ensure detector index from beam properties is a valid index for the geometry of this layout
                if det_idx_in_layout < len(detector_layer_numeric_indices): 
                    layer_num_idx = detector_layer_numeric_indices[det_idx_in_layout]
                    # Ensure layer_num_idx is valid for layer_labels (should be by clip)
                    if 0 <= layer_num_idx < len(layer_labels):
                        layer_str_label = layer_labels[layer_num_idx]
                        
                        all_detector_layer_beam_counts_list.append({
                            'layout': layout_idx,
                            'detector_idx_in_layout': det_idx_in_layout, # This is detector_unit_idx from beam_properties
                            'layer_label': layer_str_label,
                            'layer_numeric_idx': layer_num_idx,
                            'num_beams': num_beams
                        })
            progress_bar.update(task_layouts, advance=1)


    if not all_detector_layer_beam_counts_list:
        console.print("[bold red]No data collected across all layouts after processing. Exiting.[/bold red]")
        exit()

    final_df = pd.DataFrame(all_detector_layer_beam_counts_list)
    console.print(f"\nProcessed {final_df.shape[0]} beam-detector entries, forming final DataFrame.")
    if final_df.empty:
        console.print("[bold red]DataFrame is empty. No plots will be generated.[/bold red]")
        exit()
        
    # This count might be high if detector_idx_in_layout is just 0-N per layout
    # A more meaningful unique detector count would require a global ID system if detectors persist across layouts.
    # For this analysis, we care about instances of (detector_in_a_layer, num_beams)
    # console.print(f"Found {final_df['detector_idx_in_layout'].nunique()} unique detector_idx_in_layout values contributing beams.")
    console.print(f"Max number of beams per detector instance observed: {final_df['num_beams'].max()}")

    console.rule("[bold blue]Generating Visualizations[/bold blue]")
    # Heatmap: Layer vs. Num_Beams, Value is count of detector instances
    heatmap_data = final_df.groupby(['layer_numeric_idx', 'layer_label', 'num_beams']).size().reset_index(name='detector_count')
    
    if heatmap_data.empty:
        console.print("[bold yellow]Warning: No data for heatmap pivot table (heatmap_data is empty).[/bold yellow]")
    else:
        try:
            pivot_table = heatmap_data.pivot_table(index=['layer_numeric_idx', 'layer_label'], 
                                                   columns='num_beams', 
                                                   values='detector_count',
                                                   fill_value=0) # Fill NaNs (where no detectors fit a category) with 0
            pivot_table = pivot_table.sort_index(level='layer_numeric_idx') # Sort rows by numeric layer index
            ytick_labels_sorted = [idx[1] for idx in pivot_table.index] # Get sorted string labels for y-axis

            plt.figure(figsize=(max(10, pivot_table.shape[1] * 0.7), max(8, pivot_table.shape[0] * 0.6)))
            
            max_count = pivot_table.max().max() if not pivot_table.empty else 0
            # Use LogNorm if counts vary wildly AND max_count is substantial. vmin=1 to avoid log(0).
            current_norm = LogNorm(vmin=1, vmax=max_count) if max_count > 50 else None 

            plt.imshow(pivot_table, aspect='auto', cmap='viridis', norm=current_norm, origin='lower')

            plt.yticks(ticks=np.arange(len(ytick_labels_sorted)), labels=ytick_labels_sorted, fontsize=10)
            
            num_beam_cols = pivot_table.columns
            if len(num_beam_cols) > 20: 
                step = max(1, len(num_beam_cols) // 10) # Show about 10-15 ticks
                selected_ticks_indices = np.arange(0, len(num_beam_cols), step)
                selected_labels = num_beam_cols[selected_ticks_indices]
                plt.xticks(ticks=selected_ticks_indices, labels=selected_labels, fontsize=10)
            else:
                 plt.xticks(ticks=np.arange(len(num_beam_cols)), labels=num_beam_cols, fontsize=10)


            plt.xlabel("Number of Beams per Detector", fontsize=12)
            plt.ylabel("Detector Layer (Radial Distance from FOV Center)", fontsize=12)
            plt.title(f"Detector Count by Layer and Number of Beams (All {N_LAYOUTS} Layouts)", fontsize=14)
            cbar = plt.colorbar(label="Number of Detector Instances")
            
            # Annotate cells only for smaller heatmaps for readability
            if pivot_table.shape[0] <= 20 and pivot_table.shape[1] <= 20 and max_count > 0: 
                for i_row in range(pivot_table.shape[0]): # Iterate over rows (layers)
                    for j_col_idx, num_beam_val in enumerate(pivot_table.columns): # Iterate over columns (num_beams)
                        count = pivot_table.iloc[i_row, j_col_idx]
                        if count > 0: # Annotate non-zero counts
                            # Determine text color based on background
                            val_for_color_norm = count
                            if current_norm: # If LogNorm, use log of value for brightness check
                                val_for_color_norm = np.log1p(count) 
                                norm_max_log = np.log1p(max_count)
                                normalized_val = val_for_color_norm / (norm_max_log + 1e-9) if norm_max_log > 0 else 0.0
                            else: # Linear norm
                                normalized_val = count / (max_count + 1e-9)
                            
                            text_color = "white" if normalized_val < 0.6 else "black" # Adjusted threshold
                            plt.text(j_col_idx, i_row, int(count), ha="center", va="center", color=text_color, fontsize=7)

            plt.tight_layout()
            heatmap_filename = os.path.join(OUTPUT_DIR, "detector_beams_by_layer_heatmap.png")
            plt.savefig(heatmap_filename, dpi=300)
            console.print(f"Heatmap saved to: {heatmap_filename}")
            plt.close()

        except Exception as e:
            console.print(f"[bold red]Error generating heatmap: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            if 'pivot_table' in locals() and not pivot_table.empty:
                 console.print("Pivot table head:")
                 console.print(pivot_table.head())
            elif 'heatmap_data' in locals() and not heatmap_data.empty:
                 console.print("Heatmap data (ungrouped) head:")
                 console.print(heatmap_data.head())


    # Bar chart: Total beams per layer
    if not final_df.empty:
        total_beams_per_layer = final_df.groupby(['layer_numeric_idx', 'layer_label'])['num_beams'].sum().reset_index()
        total_beams_per_layer = total_beams_per_layer.sort_values('layer_numeric_idx')

        plt.figure(figsize=(12, 7))
        plt.bar(total_beams_per_layer['layer_label'], total_beams_per_layer['num_beams'], color='skyblue', edgecolor='black')
        plt.xlabel("Detector Layer", fontsize=12)
        plt.ylabel("Total Number of Beams Originating from Layer", fontsize=12)
        plt.title(f"Total Beams per Layer (All {N_LAYOUTS} Layouts)", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        barchart_filename = os.path.join(OUTPUT_DIR, "total_beams_per_layer_barchart.png")
        plt.savefig(barchart_filename, dpi=300)
        console.print(f"Barchart of total beams per layer saved to: {barchart_filename}")
        plt.close()

    if not final_df.empty:
        console.print("Generating plot: Average Beams per Detector in Layer...")
        # Step 1: Count unique detector instances per layer.
        # A detector instance is unique by its (layout, detector_idx_in_layout) pair.
        # We group by layer and then count unique combinations of layout and detector_idx_in_layout.
        # This correctly handles if a detector (by its ID within a layout) appears in multiple beam entries.
        
        # First, ensure we identify unique detectors within each layer correctly.
        # The 'detector_idx_in_layout' is unique *within* a given layout.
        # To count unique detector *instances* across all layouts that fall into a layer:
        unique_detectors_per_layer = final_df.groupby(['layer_numeric_idx', 'layer_label']) \
                                             .apply(lambda x: x[['layout', 'detector_idx_in_layout']].drop_duplicates().shape[0]) \
                                             .reset_index(name='num_detector_instances')

        # Step 2: Get total beams per layer (already calculated as total_beams_per_layer)
        # Merge this with the unique detector counts.
        avg_beams_data = pd.merge(total_beams_per_layer, unique_detectors_per_layer, 
                                  on=['layer_numeric_idx', 'layer_label'])

        # Step 3: Calculate average beams per detector instance
        avg_beams_data['avg_beams_per_detector'] = avg_beams_data['num_beams'] / avg_beams_data['num_detector_instances']
        # Handle cases where num_detector_instances might be 0 (though unlikely if it has beams)
        avg_beams_data['avg_beams_per_detector'].fillna(0, inplace=True) 

        avg_beams_data = avg_beams_data.sort_values('layer_numeric_idx')

        plt.figure(figsize=(12, 7))
        bars = plt.bar(avg_beams_data['layer_label'], avg_beams_data['avg_beams_per_detector'], 
                       color='lightcoral', edgecolor='black')
        
        # Add text labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * avg_beams_data['avg_beams_per_detector'].max(), 
                     f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

        plt.xlabel("Detector Layer", fontsize=12)
        plt.ylabel("Average Number of Beams per Detector Instance", fontsize=12)
        plt.title(f"Average Beams per Detector in Each Layer (All {N_LAYOUTS} Layouts)", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        avg_barchart_filename = os.path.join(OUTPUT_DIR, "avg_beams_per_detector_in_layer_barchart.png")
        plt.savefig(avg_barchart_filename, dpi=300)
        console.print(f"Barchart of average beams per detector in layer saved to: {avg_barchart_filename}")
        plt.close()

    console.print(f"\n[bold green]Multiplexing analysis by layer finished. Output in: {OUTPUT_DIR}[/bold green]")
