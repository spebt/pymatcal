#!/usr/bin/env python
"""
visualize_combined_ppdf.py

Streams and combines PPDF shards from a folder, then visualizes a 2D projection
along a specified plane (xy, yz, or zx).

Usage:
    python visualize_combined_ppdf.py ./shards --plane xy --det-mode sum
    python visualize_combined_ppdf.py ./shards --plane yz --z-mode max
"""

import os
import glob
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt


def collapse_dim(volume, mode, axis):
    """
    Collapse a 3D volume along a specific axis.
    
    volume: (Nx, Ny, Nz)
    mode: 'sum', 'mean', 'max'
    axis: 0, 1, or 2
    """
    if mode == "sum":
        return np.sum(volume, axis=axis)
    elif mode == "mean":
        return np.mean(volume, axis=axis)
    elif mode == "max":
        return np.max(volume, axis=axis)
    else:
        raise ValueError(f"Unknown collapse mode: {mode}")


def streaming_combine_shards(
    folder,
    det_mode="sum",
):
    """
    Stream over all .h5 files in 'folder' and combine across detectors.

    Returns:
        combined_vol: (Nx, Ny, Nz) after collapsing over detectors
        size_mm: np.array([sx, sy, sz])
        total_det_count: int (used for mean det_mode)
    """
    h5_files = sorted(
        [f for f in glob.glob(os.path.join(folder, "*.h5"))]
    )
    if not h5_files:
        raise RuntimeError(f"No .h5 files found in {folder}")

    Nx = Ny = Nz = None
    size_mm = None

    # Accumulators
    det_sum_vol = None        # for det_mode 'sum' and 'mean'
    det_max_vol = None        # for det_mode 'max'
    total_det_count = 0       # for det_mode 'mean'

    print(f"Found {len(h5_files)} shards in {folder}. Combining...")

    for fi, path in enumerate(h5_files):
        with h5py.File(path, "r") as f:
            if "ppdfs" not in f:
                print(f"[WARN] No 'ppdfs' in {path}, skipping.")
                continue

            dset = f["ppdfs"]
            n_det_file, total_vox = dset.shape

            # Read grid info (ensure consistency)
            this_Nx = int(dset.attrs["Nx"])
            this_Ny = int(dset.attrs["Ny"])
            this_Nz = int(dset.attrs["Nz"])
            this_size_mm = np.array(dset.attrs["size_mm"], dtype=float)

            if Nx is None:
                Nx, Ny, Nz = this_Nx, this_Ny, this_Nz
                size_mm = this_size_mm
                # initialize accumulators
                if det_mode in ["sum", "mean"]:
                    det_sum_vol = np.zeros((Nx, Ny, Nz), dtype=np.float64)
                elif det_mode == "max":
                    det_max_vol = np.full((Nx, Ny, Nz), -np.inf, dtype=np.float64)
            else:
                # sanity check consistency
                if (Nx, Ny, Nz) != (this_Nx, this_Ny, this_Nz):
                    raise ValueError(
                        f"Inconsistent volume shape in file {path}: "
                        f"({this_Nx}, {this_Ny}, {this_Nz}) vs ({Nx}, {Ny}, {Nz})"
                    )

            if n_det_file == 0:
                continue

            # Load only this file's detectors into memory
            data = dset[()]  # shape (n_det_file, Nx*Ny*Nz)
            data = data.reshape(n_det_file, Nx, Ny, Nz)

            # collapse across detectors for this file
            if det_mode == "sum" or det_mode == "mean":
                file_sum = np.sum(data, axis=0)  # (Nx, Ny, Nz)
                det_sum_vol += file_sum
                total_det_count += n_det_file
            elif det_mode == "max":
                file_max = np.max(data, axis=0)  # (Nx, Ny, Nz)
                det_max_vol = np.maximum(det_max_vol, file_max)
            else:
                raise ValueError(f"Unknown det-mode: {det_mode}")

            if (fi + 1) % 5 == 0 or (fi + 1) == len(h5_files):
                 print(f"[{fi+1}/{len(h5_files)}] Processed {n_det_file} det rows from {os.path.basename(path)}")

    # build final combined volume across detectors
    if det_mode == "sum":
        combined_vol = det_sum_vol   # already sum of all detectors
    elif det_mode == "mean":
        if total_det_count > 0:
            combined_vol = det_sum_vol / float(total_det_count)
        else:
            combined_vol = det_sum_vol
    elif det_mode == "max":
        combined_vol = det_max_vol
    else:
        raise ValueError(f"Unknown det-mode: {det_mode}")

    return combined_vol, size_mm, total_det_count


def visualize_combined_ppdf(
    folder,
    plane="xy",
    det_mode="sum",
    collapse_mode="mean",
    normalize=False,
    save=False,
    outfile="ppdf_combined.png",
):

    print(f"Streaming shards from: {folder}")
    vol3d, size_mm, total_det_count = streaming_combine_shards(folder, det_mode)
    Nx, Ny, Nz = vol3d.shape

    print(f"Combined volume shape: {vol3d.shape}")
    print(f"Physical Size (mm): {size_mm}")

    sx, sy, sz = size_mm

    # --- Plane Logic ---
    # We transpose (.T) the 2D projection for imshow because imshow expects (rows, cols) = (Y, X)
    # but our arrays are typically (X, Y). 
    
    if plane == "xy":
        # Collapse Z (axis 2)
        axis_to_collapse = 2
        xlabel, ylabel = "X (mm)", "Y (mm)"
        # Extent format: [left, right, bottom, top]
        extent = [-sx / 2.0, sx / 2.0, -sy / 2.0, sy / 2.0]
        
    elif plane == "yz":
        # Collapse X (axis 0) -> resulting shape (Ny, Nz)
        axis_to_collapse = 0
        xlabel, ylabel = "Y (mm)", "Z (mm)"
        extent = [-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0]

    elif plane == "zx" or plane == "xz":
        # Collapse Y (axis 1) -> resulting shape (Nx, Nz)
        axis_to_collapse = 1
        xlabel, ylabel = "X (mm)", "Z (mm)"
        extent = [-sx / 2.0, sx / 2.0, -sz / 2.0, sz / 2.0]
        
    else:
        raise ValueError(f"Unknown plane: {plane}. Use xy, yz, or zx.")

    # Collapse the 3rd dimension
    proj2d = collapse_dim(vol3d, collapse_mode, axis_to_collapse)

    # Normalize if requested
    if normalize:
        vmax = proj2d.max()
        if vmax > 0:
            proj2d = proj2d / vmax

    plt.figure(figsize=(7, 6))
    
    # proj2d.T maps the first dimension (horizontal in array notation) to columns (X-axis in plot)
    # and second dimension to rows (Y-axis in plot).
    im = plt.imshow(
        proj2d.T,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap="viridis"
    )
    plt.colorbar(im, label="PPDF Intensity")
    plt.title(
        f"Combined PPDF ({plane.upper()} Plane)\n"
        f"Det-mode={det_mode}, Collapse-3rd={collapse_mode}"
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save:
        plt.savefig(outfile, dpi=150)
        print(f"Saved image to {outfile}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Visualize combined PPDF from shards.")
    ap.add_argument("folder", type=str, help="Folder with .h5 PPDF shard files")
    
    ap.add_argument(
        "--plane",
        type=str,
        default="xy",
        choices=["xy", "yz", "zx", "xz"],
        help="The 2D plane to visualize (collapses the 3rd axis).",
    )

    ap.add_argument(
        "--det-mode",
        type=str,
        default="sum",
        choices=["sum", "mean", "max"],
        help="How to combine values across different detectors.",
    )
    ap.add_argument(
        "--collapse-mode",
        type=str,
        default="mean",
        choices=["sum", "mean", "max"],
        help="How to collapse the 3rd dimension (e.g., if plane=xy, how to flatten z).",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize final 2D image to [0,1]",
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="Save output figure instead of showing interactively.",
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="ppdf_combined.png",
        help="Output PNG file name if --save is given.",
    )

    args = ap.parse_args()

    visualize_combined_ppdf(
        folder=args.folder,
        plane=args.plane,
        det_mode=args.det_mode,
        collapse_mode=args.collapse_mode,
        normalize=args.normalize,
        save=args.save,
        outfile=args.outfile,
    )