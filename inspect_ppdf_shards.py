#!/usr/bin/env python
import os
import glob
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt


def summarize_and_visualize_shards(
    folder,
    max_files=None,
    max_det_per_file=3,
    save_plots=False,
    out_dir="ppdf_plots",
):
    """
    Scan a folder for HDF5 PPDF shard files, print summaries, and visualize
    a few detector volumes from each file.

    Assumes each file has:
      - dataset "ppdfs" with shape (N_det_in_file, Nx*Ny*Nz)
      - attrs on that dataset: "Nx", "Ny", "Nz", "size_mm", "start_idx", "end_idx"
    """

    patterns = ["*.h5", "*.hdf5"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files = sorted(files)

    if not files:
        print(f"No HDF5 files found in folder: {folder}")
        return

    if max_files is not None:
        files = files[:max_files]

    if save_plots:
        os.makedirs(out_dir, exist_ok=True)

    for f_idx, path in enumerate(files):
        print("\n========================================")
        print(f"[{f_idx+1}/{len(files)}] File: {path}")

        with h5py.File(path, "r") as f:
            if "ppdfs" not in f:
                print("  WARNING: no 'ppdfs' dataset found, skipping.")
                continue

            dset = f["ppdfs"]

            # Read attributes (they were stored on the dataset in your batch script)
            Nx = int(dset.attrs["Nx"])
            Ny = int(dset.attrs["Ny"])
            Nz = int(dset.attrs["Nz"])

            size_mm = dset.attrs.get("size_mm", None)
            if size_mm is not None:
                size_mm = np.array(size_mm, dtype=float)
            else:
                size_mm = np.array([Nx, Ny, Nz], dtype=float)  # fallback

            start_idx = int(dset.attrs.get("start_idx", 0))
            end_idx = int(dset.attrs.get("end_idx", start_idx + dset.shape[0] - 1))

            num_det_rows, total_voxels = dset.shape

            print(f"  Detectors in this file: {num_det_rows}")
            print(f"  Detector indices: {start_idx} .. {end_idx}")
            print(f"  Volume shape: Nx={Nx}, Ny={Ny}, Nz={Nz}")
            print(f"  total_voxels (flattened): {total_voxels}")
            print(f"  size_mm: {size_mm}")

            # Basic stats for the whole file (sampled or full)
            data_sample = dset[()]  # load everything; for big data, you could sample
            min_val = np.min(data_sample)
            max_val = np.max(data_sample)
            nnz = np.count_nonzero(data_sample)
            frac_nnz = nnz / data_sample.size

            print(f"  Value range: [{min_val:.3e}, {max_val:.3e}]")
            print(f"  Non-zeros: {nnz} / {data_sample.size} ({100*frac_nnz:.3f}%)")

            # Determine extents in mm for plotting
            sx, sy, sz = size_mm
            extent_xy = [
                -sx / 2.0,
                sx / 2.0,
                -sy / 2.0,
                sy / 2.0,
            ]

            # Visualize up to max_det_per_file detectors
            num_to_plot = min(num_det_rows, max_det_per_file)
            print(f"  Plotting central z-slice for {num_to_plot} detectors...")

            for local_i in range(num_to_plot):
                det_index = start_idx + local_i
                flat = dset[local_i]  # shape: (Nx*Ny*Nz,)
                vol = flat.reshape(Nx, Ny, Nz)

                # central z slice
                z_mid = Nz // 2
                slice_xy = vol[:, :, z_mid]

                plt.figure(figsize=(6, 5))
                im = plt.imshow(
                    slice_xy.T,  # transpose so x is horizontal, y is vertical
                    origin="lower",
                    interpolation="nearest",
                    extent=extent_xy,
                )
                plt.colorbar(im, label="PPDF value")
                plt.title(f"Detector {det_index} (file: {os.path.basename(path)}), z-index={z_mid}")
                plt.xlabel("x (mm)")
                plt.ylabel("y (mm)")
                plt.tight_layout()

                if save_plots:
                    fname = f"det_{det_index:04d}_file_{f_idx:02d}.png"
                    out_path = os.path.join(out_dir, fname)
                    plt.savefig(out_path, dpi=150)
                    plt.close()
                    print(f"    Saved plot: {out_path}")
                else:
                    # Show interactively (may open a lot of windows if many files)
                    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect and visualize PPDF HDF5 shards.")
    ap.add_argument("folder", type=str, help="Folder containing .h5 shard files")
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max number of HDF5 files to process (default: all)",
    )
    ap.add_argument(
        "--max-det-per-file",
        type=int,
        default=25,
        help="Max number of detector rows to plot per file",
    )
    ap.add_argument(
        "--save-plots",
        action="store_true",
        help="If set, save plots to PNG instead of showing them interactively.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="ppdf_plots",
        help="Output directory for saved plots (if --save-plots is set).",
    )

    args = ap.parse_args()

    summarize_and_visualize_shards(
        folder=args.folder,
        max_files=args.max_files,
        max_det_per_file=args.max_det_per_file,
        save_plots=args.save_plots,
        out_dir=args.out_dir,
    )
