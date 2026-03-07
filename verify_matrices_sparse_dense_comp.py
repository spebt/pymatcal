import os
import h5py
import numpy as np
import scipy.sparse as sp
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track

def get_file_size_mb(filepath: str) -> float:
    return os.path.getsize(filepath) / (1024 * 1024)

def load_dense(filepath: str) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        if "ppdfs" not in f:
            raise KeyError(f"'ppdfs' not found in {filepath}. Is this the dense file?")
        return f["ppdfs"][:]

def load_sparse_as_dense(filepath: str) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        if "data" not in f:
            raise KeyError(f"'data' not found in {filepath}. Is this the sparse file?")
        
        data = f["data"][:]
        indices = f["indices"][:]
        indptr = f["indptr"][:]
        shape = tuple(f.attrs["shape"])
        
        csr_mat = sp.csr_matrix((data, indices, indptr), shape=shape)
        return csr_mat.toarray()

def main():
    parser = argparse.ArgumentParser(description="Batch verify dense vs sparse HDF5 folders.")
    parser.add_argument("--dense_dir", required=True, help="Directory containing legacy dense HDF5 files.")
    parser.add_argument("--sparse_dir", required=True, help="Directory containing new sparse HDF5 files.")
    parser.add_argument("--threshold", type=float, default=1e-9, help="Threshold used during sparsification.")
    
    args = parser.parse_args()
    console = Console()

    # 1. Identify common files
    dense_files = set([f for f in os.listdir(args.dense_dir) if f.endswith('.hdf5')])
    sparse_files = set([f for f in os.listdir(args.sparse_dir) if f.endswith('.hdf5')])
    
    common_files = sorted(list(dense_files.intersection(sparse_files)))
    
    if not common_files:
        console.print("[red]❌ No matching .hdf5 files found between the two directories.[/red]")
        return
        
    console.print(f"[cyan]Found {len(common_files)} matching files to verify.[/cyan]\n")

    # Setup the Summary Table
    table = Table(title="Batch Verification Summary", show_header=True, header_style="bold magenta")
    table.add_column("Filename", style="dim", width=25)
    table.add_column("Status", justify="center")
    table.add_column("Dense Size", justify="right")
    table.add_column("Sparse Size", justify="right")
    table.add_column("Reduction", justify="right", style="green")
    table.add_column("Fill Rate", justify="right", style="blue")

    passed_count = 0
    total_dense_mb = 0
    total_sparse_mb = 0

    # 2. Process each file
    for filename in track(common_files, description="Verifying matrices..."):
        dense_path = os.path.join(args.dense_dir, filename)
        sparse_path = os.path.join(args.sparse_dir, filename)

        try:
            # Load
            dense_matrix = load_dense(dense_path)
            sparse_matrix_decompressed = load_sparse_as_dense(sparse_path)

            # Check Shape
            if dense_matrix.shape != sparse_matrix_decompressed.shape:
                status = "[red]Shape Mismatch[/red]"
                table.add_row(filename, status, "-", "-", "-", "-")
                continue

            # Apply Threshold to Dense
            dense_matrix[dense_matrix < args.threshold] = 0.0

            # Verify Values
            is_equivalent = np.allclose(dense_matrix, sparse_matrix_decompressed, atol=1e-8)
            
            if is_equivalent:
                status = "[green]✅ PASS[/green]"
                passed_count += 1
            else:
                diff_count = np.sum(~np.isclose(dense_matrix, sparse_matrix_decompressed, atol=1e-8))
                status = f"[red]❌ FAIL ({diff_count} diff)[/red]"

            # Calculate Metrics
            d_size = get_file_size_mb(dense_path)
            s_size = get_file_size_mb(sparse_path)
            total_dense_mb += d_size
            total_sparse_mb += s_size
            
            reduction = (1 - (s_size / d_size)) * 100
            
            total_elements = dense_matrix.size
            non_zero_elements = np.count_nonzero(sparse_matrix_decompressed)
            fill_rate = (non_zero_elements / total_elements) * 100

            # Add to table
            table.add_row(
                filename, 
                status, 
                f"{d_size:.1f} MB", 
                f"{s_size:.1f} MB", 
                f"-{reduction:.1f}%", 
                f"{fill_rate:.2f}%"
            )

        except Exception as e:
            table.add_row(filename, f"[red]Error: {str(e)}[/red]", "-", "-", "-", "-")

    # 3. Final Report
    console.print(table)
    
    console.print("-" * 50)
    console.print(f"Total Files Verified : {len(common_files)}")
    console.print(f"Total Passed         : [green]{passed_count}/{len(common_files)}[/green]")
    
    if total_dense_mb > 0:
        total_reduction = (1 - (total_sparse_mb / total_dense_mb)) * 100
        console.print(f"Total Dense Storage  : {total_dense_mb:.2f} MB")
        console.print(f"Total Sparse Storage : {total_sparse_mb:.2f} MB")
        console.print(f"Overall Disk Savings : [green]-{total_reduction:.2f}%[/green]")
    console.print("-" * 50)

if __name__ == "__main__":
    main()