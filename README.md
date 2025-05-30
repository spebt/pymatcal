# PyMatCal

The repository contains a Python implementation of system response matrix analytical calculation code for single photon emission computed tomography (SPECT) scanner system.

Currently the code works only for two-dimensional (2D) systems, but it can be extended to three-dimensional (3D) systems.  Here are some of the key features of the code:

- **Modular Design**: The code is designed to be modular, allowing for easy addition of new features and functionality.
- **`PyTorch` Integration**: The code is built on top of the `PyTorch` library to leverage its powerful tensor operations, built-in GPU support and parallelization capabilities.

## How to Use

### Running locally

#### `test_ppdf_local.py`

Script for running locally with `Python` `for loop`.

Run the script with the following command:

```bash
python test_ppdf_local.py
```

#### `test_ppdf_distributed_local.py`

Script for running locally with `torch.distributed`

```bash
OMP_NUM_THREADS=2 torchrun --nproc_per_node=4 test_ppdf_distributed_local.py <layouts_path>/<layouts_filename>
```

#### `plot_ppdf.py`

Script for plotting the calculated system response matrices.

The script can be run in batch mode, i.e., it can take multiple files as input and generate plots for each file.

```bash
python plot_ppdf.py <filename_1> <filename_2> ...
```

> [!TIP]
> Example: _all the files in the current directory_ (`Linux`)

```bash
python plot_ppdf.py $(ls *.tensor)
```

### Running on the HPC cluster

Work in progress.
