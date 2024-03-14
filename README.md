# pyMatcal

Python implementation of the system response matrix analytical calculation

## `pair-calc.py`

### Dependency

- `Python` version 3
- `Numpy`
- `MPI` can be `Open MPI` or `Intel MPI`

### Running on UB CCR HPC

#### On interactive node

1. Ask for an interactive node and login

```
salloc --qos=nih --partition=general-compute --job-name "interactive-job" --nodes=1 --ntasks=16 --mem=32G  --time=03:00:00;srun --pty /bin/bash --login
```

2. Setup the environment with `module load`, we need OpenMPI.

`module load ucx/1.13.1` is suggested by the ccr support team to suppress the error message but not working for now. However the error message will not prevent us from getting the result.
```
module load gcc/11.2.0 openmpi/4.1.1
module load scipy-bundle/2021.10
module load ucx/1.13.1
```

3. Install the python package dependency: mpi4py, numpy and pyyaml. This step only needs to be done once. But it won't hurt to put it in the `sbatch` script.
```
python3 -m pip install -r requirements.txt
```

4. Running `pair-calc.py` to produce the system response matrix file
```
mpirun python pair-calc.py
```

### Running on local desktops
#### Tested Enviornment:

- `Ubuntu` 22.04.1, `Linux kernel` 6.5.0-21-generic
- `Python` 3.10.12 with `GCC` 11.4.0
- `Open MPI` 4.1.2
- `Numpy` 1.26.3

## readSysmat.py
This script can be used to plot the calculated system response matrix
### Dependency

- `Python` version 3
- `Numpy`
- `Matplotlib`
### Running the script
Simply do:
```
python readSysmat.py
```
The script will read `configs/config.yml` and find the produced matrix. If the matrix file `.npz` is moved, you will need to modify the `readSysmat.py`
