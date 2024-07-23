# Introduction
`pymatcal` is a Python implementation of the ray-tracing-based analytical calculation of system response matrix of a single photon emission tomography system. Key features of the project are:
- Almost 100% Numpy-based, transition to a CUDA capable system should be easy with Python packages like [CuPy](https://cupy.dev/)
- It is written in an MPI-ready style. With the help of [mpi4py](https://mpi4py.readthedocs.io/en/stable/), we can run the calculation in parallel with a short Python script.
- System matrix I/O is handled with [h5py](https://docs.h5py.org/en/stable/) package. The package provides a convenient API for writing data in parallel with MPI.

## Dependency:
### Must have:
#### Python 3.9.6 and above
#### Python packages:
- [Numpy](https://numpy.org/install/)
- [pyyaml](https://pypi.org/project/PyYAML/)
- [jsonschema](https://python-jsonschema.readthedocs.io/en/stable/)

### Run in parallel on the cluster
#### MPI
To run with MPI, an MPI installation on the system is needed. We need to either use the Intel MPI or OpenMPI or even both. 

For now, we use Inter MPI with
``` Bash
module load intel
```
or simply
``` Bash
ml intel
```
To run in MPI paralellization with Python, we ca use 
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

mpi4py can be installed with pip. On the cluster, either use 
``` Bash
pip install --user mpi4py
``` 
or with a virtualenv
``` Bash
python -m venv venv
source venv/bin/activate
pip install mpi4py
``` 
mpi4py installation can be verify by
``` Bash
mpiexec -n 4 python -m mpi4py.bench helloworld
```

#### HDF5 Python binding
We use h5py mainly for its parallel writing with MPI. It provides a simple API for this purpose.
- [h5py](https://pypi.org/project/h5py/)

##### Building Paralell

## Example result
### Single pinhole, 1 detector unit
```{image} ./test_20240722_182815_data_det_000.png
:alt: singe-pinhole-1-detetor-unit
:width: 1024px
:align: center
```