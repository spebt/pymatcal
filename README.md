
# Introduction

**pymatcal** is a Python implementation of the ray-tracing-based analytical calculation of system response matrix of a single photon emission tomography system. Key features of the project are:
- Almost 100% Numpy-based, transition to a CUDA capable system should be easy with Python packages like [CuPy](https://cupy.dev/)
- It is written in an MPI-ready style. With the help of [mpi4py](https://mpi4py.readthedocs.io/en/stable/), we can run the calculation in parallel with a short Python script.
- System matrix I/O is handled with [h5py](https://docs.h5py.org/en/stable/) package. The package provides a convenient API for writing data in parallel with MPI.



# Get Started 

1. Install mpi4py 

2. Install HDF5 parallel version, check [h5py-paralell](https://docs.h5py.org/en/stable/mpi.html#)

3. Install h5py

4. Download the code
``` 
git https://github.com/spebt/pymatcal.git
```

5. Setup with `pipenv`
```
cd pymatcal
pipenv install && pipenv shell
```
6. Run the test
```
cd test
mpiexe -n 4 python get_all_ppdfs_mpi.py test_20240722_182815.yml
```