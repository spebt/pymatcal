
# Introduction

**pymatcal** is a Python implementation of the ray-tracing-based analytical calculation of system response matrix of a single photon emission tomography system. Key features of the project are:
- Almost 100% Numpy-based, transition to a CUDA capable system should be easy with Python packages like [CuPy](https://cupy.dev/)
- It is written in an MPI-ready style. With the help of [mpi4py](https://mpi4py.readthedocs.io/en/stable/), we can run the calculation in parallel with a short Python script.
- System matrix I/O is handled with [h5py](https://docs.h5py.org/en/stable/) package. The package provides a convenient API for writing data in parallel with MPI.



# Get Started 
*Note* Parallel HDF5 installation on MS Windows OS is complicated, try the non-parallel test.

1. Install mpi4py (Optional on desktop/laptop)

2. Install HDF5 parallel version, check [h5py-paralell](https://docs.h5py.org/en/stable/mpi.html#) (Optional on desktop/laptop)

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
6. Run the test to generate the system matrix
```
cd test
```
- With MPI library and [Parallel-HDF5](https://support.hdfgroup.org/HDF5/PHDF5/) library properly setup
   
```bash
mpiexec -n 4 python get_all_ppdfs_mpi.py test_20240722_182815.yml
```
- No MPI but serial HDF5
```bash
python get_all_ppdfs_loop.py test_20240722_182815.yml
```

7. Plot the matrix together with the detector setup
```bash
python plot_ppdf_hdf5.py test/test_20240722_182815.hdf5 test_20240722_182815.yml
```

