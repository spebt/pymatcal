=================
Quick Start
=================
*Note: Parallel HDF5 installation on MS Windows OS is complicated, try the non-parallel test.*

1. Install mpi4py (Optional on desktop/laptop)

2. Install HDF5 parallel version, check [h5py-paralell](https://docs.h5py.org/en/stable/mpi.html#) (Optional on desktop/laptop)

3. Install h5py

4. Download the code

.. code-block:: bash

   git https://github.com/spebt/pymatcal.git

1. Install the package locally
   
 
.. code-block:: bash

   cd pymatcal
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   git submodule update --init --recursive
   pip install -e .

2. Setup mpi4y and h5py parallel (Optional)
 
   This part varies depending on the system. Please refer to the official documentation of the libraries.

3. Run the test to generate the system matrix

.. code-block:: bash

   cd examples

- With MPI library and [Parallel-HDF5](https://support.hdfgroup.org/HDF5/PHDF5/) library properly setup
   
.. code-block:: bash

   mpiexec -n 4 python get_all_ppdfs_mpi.py test_small.yml

*4 is the number of processes to run in parallel. The number of processes should be less than the number of cores in the system.*

- No MPI but serial HDF5

.. code-block:: bash

   python get_all_ppdfs_loop.py test_small.yml

1. Check the output

   run *read_sysmat_hdf5.ipynb* Jupyter notebook in the examples folder to see the output

