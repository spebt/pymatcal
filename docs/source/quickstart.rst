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


5. Setup with `pipenv`

.. code-block:: bash

   cd pymatcal
   pipenv install && pipenv shell

6. Run the test to generate the system matrix

.. code-block:: bash

   cd examples

- With MPI library and [Parallel-HDF5](https://support.hdfgroup.org/HDF5/PHDF5/) library properly setup
   
.. code-block:: bash

   mpiexec -n 4 python get_all_ppdfs_mpi.py test_20240722_182815.yml

- No MPI but serial HDF5

.. code-block::: bash

   python get_all_ppdfs_loop.py test_20240722_182815.yml

7. Plot the matrix together with the detector setup

.. code-block::: bash

   python plot_ppdf_hdf5.py test/test_20240722_182815.hdf5 test_20240722_182815.yml

Install the package locally
===========================
To install the package locally, run the following command in the root directory of the package:

.. code-block:: bash

   pip install .

