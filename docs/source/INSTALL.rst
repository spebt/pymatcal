=======
INSTALL
=======

Get Started
===========

Download the package
--------------------
- `Check out version v1.0.0 <https://github.com/spebt/pymatcal/archive/refs/tags/v1.0.0.zip>`_ 
- Version v1.0.0 is the first release of the package.
- More information can be found in the `PyMatCal releases page  <https://github.com/spebt/pymatcal/releases>`_
- On the CCR HPC cluster, you can download and unzip with:

.. code-block:: bash

	 wget \
	 https://github.com/spebt/pymatcal/archive/refs/tags/v1.0.0.zip;
	 unzip v1.0.0.zip;
	 rm v1.0.0.zip

Dependencies
------------

Core python package
^^^^^^^^^^^^^^^^^^^

- numpy==2.0.2
- jsonschema==4.23.0
- pyyaml== 6.0.2

Parallelization python package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- mpi4py

File I/O python package
^^^^^^^^^^^^^^^^^^^^^^^

- h5py == 3.11.0

h5py must be built againt a parallel HDF5 library.
``hdf5/1.14.1`` on CCR HPC node already has parallel support.

Install dependencies on CCR HPC Cluster
------------------------------------------

.. important::

   The following method is for CCR HPC cluster.

1. Load the required modules

.. code-block:: bash

	 module load intel python/3.9.6-bare hdf5/1.14.1

1. Install `mpi4py`

.. code-block:: bash

	 pip install --user mpi4py

2. Install `h5py` with parallel support

.. code-block:: bash

	 CC=mpicc HDF5_MPI="ON" \
	 HDF5_DIR="$(dirname "$(dirname "$(which h5pcc)")")" \
	 pip install --user --no-binary=h5py h5py

Install PyMatCal module on UB CCR HPC Cluster
---------------------------------------------

Install with `pip`

.. code-block:: bash
	
	cd pymatcal-1.0.0
	pip install --user .

