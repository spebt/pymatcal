:html_theme.sidebar_secondary.remove:

.. pymatcal documentation master file, created by
   sphinx-quickstart on Wed Jul 17 13:48:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========
PyMatCal
========
.. Add your content using ``reStructuredText`` syntax. See the reStructuredText
   https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html
   documentation for details.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Table of Contents:

   PyMatCal <self>
   INSTALL <INSTALL.rst>
   Tutorial <guides/index.rst>
   API Reference <api-ref.md>

**pymatcal** is a Python implementation of the ray-tracing-based analytical calculation of system response matrix of a single photon emission tomography system. 

Key features of the project are:

- **Almost 100% Numpy-based.**

Transition to a CUDA capable system should be easy with Python packages like `CuPy <https://cupy.dev/>`_

- **It is written in an MPI-ready style.** 

With the help of `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_, we can run the calculation in parallel with a short Python script.

- **System matrix I/O can be handled with the** `h5py <https://docs.h5py.org/en/stable/>`_ **package.** 

The ``h5py`` package provides a convenient API for writing data in parallel with MPI.

------------------
Example PPDF
------------------

The following image shows the PPDFs of a single pinhole detector unit. The detector system consists of a single pinhole and a detector system. The image is generated by the `pymatcal` package.


.. image:: _static/img/singe-pinhole-1-detetor-unit.png
   :alt: singe-pinhole-1-detetor-unit
   :width: 1024px
   :align: center



