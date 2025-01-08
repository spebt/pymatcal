import tables as tb
import numpy as np

# Create a new HDF5 file
h5file = tb.open_file(
    "system_configs.hdf5", mode="w", title="detector system configuration"
)

# Create a geometry group
# This group will contain the detector and collimator geometries
geom_group = h5file.create_group(
    "/", "detectors and collimators", "detectors and collimators"
)

# Create a table to store the detector geometry
collimator_geom_table = h5file.create_table(
    geom_group, "collimator geometry", {}, "collimator geometry"
)
