import tables as tb
import numpy as np

# Create a new HDF5 file
h5file = tb.open_file(
    "system_configs.hdf5", mode="w", title="Scanner System Configuration"
)

# Create a detector group
# This group will contain the detector and collimator geometries
detector_group = h5file.create_group("/", "detectors", "Detectors and collimators")


class cuboidGeometry(tb.IsDescription):
    """
    Define the cuboid geometry
    """

    name = tb.StringCol(16)
    id = tb.Int32Col()
    center_x = tb.Float32Col()
    center_y = tb.Float32Col()
    v1_x = tb.Float32Col()
    v1_y = tb.Float32Col()
    v1_z = tb.Float32Col()
    v2_x = tb.Float32Col()
    v2_y = tb.Float32Col()
    v1_z = tb.Float32Col()
    v3_x = tb.Float32Col()
    v3_y = tb.Float32Col()
    v3_z = tb.Float32Col()


# Create a table to store the geometries
geoms_table = h5file.create_table(
    detector_group, "geometries", cuboidGeometry, "Geometries"
)
cuboid = geoms_table.row


print(h5file)
h5file.close()
