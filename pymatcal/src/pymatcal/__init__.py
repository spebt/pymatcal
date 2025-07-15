"""

pymatcal

========

Analytical system-matrix toolkit for SPECT.



Exposes high-level helpers plus low-level geometry / intersection

routines so that external scripts or notebooks can call them directly.

"""



# ---------------------------------------------------------------------

# Re-export public symbols from sub-modules

# ---------------------------------------------------------------------

from pymatcal.coord_transform import *          # noqa: F401,F403

from pymatcal.get_config       import *          # noqa: F401,F403

from pymatcal.intersections    import *          # noqa: F401,F403

from pymatcal.subdivs          import *          # noqa: F401,F403

from pymatcal.solid_angle      import *          # noqa: F401,F403

from pymatcal.pair_ppdf        import *          # noqa: F401,F403





# ---------------------------------------------------------------------

# Package metadata

# ---------------------------------------------------------------------

__version__      = "0.2.0"

__author__       = "Fang Han"

__email__        = "fhan0904@gmail.com"

__annotations__ = (

    "A package for calculating system response matrices for SPECT."

)



# ---------------------------------------------------------------------

# Explicit public API

# ---------------------------------------------------------------------

__all__ = [

    # config & top-level helpers

    "get_config",

    "get_pair_ppdf",

    "get_pair_ppdf_area",

    "get_pair_ppdf_binary",

    # coordinate transforms

    "coord_transform",

    "get_mtransform",

    # intersection routines

    "get_intersections_2d",

    "get_intersections_3d",          

    # geometry subdivision & rays

    "get_det_subdivs",

    "get_solid_angles",

    "get_AB_pairs",

    "get_centroids",

    "append_subdivs",

    "get_fov_voxel_center",

]

# __all__.append("schema")  # keep commented placeholder as before

