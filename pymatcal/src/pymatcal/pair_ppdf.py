


from __future__ import annotations

import numpy as np

from numpy.typing import NDArray



from .intersections import get_intersections_2d, get_intersections_3d



__all__ = ["pair_ppdf"]





def pair_ppdf(

    geoms: NDArray[np.float64],

    abpairs: NDArray[np.float64],

    config: dict,

) -> NDArray[np.float64]:

    """

    Parameters

    ----------

    geoms

        * 2-D: (N_det, 6) [x0,x1,y0,y1,seq,μ]

        * 3-D: (N_det, 8) [x0,x1,y0,y1,z0,z1,seq,μ]

    abpairs

        (N_ray, 6) [Ax,Ay,Az,Bx,By,Bz]

    config

        Must contain "fov nvx"  (tuple of ints).

        Optional  "solid_angle" (scalar or array).



    Returns

    -------

    (N_det, N_ray) array of attenuated path lengths.

    """

    volumetric = config["fov nvx"][2] > 1



    if volumetric:

        ret = get_intersections_3d(geoms, abpairs)

        mu_col = 7                          # μ is 8-th column

    else:

        ret = get_intersections_2d(geoms, abpairs)

        mu_col = 5                          # μ is 6-th column



    L = ret["intersections"]                # (N_det, N_ray)

    μ = geoms[:, mu_col][:, None]           # broadcast (N_det,1)



    atten = np.exp(-μ * L)                  

    solid = config.get("solid_angle", 1.0)  



    return solid * L * atten

