


from __future__ import annotations

import numpy as np

from numpy.typing import NDArray



__all__ = [

    "get_intersections_2d",

    "get_intersections_3d",

]




def get_intersections_2d(

    geoms: NDArray[np.float64],

    abpairs: NDArray[np.float64],

) -> dict[str, NDArray[np.float64]]:

    """

    Ray–rectangle intersections in the x-y plane.



    Parameters

    ----------

    geoms   (N,6)  [x0,x1,y0,y1,seq,μ]

    abpairs (M,4)  [Ax,Ay,Bx,By]

    """

    x0, x1, y0, y1 = (geoms[:, i][:, None] for i in range(4))



    A = abpairs[:, :2]                      # (M,2)

    d = abpairs[:, 2:] - A                  # (M,2)

    invd = np.where(d != 0, 1.0 / d, np.inf)



    t0x = (x0 - A[:, 0]) * invd[:, 0]

    t1x = (x1 - A[:, 0]) * invd[:, 0]

    t0y = (y0 - A[:, 1]) * invd[:, 1]

    t1y = (y1 - A[:, 1]) * invd[:, 1]



    t_entry = np.maximum(np.minimum(t0x, t1x), np.minimum(t0y, t1y))

    t_exit  = np.minimum(np.maximum(t0x, t1x), np.maximum(t0y, t1y))



    hit = (t_exit > t_entry) & (t_exit > 0)

    seg_len = np.where(

        hit,

        (t_exit - t_entry) * np.linalg.norm(d, axis=1),

        0.0,

    )



    return {"intersections": seg_len, "ts": np.stack((t_entry, t_exit), axis=2)}






def get_intersections_3d(

    geoms: NDArray[np.float64],

    abpairs: NDArray[np.float64],

) -> dict[str, NDArray[np.float64]]:

    """

    Parameters

    ----------

    geoms   (N,8)  [x0,x1,y0,y1,z0,z1,seq,μ]

    abpairs (M,6)  [Ax,Ay,Az,Bx,By,Bz]

    """

    A = abpairs[:, :3]                      # (M,3)

    d = abpairs[:, 3:] - A                  # (M,3)

    invd = np.where(d != 0, 1.0 / d, np.inf)



    mins = geoms[:, ::2][:, None, :]        # (N,1,3)

    maxs = geoms[:, 1::2][:, None, :]       # (N,1,3)



    t0 = (mins - A[None, :, :]) * invd[None, :, :]

    t1 = (maxs - A[None, :, :]) * invd[None, :, :]



    t_entry = np.maximum.reduce(np.minimum(t0, t1), axis=2)  # (N,M)

    t_exit  = np.minimum.reduce(np.maximum(t0, t1), axis=2)



    hit = (t_exit > t_entry) & (t_exit > 0)

    seg_len = np.where(

        hit,

        (t_exit - t_entry) * np.linalg.norm(d, axis=1)[None, :],

        0.0,

    )



    return {"intersections": seg_len, "ts": np.stack((t_entry, t_exit), axis=2)}

