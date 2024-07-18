# 2d coordinate transformation, N x 3 Numpy array as input,
# z coordinate is untouched
# The coordinate frame rotates counter-clock-wise by angle_rad
import math
import numpy as np
from ._utils import set_module

def coord_transformansform_v2(angle_rad, x_shift, y_shift, input):
    # Rotational
    # Angle in radians
    out = np.zeros((input.shape[0], 3))
    # print(angle_rad)
    out[:, 0] = input[:, 0] * np.cos(angle_rad) + input[:, 1] * np.sin(angle_rad)
    out[:, 1] = input[:, 1] * np.cos(angle_rad) - input[:, 0] * np.sin(angle_rad)
    # print(out)
    # Translational transformation
    out[:, 0] = out[:, 0] + x_shift
    out[:, 1] = out[:, 1] + y_shift
    return out

@set_module('pymatcal')
def get_mtransform(angle_deg: float, tx, ty) -> np.ndarray:
    """
    Get the transformation matrix for a given angle in degrees and translation values.

    :param angle_deg: The angle in degrees.
    :param tx: The translation value along the x-axis.
    :param ty: The translation value along the y-axis.
    :return: The transformation matrix as a numpy array.
    """
    # convert degrees to radians
    angle_rad = math.radians(angle_deg)
    return (
        np.array(
            [
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0],
                [0, 0, 1],
            ]
        ),
        np.array([tx, ty, 0]),
    )


# M(x+M_inv*a)=Mx+a

@set_module('pymatcal')
def coord_transformansform(m: tuple[np.ndarray], input: np.ndarray):
    """
    Apply coordinate transformation to the input array.

    Parameters:
    m (tuple[np.ndarray]): A tuple containing two numpy arrays representing the transformation matrix.
    input (np.ndarray): The input array to be transformed.

    Returns:
    np.ndarray: The transformed array.

    """
    return np.matmul(input, m[0]) + m[1]
