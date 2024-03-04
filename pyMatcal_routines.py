import numpy as np


def coord_transform(angle_rad, x_shift, y_shift, trans_x, trans_y, input):
    out = np.zeros(3)
    out[0] = input[0] - trans_x
    out[1] = input[1] - trans_y

    # Rotational
    # Angle in radians
    out[0] = out[0] * np.cos(angle_rad) + out[1] * np.sin(angle_rad)
    out[1] = out[1] * np.cos(angle_rad) - out[0] * np.sin(angle_rad)

    # Translational transformation
    out[0] = out[0] - x_shift
    out[1] = out[1] + y_shift
    return out
