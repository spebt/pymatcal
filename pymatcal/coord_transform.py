# 2d coordinate transformation, N x 3 Numpy array as input,
# z coordinate is untouched
import math
def coord_transform(angle_rad, x_shift, y_shift, input):
    m_rotation = []

    # Rotational
    # Angle in radians
    out[:, 0] = out[:, 0] * np.cos(angle_rad) + out[:, 1] * np.sin(angle_rad)
    out[:, 1] = out[:, 1] * np.cos(angle_rad) - out[:, 0] * np.sin(angle_rad)
    # Translational transformation
    out[:, 0] = out[:, 0] - x_shift
    out[:, 1] = out[:, 1] + y_shift
    return out