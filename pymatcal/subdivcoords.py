import numpy as np
from coord_transform import *


def get_subdivcoords(config: dict):
    print(config["img nsub"])
    print(config["mmpvx"])
    imgSubLinespace_x = np.linspace(
        0, config["mmpvx"][0], int(config["img nsub"][0]) + 1
    )
    imgSubLinespace_y = np.linspace(
        0, config["mmpvx"][1], int(config["img nsub"][1]) + 1
    )
    imgSubLinespace_z = np.linspace(
        0, config["mmpvx"][2], int(config["img nsub"][2]) + 1
    )
    return np.array(
        np.meshgrid(
            0.5 * (imgSubLinespace_x[1:] + imgSubLinespace_x[:-1]),
            0.5 * (imgSubLinespace_y[1:] + imgSubLinespace_y[:-1]),
            0.5 * (imgSubLinespace_z[1:] + imgSubLinespace_z[:-1]),
        )
    ).T.reshape(int(config["img nsub"].prod()), 3)
