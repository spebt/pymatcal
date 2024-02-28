import numpy as np
import yaml

configFileName = "configs/config.yml"
with open(configFileName, "r") as stream:
    try:
        yamlConfig = yaml.safe_load(stream)
    except yaml.YAMLError as err:
        print(err)

systemGeom = np.asarray(yamlConfig["detector geometry"])
detSubs = np.asarray(yamlConfig["detector"]["crystal n subdivision xyz"])
geom = systemGeom[2]

xlin = np.linspace(geom[0], geom[1], detSubs[0] + 1)
x_c = 0.5 * (xlin[1:] + xlin[:-1])
ylin = np.linspace(geom[2], geom[3], detSubs[1] + 1)
y_c = 0.5 * (ylin[1:] + ylin[:-1])
zlin = np.linspace(geom[4], geom[5], detSubs[2] + 1)
z_c = 0.5 * (zlin[1:] + zlin[:-1])
centers = np.array(np.meshgrid(x_c, y_c, z_c))
centers = centers.T.reshape(detSubs.prod(), 3)

imageVxpms = np.asarray(yamlConfig["image"]["voxel per mm xyz"])
print(1.0 / imageVxpms)
