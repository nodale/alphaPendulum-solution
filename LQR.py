import numpy as np
from control import lqr

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_file = config["data_file"]
r = config["r"]
m = config["m"]
M = config["M"]
l = config["l"]
g = config["g"]
Bx = config["Bx"]
B0 = config["B0"]

#USER INPUT HERE
###############################################################

Q = np.diag([1, 1, 1, 1]) 
R = np.array([[0.3]])        

################################################################

dM = (m + M) * (m * l * l) / 3 - 0.25 * m * m * l * l

A = np.array([
    [0, 1, 0, 0],
    [0, -(m * l * l * Bx) / (3 * dM), (m * m * g * l * l) / (4 * dM), (-0.5 * m * l * B0) / dM],
    [0, 0, 0, 1],
    [0, (-0.5 * m * l * Bx) / dM, 0.5 * (m * g * l * (m + M)) / dM, (-(m + M) * B0) / dM]
])

B = np.array([
    [0],
    [(m * l * l) / (3 * dM)],
    [0],
    [(m * l) / (2 * dM)]
])


K, S, E = lqr(A, B, Q, R)

print(" ".join(f"{k:.2f}" for k in K.flatten()))

