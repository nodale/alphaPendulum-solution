import cvxpy as cp
import numpy as np
from itertools import product

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
##############################################################

x_max = np.array([1/10.8, 1/120, 1/6.28, 1/0.8])
x_min = -x_max

u_max = 1/100
u_min = -u_max

##############################################################
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

n = A.shape[0]
m = B.shape[1]


vertices = np.array(list(product(*zip(x_min, x_max)))).T
Q = cp.Variable((n, n), symmetric=True)
Z = cp.Variable((m, n))  
#c = cp.Variable()  
#LMI = A.T @ P + P @ A - B @ Y - Y.T @ B.T
LMI = Q @ A.T + A @ Q + Z.T @ B.T + B @ Z

LMI_CC_placeholder = np.eye(B.shape[1])
LMI_CC1 = cp.bmat([
        [LMI_CC_placeholder, u_max * Z],
        [Z.T * u_max, Q]
    ])

LMI_CC2 = cp.bmat([
        [LMI_CC_placeholder, u_min * Z],
        [Z.T * u_min, Q]
    ])

constraints = [Q >> 1e-4*np.eye(n),
               LMI << -1e-6*np.eye(n),
               LMI_CC1 >> 1e-4*np.eye(n + B.shape[1]),
               LMI_CC2 >> 1e-4*np.eye(n + B.shape[1]),
               ]

for v in vertices.T:
    constraints.append(cp.quad_form(v, Q) <= 1)

objective = cp.Maximize(cp.log_det(Q))

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)

K = -Z.value @ np.linalg.inv(Q.value)

#print("Optimal P:\n", P.value)
#print("Optimal K:\n", K)
#print("Lyapunov sublevel c:\n", c.value)
print(" ".join(f"{k:.2f}, " for k in K.flatten()))

