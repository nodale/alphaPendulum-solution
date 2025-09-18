import numpy as np
#your machine prolly dont need this
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import cvxpy as cp
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

scatter_color = config["plot"]["color_scatter"]
line_color = config["plot"]["color_line"]
xlabel = config["plot"]["xlabel"]
ylabel = config["plot"]["ylabel"]

#USER INPUT FROM DOWN HERE ONLY
#####################################################

one = 0
two = 2

x_max = np.array([1/0.8, 1/8, 1/1.54, 1/3.14])
x_min = -x_max

K_HPC = np.array([[-3.1623,  -8.3641,  93.3280,  34.9395]])
K_HAC = np.array([[-19.55,  -7.40,  36.44,  4.62]])
#K_HAC = np.array([[-20.00,  -14.93,  59.96,  9.59]])
#K_HPC = np.array([[-44.72,  -27.74,  90.01,  14.74]])
#K_HAC = np.array([[-282.93,  -76.34,  855.0,  30.0]])
#K_HPC = np.array([[-282.93,  -97.56,  285,  22.0]])

#NO MORE USER INPUT NEEDED
#####################################################

dM = (m + M) * (m*l*l)/3 - 0.25*m*m*l*l

A = np.array([
    [0, 1, 0, 0],
    [0, -(m*l*l*Bx)/(3*dM), (m*m*g*l*l)/(4*dM), (-0.5*m*l*B0)/dM],
    [0, 0, 0, 1],
    [0, (-0.5*m*l*Bx)/dM, 0.5*(m*g*l*(m+M))/dM, (-(m+M)*B0)/dM]
])

B = np.array([
    [0],
    [(m*l*l)/(3*dM)],
    [0],
    [(m*l)/(2*dM)]
])

n = A.shape[0]
m = B.shape[1]


############################################################

Acl = A - B @ K_HPC

vertices = np.array(list(product(*zip(x_min, x_max)))).T
Q = cp.Variable((n, n), symmetric=True)
Z = cp.Variable((m, n))  
LMI = Q @ Acl.T + Acl @ Q 

constraints = [Q >> 1e-4*np.eye(n),
               LMI << -1e-6*np.eye(n),
               ]

for v in vertices.T:
    constraints.append(cp.quad_form(v, Q) <= 1)

objective = cp.Maximize(cp.log_det(Q))

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)

Pa_HPC = np.linalg.inv(Q.value)

print("HPC P:\n", Pa_HPC)

P = np.array([
    [Pa_HPC[one][one], Pa_HPC[one][two]],
    [Pa_HPC[two][one], Pa_HPC[two][two]],
    ])

def V(x):
    return x.T @ P @ x

theta = np.linspace(0, 2*np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])  # unit circle
c = 1.0
L = np.linalg.cholesky(P)
#print(np.linalg.inv(L).shape[0])
#print(np.linalg.inv(L).shape[1])
#print(circle.shape[0])
#print(circle.shape[1])
ellipse_HPC = np.linalg.inv(L) @ circle * np.sqrt(c)

#############################################################
#K_HAC = np.array([[-10.00, -120, 199.40, 7.72]])

Acl = A - B @ K_HAC

vertices = np.array(list(product(*zip(x_min, x_max)))).T
Q = cp.Variable((n, n), symmetric=True)
Z = cp.Variable((m, n))  
LMI = Q @ Acl.T + Acl @ Q 

constraints = [Q >> 1e-4*np.eye(n),
               LMI << -1e-6*np.eye(n),
               ]

for v in vertices.T:
    constraints.append(cp.quad_form(v, Q) <= 1)

objective = cp.Maximize(cp.log_det(Q))

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)

Pa_HAC = np.linalg.inv(Q.value)

print("HAC P:\n", Pa_HAC)

P = np.array([
    [Pa_HAC[one][one], Pa_HAC[one][two]],
    [Pa_HAC[two][one], Pa_HAC[two][two]],
    ])

def V(x):
    return x.T @ P @ x

theta = np.linspace(0, 2*np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])  # unit circle
c = 1.0
L = np.linalg.cholesky(P)
#print(np.linalg.inv(L).shape[0])
#print(np.linalg.inv(L).shape[1])
#print(circle.shape[0])
#print(circle.shape[1])
ellipse_HAC = np.linalg.inv(L) @ circle * np.sqrt(c)

############################################################

data = {
    "K_HPC": K_HPC.tolist(),
    "K_HAC": K_HAC.tolist(),
    "P_HPC": Pa_HPC.tolist(),
    "P_HAC": Pa_HAC.tolist()
}

with open("controller.yaml", "w") as f:
    yaml.dump(data, f)

############################################################

plt.figure(figsize=(6,6))
plt.plot(ellipse_HPC[0, :], ellipse_HPC[1, :], 'b-', label=r'$x^T P x \leq 1$ (HPC Lyapunov region)')
plt.plot(ellipse_HAC[0, :], ellipse_HAC[1, :], 'r-', label=r'$x^T P x \leq 1$ (HAC Lyapunov region)')
#plt.axhline(0, color='k', linewidth=0.5)
#plt.axvline(0, color='k', linewidth=0.5)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
#plt.axis("equal")
plt.title("stability region")
plt.show()

