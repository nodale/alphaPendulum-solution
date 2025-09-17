import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import cvxpy as cp
from itertools import product

one = 2
two = 3

x_max = np.array([1/10.8, 1/120, 1/6.28, 1/0.8])
x_min = -x_max

m = 0.0194778
M = 0.5243
l = 0.50
g = 9.81
Bx = 10
B0 = 0.06

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
K_HPC = np.array([[-1.41, -22.55, 205.18, 7.95]])

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

Pa = np.linalg.inv(Q.value)

#print("Lyapunov matrix P:\n", P)

P = np.array([
    [Pa[one][one], Pa[one][two]],
    [Pa[two][one], Pa[two][two]],
    ])

def V(x):
    return x.T @ P @ x

theta = np.linspace(0, 2*np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])  # unit circle
c = 1.0
L = np.linalg.cholesky(P)
print(np.linalg.inv(L).shape[0])
print(np.linalg.inv(L).shape[1])
print(circle.shape[0])
print(circle.shape[1])
ellipse_HPC = np.linalg.inv(L) @ circle * np.sqrt(c)

#############################################################
#K_HAC = np.array([[-10.00, -120, 199.40, 7.72]])

K_HAC = np.array([[-0.04,  -10.22,  8.79,  -1.88]])
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

Pa = np.linalg.inv(Q.value)

#print("Lyapunov matrix P:\n", P)

P = np.array([
    [Pa[one][one], Pa[one][two]],
    [Pa[two][one], Pa[two][two]],
    ])

def V(x):
    return x.T @ P @ x

theta = np.linspace(0, 2*np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])  # unit circle
c = 1.0
L = np.linalg.cholesky(P)
print(np.linalg.inv(L).shape[0])
print(np.linalg.inv(L).shape[1])
print(circle.shape[0])
print(circle.shape[1])
ellipse_HAC = np.linalg.inv(L) @ circle * np.sqrt(c)

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
plt.title("Controller Stability Region via Lyapunov Function")
plt.show()

