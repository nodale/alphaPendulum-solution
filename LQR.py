import numpy as np
from control import lqr

m = 0.0194778
M = 0.5243
l = 0.50
g = 9.81
Bx = 10
B0 = 0.04

#Bx = 10
#B0 = 0.04
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

#Q = np.diag([2, 4, 6, 4]) 
#Q = np.diag([2, 0, 2, 0]) 
#R = np.array([[1]])        

Q = np.diag([1, 1, 1, 1]) 
R = np.array([[0.3]])        

K, S, E = lqr(A, B, Q, R)

#print("LQR gain K =\n", K)
print("Solution to Riccati equation S =\n", S)
#print("Closed-loop eigenvalues =\n", E)

print(" ".join(f"{k:.2f}" for k in K.flatten()))

