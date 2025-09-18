import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are

import random

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

with open("controller.yaml", "r") as dg:
    controller = yaml.safe_load(dg)

K_HPC = np.array(controller["K_HPC"])
K_HAC = np.array(controller["K_HAC"])
P_HPC = np.array(controller["P_HPC"])
P_HAC = np.array(controller["P_HAC"])


########################################################

noise = 0.01

K = np.array([0,0,0,0])

########################################################

dM = (m + M) * (m * l * l) / 3 - 0.25 * m * m * l * l

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

dt = 0.01
t_max = 10000
steps = int(t_max / dt)

x = np.array([0, 0, 0.6, 0]) 

history = np.zeros((steps, 4))
time = np.linspace(0, t_max, steps)

fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.1, 1.5)
cart_width = 0.153
cart_height = 0.02
line, = ax.plot([], [], 'o-', lw=3, markersize=1)
cart_rect = plt.Rectangle((0,0), cart_width, cart_height, fc='blue')
ax.add_patch(cart_rect)

def init():
    line.set_data([], [])
    cart_rect.set_xy((-cart_width/2, 0))
    return line, cart_rect

def update(frame):
    global x

    u = -K @ x
    
    dx = A @ x + B.flatten() * u 
    x = x + dx * dt  
    history[frame] = x
    
    x_cart = x[0]
    theta = x[2]

    if random.uniform(0, 70) < 2:
        x[2] = x[2] + random.uniform(-noise, noise)
    else:
        x[2] = x[2]

    pend_x = x_cart + l * np.sin(theta)
    pend_y = cart_height + l * np.cos(theta)
    
    line.set_data([x_cart, pend_x], [cart_height, pend_y])
    cart_rect.set_xy((x_cart - cart_width/2, 0))
    
    return line, cart_rect

ani = FuncAnimation(fig, update, frames=steps, init_func=init,
                    blit=True, interval=dt*1000)
plt.show()
