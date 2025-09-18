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

x = np.array([0.0, 0.1, 1.54, 0.02])  

noise = 0.0001

K = np.array([0,0,0,0])

x_ddot = 0.0
theta_ddot = 0.0

########################################################

dt = 0.01
t_max = 10
steps = int(t_max / dt)

history = np.zeros((steps, 4))

fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-0.1, 1.5)
cart_width = 0.153
cart_height = 0.05
line, = ax.plot([], [], 'o-', lw=3, markersize=1)
cart_rect = plt.Rectangle((0,0), cart_width, cart_height, fc='blue')
ax.add_patch(cart_rect)

def init():
    line.set_data([], [])
    cart_rect.set_xy((-cart_width/2, 0))
    return line, cart_rect


def update(frame):
    global x, x_ddot, theta_ddot
    
    theta = x[2]
    theta_dot = x[3]
    x_cart = x[0]
    x_dot = x[1]
    
    u = -K @ x
    
    S = np.sin(theta)
    C = np.cos(theta)
    D = m*l*l*(M + m*(1 - C**2))
    
    theta_ddot = ( -(B0 * theta_dot) - (0.5 * m * l * C * x_ddot) - (0.5 * m * g * l * S) ) / (m * l * l * (1/3))
    x_ddot = (u - (Bx * x_dot) - (0.5 * m * l * C * theta_ddot) + (0.5 * m * l * S * theta_dot * theta_dot)) / (m + M)

    x[0] += x_dot * dt
    x[1] += x_ddot * dt
    x[2] += theta_dot * dt
    x[3] += theta_ddot * dt
    
    history[frame] = x
    
    pend_x = x[0] + l * np.sin(x[2])
    pend_y = cart_height + l * np.cos(x[2])
    
    line.set_data([x[0], pend_x], [cart_height, pend_y])
    cart_rect.set_xy((x[0] - cart_width/2, 0))
    
    return line, cart_rect

ani = FuncAnimation(fig, update, frames=steps, init_func=init,
                    blit=True, interval=dt*1000)
plt.show()
