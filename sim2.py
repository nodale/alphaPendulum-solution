import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are

import math

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

vel_max = [0.0]
########################################################

x = np.array([0.0, 0.0, 3.14, -14])  

noise = 0.05

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
cart_width = 0.05
cart_height = 0.02
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

    if abs(theta) > 1.57:
        print("SWING UP")
        u = -0 * ((0.0374 * theta_dot * theta_dot / 6) + 0.7443 * (1 - math.cos(theta))) * np.sign(theta_dot * math.cos(theta))
    elif x.T @ P_HPC @ x < 1:
        print("HPC")
        u = K_HPC @ x
    elif x.T @ P_HAC @ x < 1:
        print("HAC")
        u = K_HAC @ x
    else:
        u = K_HAC @ x * 0

    #print(f"{x.T @ P_HAC @ x:.4f}, {x.T @ P_HPC @ x:.4f}")

    S = np.sin(theta)
    C = np.cos(theta)
    
    theta_ddot = 3 * ( -(B0 * theta_dot) - (0.5 * m * l * C * x_ddot) + (0.5 * m * g * l * S) ) / (m * l * l) 
    x_ddot = (u - (Bx * x_dot) - (0.5 * m * l * C * theta_ddot) + (0.5 * m * l * S * theta_dot * theta_dot)) / (m + M)

    x[0] += x_dot * dt
    x[1] += x_ddot * dt
    x[2] += theta_dot * dt
    x[2] = (x[2] + np.pi) % (2 * np.pi) - np.pi
    x[3] += theta_ddot * dt

    global vel_max
    vel_max.append(x[1])
    print(max(vel_max))
    if random.uniform(0, 70) < 2:
        x[2] = x[2] + random.uniform(-noise, noise)
    else:
        x[2] = x[2]
    
    history[frame] = x
    
    pend_x = x[0] + l * np.sin(x[2])
    pend_y = cart_height + l * np.cos(x[2])
    
    line.set_data([x[0], pend_x], [cart_height, pend_y])
    cart_rect.set_xy((x[0] - cart_width/2, 0))
    

    ax.set_xlim(x[0] - 1.0, x[0] + 1.0) 

    return line, cart_rect

ani = FuncAnimation(fig, update, frames=steps, init_func=init,
                    blit=False, interval=dt*1000)

plt.show()
