import matplotlib.pyplot as plt
import numpy as np

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_file = config["data_file"]
m = config["m"]
M = config["M"]
r = config["r"]

def read_data(filename, delimiter=None):
    columns = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            values = line.split(delimiter)
            if not columns:
                columns = [[] for _ in values]
            for i, value in enumerate(values):
                columns[i].append(float(value))
    return columns

def define_x(array):
    return [c / r for c in array]

def define_y(array):
    return [c for c in array]

columns = read_data('things.txt')

x = define_x(columns[0])

y = define_y(columns[1])

import numpy as np

def LSM(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    
    A = np.vstack([x, np.ones_like(x)]).T
    
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return a, b


Bx, _err = LSM(x, y)
print(1/Bx)

_ac = np.array([_err + ((Bx) * c / r) for c in columns[0]])

plt.scatter(columns[0], columns[1], color="r", label="experimental")
plt.plot(columns[0], _ac, color="b", label="model")
plt.xlabel('')

plt.ylabel('vel')
plt.xlabel('torque')
plt.legend()
plt.show()
