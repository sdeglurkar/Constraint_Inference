# From https://github.com/AndrewWalker/Dubins-Curves

import numpy as np
#import dubins
import matplotlib.pyplot as plt

x0 = 0
y0 = 0
theta0 = np.pi/2

x1 = 1
y1 = 3
theta1 = np.pi

q0 = (x0, y0, theta0)
q1 = (x1, y1, theta1)
turning_radius = 1.0
step_size = 0.5

#path = dubins.shortest_path(q0, q1, turning_radius)
#configurations, _ = path.sample_many(step_size)

#print(configuragtions)

#x = [conf[0] for conf in configurations]
#y = [conf[1] for conf in configurations]
plt.scatter([1, 2], [2, 3])
plt.show()

def g():
    return theta0 + theta1

if __name__ == '__main__':
    g()