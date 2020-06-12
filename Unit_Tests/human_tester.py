import numpy as np
from Human import Human

env = np.array([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 0]])

human = Human(environment=env, beta=10, discount=0.9, robot_goal=[3, 3], grid_size=[4, 4])
print(human.give_correction([1, 2], 2))