import numpy as np
from Robot import Robot

env = np.array([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

robot = Robot(environment=env, state=[0, 0], goal=[3, 3], grid_size=[4, 4], discount=0.9)
print(robot.optimal_policy())
robot.move(8)
print(robot.state)

new_env = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 0]])
robot.update_env(new_env)
print(robot.optimal_policy())
robot.move(8)
print(robot.state)