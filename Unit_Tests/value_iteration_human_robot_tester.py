import numpy as np
from value_iteration_human_robot import ValueIterationHumanRobot

grid_size = [5, 5]
final_value = np.zeros((grid_size[0], grid_size[1]))
final_value[0][0] = 10
final_value[2][1] = -10
discount = 0.9

valiter = ValueIterationHumanRobot(grid_size)
value, q_value, optimal_policies_human, optimal_policies_robot = valiter.value_iteration(final_value, discount)
print(value)
print(optimal_policies_human)
print(optimal_policies_robot)