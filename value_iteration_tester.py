import numpy as np
from value_iteration import ValueIteration

grid_size = [5, 5]
final_value = np.zeros((grid_size[0], grid_size[1]))
final_value[0][0] = 50
final_value[1][1] = 50
discount = 0.9

valiter = ValueIteration(grid_size)
value, q_value, optimal_policies = valiter.value_iteration(final_value, discount)
print(value)
print(optimal_policies)