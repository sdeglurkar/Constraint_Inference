import numpy as np
from value_iteration import ValueIteration

grid_size = [3, 3]
final_value = np.zeros((grid_size[0], grid_size[1]))
#final_value[0][0] = 1
final_value[2][1] = 1
discount = 0.9

valiter = ValueIteration(grid_size)
value, q_value, optimal_policies = valiter.value_iteration(final_value, discount)
print(value)
print(optimal_policies)