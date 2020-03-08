from human_model import human_model
from generate_thetas import generate_thetas

grid_size = [2, 2]
list_binary_arrays = generate_thetas(grid_size)
discount = 0.9

theta = list_binary_arrays[2]
print(theta)
robot_state = [1, 1]
probability = human_model(0, theta, robot_state, discount, grid_size)

print(probability)