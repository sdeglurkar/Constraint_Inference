from Unit_Tests.human_model import human_model
from Unit_Tests.generate_thetas import generate_thetas

grid_size = [3, 3]
list_binary_arrays = generate_thetas(grid_size)
discount = 0.9

theta = 1 * list_binary_arrays[2]
theta = theta.astype('float')
print(theta)
robot_state = [1, 1]
probability = human_model(0, theta, robot_state, discount, grid_size)

print(probability)