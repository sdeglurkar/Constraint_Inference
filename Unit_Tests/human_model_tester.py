from Unit_Tests.human_model import human_model
from Unit_Tests.generate_thetas import generate_thetas
from Unit_Tests.generate_parametrized_thetas import generate_parametrized_thetas

grid_size = [5, 5]
thetas = []
thetas.extend(generate_parametrized_thetas(size_obstacle=[1, 1], grid_size=grid_size))
thetas.extend(generate_parametrized_thetas(size_obstacle=[2, 2], grid_size=grid_size))
thetas.extend(generate_parametrized_thetas(size_obstacle=[3, 3], grid_size=grid_size))

discount = 0.9

final_value_param = 1
theta = final_value_param * thetas[40]
theta = -theta
theta[1, 4] = 1
theta = theta.astype('float')
print(theta)

# grid_size = [3, 3]
# list_binary_arrays = generate_thetas(grid_size)
# discount = 0.9
#
# theta = 1 * list_binary_arrays[2]
# theta = theta.astype('float')
# print(theta)
robot_state = [3, 2]
probability = human_model(1, theta, robot_state, discount, grid_size)

print(probability)