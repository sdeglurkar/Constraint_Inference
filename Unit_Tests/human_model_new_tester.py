from Unit_Tests.human_model_new import human_model_new
from Unit_Tests.generate_parametrized_thetas import generate_parametrized_thetas

grid_size = [5, 5]
thetas = []
thetas.extend(generate_parametrized_thetas(size_obstacle=[1, 1], grid_size=grid_size))
thetas.extend(generate_parametrized_thetas(size_obstacle=[2, 2], grid_size=grid_size))
thetas.extend(generate_parametrized_thetas(size_obstacle=[3, 3], grid_size=grid_size))

discount = 0.9

final_value_param = 1
theta = final_value_param * thetas[26]
theta = theta.astype('float')
print(theta)
robot_state = [1, 2]
probability = human_model_new(2, theta, robot_state, discount, grid_size, final_value_param)

print(probability)