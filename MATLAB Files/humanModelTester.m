grid_size = 2;
policies = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "exit"];
list_binary_arrays = generateThetas(grid_size)
discount = 0.9;

theta = list_binary_arrays(:, :, 5);
robot_state = [2, 1];
probability = humanModel(9, theta, robot_state, policies, discount)












