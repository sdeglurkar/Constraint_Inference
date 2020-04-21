import matplotlib.pyplot as plt
import numpy as np
from value_iteration import ValueIteration
from Unit_Tests.generate_parametrized_thetas import generate_parametrized_thetas

class HeuristicDistributions:

    def __init__(self, grid_size):
        self.policies = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "exit"]
        self.opposites = {"north":"south", "south":"north", "east":"west", "west":"east", "northeast":"southwest", "southwest":"northeast", "northwest":"southeast", "southeast":"northwest"}
        self.grid_size = grid_size
        self.dstb = np.zeros((self.grid_size[0], self.grid_size[1]))

    def generate_distribution(self, robot_state, correction):
        dstb = np.zeros((self.grid_size[0], self.grid_size[1]))
        thetas_1 = generate_parametrized_thetas([1, 1], self.grid_size)
        thetas_2 = generate_parametrized_thetas([2, 2], self.grid_size)
        thetas_3 = generate_parametrized_thetas([3, 3], self.grid_size)
        thetas_4 = generate_parametrized_thetas([4, 4], self.grid_size)
        valiter = ValueIteration(self.grid_size)
        opposite_dir = self.opposites[self.policies[correction]]
        neighboring_state = valiter.dynamics(robot_state, opposite_dir)
        if neighboring_state != 0:
            row = neighboring_state[0]
            col = neighboring_state[1]
            for theta in thetas_1:
                if theta[row, col] == 1 and theta[robot_state[0], robot_state[1]] != 1:
                    [obs_rows, obs_cols] = np.nonzero(theta)
                    for i in range(len(obs_rows)):
                        dstb[obs_rows[i], obs_cols[i]] = 0.5
            for theta in thetas_2:
                if theta[row, col] == 1 and theta[robot_state[0], robot_state[1]] != 1:
                    [obs_rows, obs_cols] = np.nonzero(theta)
                    for i in range(len(obs_rows)):
                        if dstb[obs_rows[i], obs_cols[i]] == 0:
                            dstb[obs_rows[i], obs_cols[i]] = 0.3
            for theta in thetas_3:
                if theta[row, col] == 1 and theta[robot_state[0], robot_state[1]] != 1:
                    [obs_rows, obs_cols] = np.nonzero(theta)
                    for i in range(len(obs_rows)):
                        if dstb[obs_rows[i], obs_cols[i]] == 0:
                            dstb[obs_rows[i], obs_cols[i]] = 0.2
            for theta in thetas_4:
                if theta[row, col] == 1 and theta[robot_state[0], robot_state[1]] != 1:
                    [obs_rows, obs_cols] = np.nonzero(theta)
                    for i in range(len(obs_rows)):
                        if dstb[obs_rows[i], obs_cols[i]] == 0:
                            dstb[obs_rows[i], obs_cols[i]] = 0.1


        self.dstb += dstb

        plt.imshow(self.dstb, cmap='hot')
        plt.colorbar()
        plt.scatter(robot_state[1], robot_state[0], s=50, c='b')
        plt.show()



grid_size = [20, 20]
hdstbs = HeuristicDistributions(grid_size)

# robot_state = [15, 0]
# correction = 3
# hdstbs.generate_distribution(robot_state, correction)

robot_state = [3, 3]
correction = 0
hdstbs.generate_distribution(robot_state, correction)
robot_state = [5, 7]
correction = 4
hdstbs.generate_distribution(robot_state, correction)
robot_state = [8, 10]
correction = 2
hdstbs.generate_distribution(robot_state, correction)
robot_state = [12, 7]
correction = 1
hdstbs.generate_distribution(robot_state, correction)