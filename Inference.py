import matplotlib.pyplot as plt
import numpy as np
import random
import time
from value_iteration import ValueIteration
from value_iteration_human_robot import ValueIterationHumanRobot

class Inference:

    def __init__(self, grid_size, discount, robot_state, beta, robot_goal):
        """
        :param grid_size: [num_rows, num_cols]
        :param discount: Discount factor for value iteration
        :param robot_state: Where is the robot currently
        :param beta: Human model "rationality" parameter
        """
        self.grid_size = grid_size
        self.discount = discount
        self.robot_state = robot_state
        self.beta = beta

        self.robot_goal = robot_goal
        if len(self.robot_goal) == 0:
            self.no_goal = True

        self.robot_policies = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "exit"]
        self.human_policies = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "none", "exit"]

        self.prior = []


    def generate_all_thetas(self):
        n = self.grid_size[0] * self.grid_size[1]
        max_number_represented = 2 ** n - 1
        numbers = range(max_number_represented + 1)
        list_binary_strings = [np.binary_repr(number, width=self.grid_size[0] * self.grid_size[1]) for number in numbers]
        list_binary_arrays = []

        for i in range(len(numbers)):
            list_binary_strings[i] = np.array([int(elem) for elem in list_binary_strings[i]])
            list_binary_arrays.append(np.reshape(list_binary_strings[i], [self.grid_size[0], self.grid_size[1]]))

        self.prior = np.ones(len(list_binary_arrays))/len(list_binary_arrays)

        return np.array(list_binary_arrays)


    def generate_parametrized_thetas(self, size_obstacle):
        if max(size_obstacle) >= min(self.grid_size[0], self.grid_size[1]):
            print("Obstacle cannot be larger than grid!")
            return
        obs = np.ones((size_obstacle[0], size_obstacle[1]))
        zeros = np.zeros((size_obstacle[0], self.grid_size[1] - size_obstacle[1]))
        block_row = np.hstack([obs, zeros])
        rolls = [block_row]
        for i in range((self.grid_size[1] + 1 - size_obstacle[1]) - 1):
            block_row = np.roll(block_row, 1)
            rolls.append(block_row)

        zeros = np.zeros((self.grid_size[0] - size_obstacle[0], self.grid_size[1]))
        preliminary_thetas = [np.vstack([roll, zeros]) for roll in rolls]
        thetas = []
        for i in range(len(preliminary_thetas)):
            theta = preliminary_thetas[i]
            if self.no_goal:
                thetas.append(np.copy(theta))
            else:
                if theta[self.robot_goal[0]][self.robot_goal[1]] == 0:
                    theta = -theta
                    theta[self.robot_goal[0]][self.robot_goal[1]] = 1
                    thetas.append(np.copy(theta))
                    theta[self.robot_goal[0]][self.robot_goal[1]] = 0
                    theta = -theta
            for j in range((self.grid_size[0] + 1 - size_obstacle[0]) - 1):
                theta = np.roll(theta, 1, axis=0)
                if self.no_goal:
                    thetas.append(np.copy(theta))
                else:
                    if theta[self.robot_goal[0]][self.robot_goal[1]] == 0:
                        theta = -theta
                        theta[self.robot_goal[0]][self.robot_goal[1]] = 1
                        thetas.append(np.copy(theta))
                        theta[self.robot_goal[0]][self.robot_goal[1]] = 0
                        theta = -theta

        if len(self.prior) == 0:
            self.prior = np.ones(len(thetas))/len(thetas)
        else:
            curr_len = len(self.prior)
            self.prior = np.ones(len(thetas) + curr_len) / (len(thetas) + curr_len)

        return thetas


    def human_model(self, policy_index, theta, final_value_param=10):
        """
        :param policy_index: Index of policy the human inputted according to self.policies
        :param theta: Potential occupancy grid
        :param final_value_param: Reward upon reaching goal state
        :return: Probability human gives the policy given theta, which is a softmax on
        the optimal policy given by value iteration
        """
        valiter = ValueIteration(self.grid_size)
        final_value = theta * final_value_param
        value, q_value, optimal_policies = valiter.value_iteration(final_value, self.discount)
        exp_q_vals = np.zeros(len(valiter.policies))
        for i in range(len(valiter.policies)):
            if self.no_goal:
                exp_q_vals[i] = np.exp(-self.beta * q_value[self.robot_state[0], self.robot_state[1], i])
            else:
                exp_q_vals[i] = np.exp(self.beta * q_value[self.robot_state[0], self.robot_state[1], i])

        sum_exp = 0
        for i in range(len(exp_q_vals)):
            if not np.isnan(exp_q_vals[i]):
                sum_exp += exp_q_vals[i]

        exp_q_vals /= sum_exp

        return exp_q_vals[policy_index]


    def exact_inference(self, policy_index):
        """
        The prior should be a list of probabilities in the same order
        that generate_thetas() generates thetas; e.g. if the thetas
        are generated from [0, 1, 2, 3] (2 grid cells), prior[1] should be
        the probability that the theta is [[0, 0], [0, 1]], and so on.
        """
        t0 = time.time()

        thetas = []
        #thetas.extend(self.generate_all_thetas())
        thetas.extend(self.generate_parametrized_thetas(size_obstacle=[1, 1]))
        #thetas.extend(self.generate_parametrized_thetas(size_obstacle=[1, 2]))
        #thetas.extend(self.generate_parametrized_thetas(size_obstacle=[2, 1]))
        thetas.extend(self.generate_parametrized_thetas(size_obstacle=[2, 2]))
        #thetas.extend(self.generate_parametrized_thetas(size_obstacle=[3, 3]))
        dstb = np.zeros(len(thetas))
        for i in range(len(thetas)):
            theta = thetas[i]
            theta = theta.astype('float')
            human_model_prob = self.human_model(policy_index, theta)
            dstb[i] = human_model_prob * self.prior[i]

        sum_dstb = 0
        for i in range(len(dstb)):
            if not np.isnan(dstb[i]):
                sum_dstb += dstb[i]

        dstb /= sum_dstb

        t1 = time.time()

        self.visualizations(dstb, thetas)

        return dstb, t1-t0


    def visualizations(self, dstb, thetas):
        dstb_states = np.zeros((self.grid_size[0], self.grid_size[1]))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                max_prob = 0
                for k in range(len(thetas)):
                    theta = thetas[k]
                    if self.no_goal:
                        val = 1
                    else:
                        val = -1
                    if theta[i][j] == val:
                        if dstb[k] > max_prob:
                            max_prob = dstb[k]
                dstb_states[i][j] = max_prob

        for i in range(len(dstb)):
            if self.no_goal:
                plt.imshow(thetas[i], cmap='hot')
            else:
                ind = np.where(thetas[i] == 1)
                theta_show = np.copy(thetas[i])
                theta_show[ind[0], ind[1]] = 0
                theta_show = -theta_show
                plt.imshow(theta_show, cmap='hot')

            plt.colorbar()
            plt.scatter(self.robot_state[1], self.robot_state[0], s=50, c='b')
            plt.title(dstb[i])
            plt.show()

        plt.bar(range(len(thetas)), dstb)
        plt.title("Belief over Theta")
        plt.ylabel("P(theta|u_H)")
        plt.show()

        plt.imshow(dstb_states, cmap='hot')
        plt.colorbar()
        plt.scatter(self.robot_state[1], self.robot_state[0], s=50, c='b')
        plt.show()

