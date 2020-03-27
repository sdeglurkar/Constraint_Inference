import numpy as np
import random
import time
from value_iteration import ValueIteration

class Inference:

    def __init__(self, grid_size, discount, robot_state, beta):
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

        self.policies = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "exit"]


    def generate_thetas(self):
        n = self.grid_size[0] * self.grid_size[1]
        max_number_represented = 2 ** n - 1
        numbers = range(max_number_represented + 1)
        list_binary_strings = [np.binary_repr(number, width=self.grid_size[0] * self.grid_size[1]) for number in numbers]
        list_binary_arrays = []

        for i in range(len(numbers)):
            list_binary_strings[i] = np.array([int(elem) for elem in list_binary_strings[i]])
            list_binary_arrays.append(np.reshape(list_binary_strings[i], [self.grid_size[0], self.grid_size[1]]))

        return np.array(list_binary_arrays)


    def sample_thetas(self, prior):
        # Sample a random theta from the prior
        numbers = range(len(prior))
        theta_index = np.random.choice(a=numbers, size=1, p=prior)[0]
        # prob = random.random()
        # theta_index = len(prior) - 1
        # sum = 0
        # for j in range(len(prior)):
        #     sum += prior[j]
        #     if prob <= sum:
        #         theta_index = j
        #         break
        theta = np.binary_repr(theta_index, width=self.grid_size[0] * self.grid_size[1])
        theta = np.array([int(elem) for elem in theta])
        theta = np.reshape(theta, [self.grid_size[0], self.grid_size[1]])

        return theta, theta_index


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
            exp_q_vals[i] = np.exp(-self.beta * q_value[self.robot_state[0], self.robot_state[1], i])

        sum_exp = 0
        for i in range(len(exp_q_vals)):
            if not np.isnan(exp_q_vals[i]):
                sum_exp += exp_q_vals[i]

        exp_q_vals /= sum_exp

        return exp_q_vals[policy_index]


    # def state_given_theta(self, theta, state_row, state_col):
    #     if theta[state_row][state_col] == 0:
    #         return 0
    #     else:
    #         nonzero_theta = theta[theta != 0]
    #         cardinality = len(nonzero_theta)
    #         return 1/cardinality

    def state_given_theta(self, theta, state_row, state_col, binary_val):
        if theta[state_row][state_col] == binary_val:
            return 1
        else:
            return 0


    # def exact_inference(self, policy_index, prior):
    #     """
    #     The prior should be a list of probabilities in the same order
    #     that generate_thetas() generates thetas; e.g. if the thetas
    #     are generated from [0, 1, 2, 3] (2 grid cells), prior[1] should be
    #     the probability that the theta is [[0, 0], [0, 1]], and so on.
    #     """
    #     t0 = time.time()
    #
    #     thetas = self.generate_thetas()
    #     dstb = np.zeros((self.grid_size[0], self.grid_size[1]))
    #     for i in range(self.grid_size[0]):
    #         for j in range(self.grid_size[1]):
    #             for k in range(len(thetas)):
    #                 theta = thetas[k]
    #                 state_row = i
    #                 state_col = j
    #                 numerator = self.state_given_theta(theta, state_row, state_col) * \
    #                             self.human_model(policy_index, theta) * prior[k]
    #                 if not np.isnan(numerator):
    #                     dstb[state_row][state_col] += numerator
    #
    #     dstb /= sum(sum(dstb))
    #
    #     t1 = time.time()
    #
    #     return dstb, t1-t0

    def exact_inference(self, policy_index, prior):
        """
        The prior should be a list of probabilities in the same order
        that generate_thetas() generates thetas; e.g. if the thetas
        are generated from [0, 1, 2, 3] (2 grid cells), prior[1] should be
        the probability that the theta is [[0, 0], [0, 1]], and so on.
        """
        t0 = time.time()

        thetas = self.generate_thetas()
        dstb_0 = np.zeros((self.grid_size[0], self.grid_size[1]))
        dstb_1 = np.zeros((self.grid_size[0], self.grid_size[1]))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(len(thetas)):
                    theta = thetas[k]
                    state_row = i
                    state_col = j
                    human_model_prob = self.human_model(policy_index, theta)
                    numerator_1 = self.state_given_theta(theta, state_row, state_col, 1) * \
                                human_model_prob * prior[k]
                    numerator_0 = self.state_given_theta(theta, state_row, state_col, 0) * \
                                  human_model_prob * prior[k]
                    if not np.isnan(numerator_1):
                        dstb_1[state_row][state_col] += numerator_1
                    if not np.isnan(numerator_0):
                        dstb_0[state_row][state_col] += numerator_0

        denom = dstb_1 + dstb_0
        dstb = dstb_1/denom

        t1 = time.time()

        return dstb, t1-t0


    # def sampling_inference(self, policy_index, num_samples, prior):
    #     """
    #     This function performs likelihood weighting.
    #
    #     The prior should be a list of probabilities in the same order
    #     that generate_thetas() generates thetas; e.g. if the thetas
    #     are generated from [0, 1, 2, 3] (2 grid cells), prior[1] should be
    #     the probability that the theta is [[0, 0], [0, 1]], and so on.
    #     """
    #     t0 = time.time()
    #
    #     dstb = np.zeros((self.grid_size[0], self.grid_size[1]))
    #     samples_list = np.zeros((num_samples, 3)) # Each row is a sample of (theta, state_row, state_col)
    #     weights = np.zeros(num_samples)
    #
    #     # Generate samples and weights
    #     for i in range(num_samples):
    #         theta, theta_index = self.sample_thetas(prior)
    #         [row, col] = np.nonzero(theta)
    #         if len(row) > 0:
    #             samples_list[i, 0] = theta_index
    #             # Sample from state_given_theta (uniform)
    #             rand_index = random.randint(0, len(row) - 1)
    #             samples_list[i, 1] = row[rand_index]
    #             samples_list[i, 2] = col[rand_index]
    #             weights[i] = self.human_model(policy_index, theta)
    #         else:
    #             # This sample will just have 0 probability
    #             samples_list[i, 0] = theta_index
    #             samples_list[i, 1] = samples_list[i, 2] = -1
    #
    #     # Weighted sum of samples
    #     for i in range(self.grid_size[0]):
    #         for j in range(self.grid_size[1]):
    #             for k in range(num_samples):
    #                 if samples_list[k, 1] == i and samples_list[k, 2] == j:
    #                     if not np.isnan(weights[k]):
    #                         dstb[i][j] += weights[k]
    #
    #
    #     dstb /= sum(sum(dstb))
    #
    #     t1 = time.time()
    #
    #     return dstb, t1-t0


    def sampling_inference(self, policy_index, num_samples, prior):
        """
        This function performs likelihood weighting.

        The prior should be a list of probabilities in the same order
        that generate_thetas() generates thetas; e.g. if the thetas
        are generated from [0, 1, 2, 3] (2 grid cells), prior[1] should be
        the probability that the theta is [[0, 0], [0, 1]], and so on.
        """
        t0 = time.time()

        dstb = np.zeros((self.grid_size[0], self.grid_size[1]))
        dstb_denom = np.zeros((self.grid_size[0], self.grid_size[1]))
        samples_list = np.zeros((num_samples, self.grid_size[0], self.grid_size[1])) # Each sample is a theta
        weights = np.zeros(num_samples)

        # Generate samples and weights
        for i in range(num_samples):
            theta, theta_index = self.sample_thetas(prior)
            samples_list[i] = theta
            weights[i] = self.human_model(policy_index, theta)

        # Weighted sum of samples
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(num_samples):
                    theta = samples_list[k]
                    if theta[i][j] == 1 and not np.isnan(weights[k]):
                            dstb[i][j] += weights[k]
                            dstb_denom[i][j] += weights[k]
                    if theta[i][j] == 0 and not np.isnan(weights[k]):
                            dstb_denom[i][j] += weights[k]


        dstb /= dstb_denom

        t1 = time.time()

        return dstb, t1-t0


