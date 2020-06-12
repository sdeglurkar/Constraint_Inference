import numpy as np
import random
from value_iteration_human_robot import ValueIterationHumanRobot

class Human:

    def __init__(self, environment, beta, discount, robot_goal, grid_size, final_value_param=10):
        self.environment = environment.astype('float')
        self.beta = beta
        self.discount = discount
        self.robot_goal = robot_goal
        self.grid_size = grid_size
        self.final_value_param = final_value_param

        self.environment *= final_value_param
        self.environment[robot_goal[0], robot_goal[1]] = self.final_value_param

        self.valiter = ValueIterationHumanRobot(self.grid_size)
        self.value, self.q_value, self.optimal_policies_human, self.optimal_policies_robot = \
            self.valiter.value_iteration(self.environment, self.discount)


    def give_correction(self, robot_state, robot_action):
        exp_q_vals = np.zeros(len(self.valiter.human_policies))
        for i in range(len(self.valiter.human_policies)):
            exp_q_vals[i] = np.exp(self.beta * self.q_value[robot_state[0], robot_state[1], i, robot_action])

        sum_exp = 0
        for i in range(len(exp_q_vals)):
            if not np.isnan(exp_q_vals[i]):
                sum_exp += exp_q_vals[i]

        exp_q_vals /= sum_exp

        non_nan = np.where(np.isnan(exp_q_vals) == False)
        non_nan = non_nan[0]

        cum_probabilities = []
        cum_sum = 0

        for index in non_nan:
            cum_sum += exp_q_vals[index]
            cum_probabilities.append(cum_sum)

        rand_num = random.random()
        for i in range(len(cum_probabilities)):
            if rand_num < cum_probabilities[i]:
                return non_nan[i]


    def update_goal(self, goal):
        self.environment[self.robot_goal[0], self.robot_goal[1]] = 0.0
        self.robot_goal = goal
        self.environment[self.robot_goal[0], self.robot_goal[1]] = self.final_value_param
        self.value, self.q_value, self.optimal_policies_human, self.optimal_policies_robot = \
            self.valiter.value_iteration(self.environment, self.discount)