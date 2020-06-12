import numpy as np
from value_iteration_human_robot import ValueIterationHumanRobot

class Robot:

    def __init__(self, environment, state, goal, grid_size, discount, final_value_param=10):
        self.environment = environment.astype('float')
        self.state = state
        self.goal = goal
        self.grid_size = grid_size
        self.discount = discount
        self.final_value_param = final_value_param

        self.environment *= final_value_param
        self.environment[goal[0], goal[1]] = self.final_value_param

        self.valiter = ValueIterationHumanRobot(self.grid_size)
        self.value, self.q_value, self.optimal_policies_human, self.optimal_policies_robot = \
            self.valiter.value_iteration(self.environment, self.discount)


    def optimal_policy(self):
        x = self.state[0]
        y = self.state[1]
        policy = int(self.optimal_policies_robot[x][y])
        return policy


    def move(self, correction):
        policy = self.optimal_policy()
        u_R = self.valiter.robot_policies[policy]
        u_H = self.valiter.human_policies[correction]
        next_state = self.valiter.dynamics(self.state, u_H, u_R)
        self.state = next_state


    def update_env(self, environment):
        self.environment = environment.astype('float')
        self.environment *= self.final_value_param
        self.environment[self.goal[0], self.goal[1]] = self.final_value_param
        self.value, self.q_value, self.optimal_policies_human, self.optimal_policies_robot = \
            self.valiter.value_iteration(self.environment, self.discount)


    def update_goal(self, goal):
        self.environment[self.goal[0], self.goal[1]] = 0.0
        self.goal = goal
        self.environment[self.goal[0], self.goal[1]] = self.final_value_param
        self.value, self.q_value, self.optimal_policies_human, self.optimal_policies_robot = \
            self.valiter.value_iteration(self.environment, self.discount)