import numpy as np
import time
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


    def move_qmdp(self, correction, dstb, thetas):
        policy = self.optimal_policy_qmdp(dstb, thetas)
        u_R = self.valiter.robot_policies[policy]
        u_H = self.valiter.human_policies[correction]
        next_state = self.valiter.dynamics(self.state, u_H, u_R)
        self.state = next_state


    def optimal_policy_qmdp(self, dstb, thetas):
        max_qb_val = []
        for i in range(len(self.valiter.robot_policies)):
            qb_val = 0
            for j in range(len(thetas)):
                theta = thetas[j].astype('float')
                theta *= self.final_value_param
                value, q_value, optimal_policies_human, optimal_policies_robot = \
                    self.valiter.value_iteration(theta, self.discount)
                if not np.isnan(q_value[self.state[0], self.state[1], 8, i]):
                    qb_val += dstb[j] * q_value[self.state[0], self.state[1], 8, i]
            max_qb_val.append(qb_val)

        return np.argmax(max_qb_val)



    def update_env(self, environment):
        self.environment = environment.astype('float')
        self.environment *= self.final_value_param
        self.environment[self.goal[0], self.goal[1]] = self.final_value_param
        self.value, self.q_value, self.optimal_policies_human, self.optimal_policies_robot = \
            self.valiter.value_iteration(self.environment, self.discount)


    def update_theta(self, policy_index, robot_action):
        t0 = time.time()

        theta = np.copy(self.environment)/self.final_value_param
        if policy_index != 8:
            #human_q_val = self.q_value[self.state[0], self.state[1], policy_index, robot_action]
            #robot_q_val = self.q_value[self.state[0], self.state[1], 8, robot_action]
            next_state = self.valiter.dynamics(self.state, "none", self.valiter.robot_policies[robot_action])
            theta[next_state[0]][next_state[1]] = -1

        t1 = time.time()

        return theta, t1-t0


    def update_goal(self, goal):
        self.environment[self.goal[0], self.goal[1]] = 0.0
        self.goal = goal
        self.environment[self.goal[0], self.goal[1]] = self.final_value_param
        self.value, self.q_value, self.optimal_policies_human, self.optimal_policies_robot = \
            self.valiter.value_iteration(self.environment, self.discount)

