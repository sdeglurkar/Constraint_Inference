import matplotlib.pyplot as plt
import numpy as np
from Human import Human
from Inference import Inference
from Robot import Robot

class Navigation:

    def __init__(self, grid_size, discount, robot_state, beta, robot_goal, obs_sizes, true_env, robot_env):
        self.human = Human(true_env, beta, discount, robot_goal, grid_size)
        self.robot = Robot(robot_env, robot_state, robot_goal, grid_size, discount)

        robot_action = self.robot.optimal_policy()
        self.infer = Inference(grid_size, discount, self.robot.state, beta, robot_goal, robot_action,
                          obs_sizes)


    def full_pipeline(self):
        while self.robot.state != self.robot.goal:
            # Get the robot's next action as predicted by the human
            robot_action = self.robot.optimal_policy()
            self.infer.robot_action = robot_action

            # Human gives correction
            policy_index = self.human.give_correction(self.robot.state, robot_action)
            if policy_index != 8:
                print("Human gave a correction! The correction was", policy_index, ", the robot state was",
                      self.robot.state, ", and the robot action was", robot_action)
            else:
                print("Human gave no correction. The robot state was",
                      self.robot.state, ", and the robot action was", robot_action)
            dstb, time = self.infer.exact_inference(policy_index)
            print("Inference took time", time)
            sorted_ind = np.argsort(dstb)
            highest_theta_ind = sorted_ind[-1]
            highest_theta = self.infer.thetas[highest_theta_ind]

            # The robot's new environment is the MAP of the inference
            self.robot.update_env(highest_theta)

            # Robot "replans"
            self.robot.move(policy_index)
            self.infer.robot_state = self.robot.state


    def update_goal(self, goal):
        self.human.update_goal(goal)
        self.robot.update_goal(goal)
        self.infer.update_thetas(goal)


    def update_robot_state(self, state):
        self.robot.state = state
        self.infer.robot_state = state







