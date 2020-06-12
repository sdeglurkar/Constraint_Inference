import numpy as np
from Inference import Inference
import time

grid_size = [5, 5]
obs_sizes = [[1, 1], [2, 2]]

infer = Inference(grid_size, discount=0.9, robot_state=[1, 1], beta=10, robot_goal=[2, 4], robot_action=2, obs_sizes=obs_sizes)

infer.exact_inference(policy_index=0)

# infer = Inference(grid_size, discount=0.9, robot_state=[4, 0], beta=10, robot_goal=[2, 4], robot_action=0, obs_sizes=obs_sizes)
#
# infer.exact_inference(policy_index=2)
#
# infer.robot_state = [4, 1]
# infer.robot_action = 2
#
# infer.exact_inference(policy_index=8)
#
# infer.robot_state = [4, 2]
# infer.robot_action = 2
#
# infer.exact_inference(policy_index=8)
#
# infer.robot_state = [4, 3]
# infer.robot_action = 2
#
# infer.exact_inference(policy_index=8)
