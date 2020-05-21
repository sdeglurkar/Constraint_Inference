import numpy as np
from Inference import Inference
import time

grid_size = [5, 5]
infer = Inference(grid_size, discount=0.9, robot_state=[1, 1], beta=10, robot_goal=[])

print(infer.exact_inference(policy_index=3))

