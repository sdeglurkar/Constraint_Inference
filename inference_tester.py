import numpy as np
from Inference import Inference
import time

grid_size = [3, 3]
infer = Inference(grid_size, discount=0.9, robot_state=[2, 0], beta=70)
n = grid_size[0] * grid_size[1]
max_number_represented = 2 ** n - 1
prior = np.ones(max_number_represented + 1)/(max_number_represented + 1) # Uniform prior
print(infer.exact_inference(policy_index=0, prior=prior))
print(infer.sampling_inference(policy_index=0, num_samples=300, prior=prior))


# t0 = time.time()
# 100 * np.exp(3)
# a = np.array([1, 2, 3])
# t1 = time.time()
# print(t1-t0)
