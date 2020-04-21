import numpy as np
from Inference import Inference
import time

grid_size = [5, 5]
infer = Inference(grid_size, discount=0.9, robot_state=[1, 1], beta=5)

# n = grid_size[0] * grid_size[1]
# max_number_represented = 2 ** n - 1
# prior = np.ones(max_number_represented + 1)/(max_number_represented + 1) # Uniform prior

#print(infer.exact_inference(policy_index=3))
print(infer.exact_inference_thetas(policy_index=3))
#print(infer.sampling_inference(policy_index=0, num_samples=300, prior=prior))


# t0 = time.time()
# 100 * np.exp(3)
# a = np.array([1, 2, 3])
# t1 = time.time()
# print(t1-t0)
