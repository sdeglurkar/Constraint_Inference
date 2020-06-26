import numpy as np
from Navigation import Navigation


grid_size = [4, 4]
obs_sizes = [[1, 1]]

true_env = np.array([[0, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 0]])

nav = Navigation(grid_size=grid_size, discount=0.9, robot_state=[0, 0], beta=10, robot_goal=[3, 3],
                 obs_sizes=obs_sizes, true_env=true_env)
nav.full_pipeline_qmdp()

# nav.update_robot_state([1, 2])
# nav.full_pipeline()
