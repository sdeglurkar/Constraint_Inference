import numpy as np

def generate_parametrized_thetas(size_obstacle, grid_size):
    if max(size_obstacle) >= min(grid_size[0], grid_size[1]):
        print("Obstacle cannot be larger than grid!")
        return
    obs = np.ones((size_obstacle[0], size_obstacle[1]))
    zeros = np.zeros((size_obstacle[0], grid_size[1] - size_obstacle[1]))
    block_row = np.hstack([obs, zeros])
    rolls = [block_row]
    for i in range((grid_size[1] + 1 - size_obstacle[1]) - 1):
        block_row = np.roll(block_row, 1)
        rolls.append(block_row)

    zeros = np.zeros((grid_size[0] - size_obstacle[0], grid_size[1]))
    preliminary_thetas = [np.vstack([roll, zeros]) for roll in rolls]
    thetas = []
    for i in range(len(preliminary_thetas)):
        theta = preliminary_thetas[i]
        thetas.append(theta)
        for j in range((grid_size[0] + 1 - size_obstacle[0]) - 1):
            theta = np.roll(theta, 1, axis=0)
            thetas.append(theta)

    return thetas
