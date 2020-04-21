import numpy as np
from value_iteration import ValueIteration

def human_model_new(policy_index, theta, robot_state, discount, grid_size, final_value_param):
    valiter = ValueIteration(grid_size)
    value, q_value, optimal_policies = valiter.value_iteration(theta, discount)
    exp_q_vals = np.zeros(len(valiter.policies))
    for i in range(len(valiter.policies)):
        robot_val = value[robot_state[0], robot_state[1]]
        if robot_val < discount * final_value_param:
            robot_q_val = 0
        else:
            robot_q_val = q_value[robot_state[0], robot_state[1], i]
        exp_q_vals[i] = np.exp(-robot_q_val)

    sum_exp = 0
    for i in range(len(exp_q_vals)):
        if not np.isnan(exp_q_vals[i]):
            sum_exp += exp_q_vals[i]

    exp_q_vals /= sum_exp

    return exp_q_vals[policy_index]
