import numpy as np
import copy

'''
Value iteration for an MDP with deterministic dynamics
'''
class ValueIteration:

    def __init__(self, grid_size):
        #self.policies = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "exit"]
        self.policies = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "none",
                         "exit"]
        self.grid_size = grid_size


    def value_iteration(self, final_value, discount):
        q_value = np.zeros((self.grid_size[0], self.grid_size[1], len(self.policies)))
        optimal_policies = np.zeros((self.grid_size[0], self.grid_size[1]))
        old_value = final_value

        if np.shape(final_value)[0] != self.grid_size[0] or np.shape(final_value)[1] != self.grid_size[1]:
            print("Dimensions of final value do not match grid size!")
            return
        
        cont = True
        while cont:
            new_value = copy.deepcopy(old_value)
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    for k in range(len(self.policies)):
                        # Checks for goal/obstacle states
                        if self.policies[k] is "exit":
                            if final_value[i][j] != 0:
                                r = old_value[i][j]
                                q_value[i][j][k] = r
                                continue
                        else:
                            if final_value[i][j] != 0:
                                q_value[i][j][k] = np.nan
                                continue
                        
                        # Non-goal or obstacle states
                        x = [i, j]
                        u = self.policies[k]
                        x_prime = self.dynamics(x, u)
                        if x_prime == 0:
                            q_value[i][j][k] = np.nan
                            continue
                        r = self.reward(x, u, x_prime, final_value)
                        # Here is the q-value iteration update
                        q_value[i][j][k] = r + discount * old_value[x_prime[0]][x_prime[1]]
                
            # Compute value from q-value
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    max_q_value = -np.inf  
                    for k in range(len(self.policies)):
                        if not np.isnan(q_value[i][j][k]) and q_value[i][j][k] > max_q_value:
                            max_q_value = q_value[i][j][k]
                            optimal_policies[i][j] = k
                        
                    new_value[i][j] = max_q_value
                
            # Check convergence
            norm = np.linalg.norm(new_value - old_value)
            if norm > 0.0001:
                cont = True
                old_value = new_value
            else:
                cont = False


        value = new_value      

        return value, q_value, optimal_policies

    
    def reward(self, x, u, x_prime, final_value):
        r = 0

        if u is not "none" and u is not "exit":
            if u is "northeast" or \
                    u is "northwest" or \
                    u is "southeast" or \
                    u is "southwest":

                r = -np.sqrt(2)

            else:
                r = -1

        return r


    def dynamics(self, x, u):
        x_prime = [0, 0]
        if u is "north":
            x_prime = [x[0] - 1, x[1]]
        elif u is "south":
            x_prime = [x[0] + 1, x[1]]
        elif u is "east":
            x_prime = [x[0], x[1] + 1]
        elif u is "west":
            x_prime = [x[0], x[1] - 1]
        elif u is "northeast":
            x_prime = [x[0] - 1, x[1] + 1]
        elif u is "northwest":
            x_prime = [x[0] - 1, x[1] - 1]
        elif u is "southeast":
            x_prime = [x[0] + 1, x[1] + 1]
        elif u is "southwest":
            x_prime = [x[0] + 1, x[1] - 1]
        elif u is "exit":
            x_prime = 0
        elif u is "none":
            x_prime = x

        if x_prime != 0 and \
            (x_prime[0] < 0 or x_prime[1] < 0 or x_prime[0] >= self.grid_size[0] or x_prime[1] >= self.grid_size[1]):
            x_prime = 0
        

        return x_prime