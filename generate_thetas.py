import numpy as np 

def generate_thetas(grid_size):
    n = grid_size[0] * grid_size[1]
    max_number_represented = 2**n - 1
    numbers = range(max_number_represented + 1)
    list_binary_strings = [np.binary_repr(number, width = grid_size[0]*grid_size[1]) for number in numbers]
    list_binary_arrays = []

    for i in range(len(numbers)):
        list_binary_strings[i] = np.array([int(elem) for elem in list_binary_strings[i]])
        list_binary_arrays.append(np.reshape(list_binary_strings[i], [grid_size[0], grid_size[1]]))
    
    return list_binary_arrays
