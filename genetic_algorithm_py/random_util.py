# Provides functions for generating various random numbers
import numpy as np
import random
import string

class RandomUtil():
    np_random_generator = np.random.default_rng()
    def __init__(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.np_random_generator = np.random.default_rng(seed=seed)
    
    # Returns an array of given size containing uniform random numbers
    # In range [min_value,max_value)
    def get_random_array(self, size, min_value, max_value):
        return (max_value-min_value)*self.np_random_generator.random(size) + min_value
    def get_random_array_discrete(self, size, min_value, max_value):
        return random.choices(range(min_value, max_value),k=size)

    def get_random_matrix(self, no_rows, no_columns, element_range):
        rows = []
        for _ in range(no_rows):
            rand_row = self.get_random_array(no_columns,element_range[0],element_range[1])
            rows.append(rand_row)
        return rows
    
    def generate_random_string(self, length):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
