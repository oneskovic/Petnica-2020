# Provides simple functions to be used with the genetic algorithm
import numpy as np
from random_util import RandomUtil
from functools import cmp_to_key

class GaUtil():
    rand_util_instance = RandomUtil(0)
    coefficients_count = 0
    def __init__(self, random_util, coefficients_count):
        self.rand_util_instance = random_util
        self.coefficients_count = coefficients_count
    # Combines coefficients from organism1 and organism2
    # Mutates the child and returns the coefficients
    def combine_organisms(self,organism1,organism2,mutation_factor_range):
        coefs1 = np.array(organism1.coefficients)
        coefs2 = np.array(organism2.coefficients)
        child_coefs = (coefs1+coefs2)/2.0 # Average the coefficients

        mutation_vector = self.rand_util_instance.get_random_array(len(child_coefs),
                                                mutation_factor_range[0],mutation_factor_range[1])
        mutation_vector = np.multiply(child_coefs,mutation_vector) # Multiply element-wise    
        child_coefs += mutation_vector # Mutate

        return list(child_coefs)
    
    
    def compare_organisms(self,organism1,organism2):
        if organism1.time_alive < organism2.time_alive:
            return 1
        elif organism1.time_alive > organism2.time_alive:
            return -1
        else:
            return 0
    
    # Returns coefficients gotten from combining random pairs of organisms
    # count - number of coefficients to return
    def get_coeffs_from_best(self, organisms, total_count, no_best, no_random, mutation_factor_range):
        ordered_organisms = sorted(organisms,key=cmp_to_key(self.compare_organisms))
        best_organisms = ordered_organisms[:no_best]

        coeffs = []
        for _ in range(total_count - no_random):
            positions = self.rand_util_instance.get_random_array_discrete(2,0,len(best_organisms))
            coeffs.append(self.combine_organisms(best_organisms[positions[0]],
                                            best_organisms[positions[1]],mutation_factor_range))

        coeffs += self.rand_util_instance.get_random_matrix(
            no_random, self.coefficients_count, [-1000,1000])

        return coeffs
