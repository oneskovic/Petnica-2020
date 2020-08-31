from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import random
import os

import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import utils
from functools import cmp_to_key
from game_environment2D_GA import GameEnv
from plotting import Plotter

random_seed = 123
log_interval = 5

np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)
np_random_generator = np.random.default_rng(seed=random_seed)
plotter = Plotter()
episode_count = 0
def calculate_average_return(blue_coeffs, red_coeffs, polynomial_degree, display_moves = False, no_tests = 10):
    py_environment = GameEnv(blue_coeffs, red_coeffs, polynomial_degree)
    env = tf_py_environment.TFPyEnvironment(py_environment)
    if display_moves:
        global episode_count
        episode_path = 'results/GA-2/game-drawn/episode-'+str(episode_count)
        if not os.path.exists(episode_path):
            os.makedirs(episode_path)
        
        time_step = env.reset()
        picture_count = 0
        while not time_step.is_last():
            observation = time_step.observation.numpy()[0]
            plotter.plot_state(observation,episode_path+'/'+str(picture_count)+'.jpeg')
            picture_count += 1
            action = np.array(0, dtype=np.int32)
            time_step = env.step(action)
        
        picture_count = 0
        episode_count += 1

    total_return = 0.0
    for _ in range(no_tests):
        time_step = env.reset()
        episode_return = 0.0

        while not time_step.is_last():
          #print(time_step.observation.numpy())
          action = np.array(0, dtype=np.int32)
          time_step = env.step(action)
          episode_return += time_step.reward
        total_return += episode_return

    return total_return/no_tests

def get_random_array(size, min_value, max_value):
    return (max_value-min_value)*np_random_generator.random(size) + min_value

def generate_random_coefficients(no_organisms, no_coefficients):
    coeffs = []
    for _ in range(no_organisms):
        organism_coeffs = get_random_array(no_coefficients,-1000,1000)
        coeffs.append(organism_coeffs)
    return coeffs

def compare_organisms(organism1,organism2):
    if organism1.time_alive < organism2.time_alive:
        return 1
    elif organism1.time_alive > organism2.time_alive:
        return -1
    else:
        return 0

# Combines coefficients from organism1 and organism2
# Mutates the child and returns the coefficients
def combine_organisms(organism1,organism2,hparams):
    coefs1 = np.array(organism1.coefficients)
    coefs2 = np.array(organism2.coefficients)
    child_coefs = (coefs1+coefs2)/2.0 # Average the coefficients
    mutation_range = hparams['mutation_factor_range']

    mutation_vector = get_random_array(len(child_coefs),mutation_range[0],mutation_range[1])
    mutation_vector = np.multiply(child_coefs,mutation_vector) # Multiply element-wise    
    child_coefs += mutation_vector # Mutate

    return list(child_coefs)

def get_coeffs_from_best(organisms, count, hparams):
    ordered_organisms = sorted(organisms,key=cmp_to_key(compare_organisms))
    best_organisms = ordered_organisms[:hparams['no_best']]

    coeffs = []
    for _ in range(count - hparams['no_random']):
        positions = random.sample(range(0,len(best_organisms)),2)
        coeffs.append(combine_organisms(best_organisms[positions[0]],
                                        best_organisms[positions[1]],hparams))

    coeffs += generate_random_coefficients(hparams['no_random'],
                                           hparams['max_function_degree'])

    return coeffs

def plot_coefs(single_organism_coefs):
    x = 0.0
    x_values = []
    function_values = []
    def evaluate_function(x):
        value = 0.0
        x_pow = 1
        for coef in single_organism_coefs:
            value += coef * x_pow
            x_pow *= x
        return value
    step = 0.03
    while x < 10:
        function_values.append(evaluate_function(x))
        x_values.append(x)
        x += step
    plt.plot(x_values,function_values)
    
def try_hparams(hparams):

    max_degree = hparams['max_function_degree']
    no_blues = hparams['no_blue_organisms']
    no_reds = hparams['no_red_organisms']
    max_degree = hparams['max_function_degree']
    
    blue_coeffs = generate_random_coefficients(no_blues,max_degree)
    red_coeffs = generate_random_coefficients(no_reds,max_degree)

    returns = []
    for generation_number in range(hparams['no_generations']):
        py_environment = GameEnv(blue_coeffs,red_coeffs,max_degree)
        #utils.validate_py_environment(py_environment, episodes=5)
        env = tf_py_environment.TFPyEnvironment(py_environment)
        if generation_number % log_interval == 0:
            avg_return = calculate_average_return(blue_coeffs,red_coeffs,max_degree,True)
            returns.append(avg_return)
            print(avg_return)
        
        time_step = env.reset()
        while not time_step.is_last():
            #print(time_step.observation.numpy())
            action = np.array(0, dtype=np.int32)
            time_step = env.step(action)

        blue_organisms = py_environment.dead_blue_organisms
        blue_coeffs = get_coeffs_from_best(blue_organisms, no_blues, hparams)
        red_organisms = py_environment.dead_red_organisms
        red_coeffs = get_coeffs_from_best(red_organisms, no_reds, hparams)

    plt.plot(list(returns))
    plt.show()
    plt.clf()

    for single_organism_coefs in red_coeffs:
        plot_coefs(single_organism_coefs)
    plt.show()
    plt.cla()
    for single_organism_coefs in blue_coeffs:
        plot_coefs(single_organism_coefs)

    plt.show()

hparams = {
    'max_function_degree': 5, # Degree of polynomial used for function approximation
    'no_blue_organisms': 10, 
    'no_red_organisms': 10,
    'no_random': 3, # Number of random organisms inserted into each new generation
    'mutation_factor_range': [-0.1,0.1], # When mutating each coefficient will be multiplied by a random value in this range
    'no_best': 5, # Number of best organisms chosen for the next generation
    'no_generations': 1000
    }
try_hparams(hparams)
