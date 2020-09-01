from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time

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
from random_util import RandomUtil
from ga_util import GaUtil

random_seed = round(time.time())
log_interval = 5

np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)
np_random_generator = np.random.default_rng(seed=random_seed)

random_util = RandomUtil(random_seed)
base_path = 'results/GA-'+random_util.generate_random_string(6)+'/'
plotter = Plotter(base_path)
episode_count = 0

def evaluate_ga(blue_coeffs, red_coeffs, eval_py_env , display_moves = False, no_tests = 10):
    env = tf_py_environment.TFPyEnvironment(eval_py_env)
    if display_moves:
        global episode_count
        episode_path = 'game-drawn/episode-'+str(episode_count)
        if not os.path.exists(base_path + episode_path):
            os.makedirs(base_path + episode_path)
        
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

def plot_coefs(single_organism_coefs, file_name):
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
    
    plotter.plot_simple_values(x=x_values,y=function_values,directory=file_name)

def try_hparams(hparams, game_params):
    max_degree = hparams['max_parameter_degree']
    no_blues = hparams['no_blue_organisms']
    no_reds = hparams['no_red_organisms']
    no_parameters = hparams['no_parameters']
    coef_count = np.power(max_degree+1,no_parameters)
    ga_util = GaUtil(random_util,coef_count)
    
    blue_coeffs = random_util.get_random_matrix(no_blues,coef_count,[-1000,1000])
    red_coeffs = random_util.get_random_matrix(no_reds,coef_count,[-1000,1000])

    returns = []
    # Train the genetic algorithm
    no_random = hparams['no_random_start'] * 1.0
    random_step = (hparams['no_random_final']-no_random)/hparams['no_random_anneal_time']
    mutation_factor_range = np.array(hparams['mutation_factor_range_start'])
    mutation_factor_range_final = np.array(hparams['mutation_factor_range_final'])
    mutation_factor_range_step = (mutation_factor_range_final - mutation_factor_range)\
                                 / hparams['mutation_factor_range_anneal_time']
    
    for generation_number in range(hparams['no_generations']):
        prev_blue_organisms = []
        prev_red_organisms = []
        if generation_number > 0:
            prev_blue_organisms = py_environment.dead_blue_organisms
            prev_red_organisms =  py_environment.dead_red_organisms 
            
        py_environment = GameEnv(blue_coeffs,red_coeffs,max_degree,hparams['food_count'],hparams['board_size'])
        #utils.validate_py_environment(py_environment, episodes=5)
        env = tf_py_environment.TFPyEnvironment(py_environment)
        
        if (generation_number+1) % log_interval == 0:
            eval_blue_coeffs = ga_util.get_coeffs_from_best(prev_blue_organisms, game_params['no_blue_organisms'], game_params['no_blue_organisms'], 0, [0,0])
            eval_red_coeffs = ga_util.get_coeffs_from_best(prev_red_organisms, game_params['no_red_organisms'], game_params['no_red_organisms'], 0, [0,0])
            eval_py_env = GameEnv(eval_blue_coeffs,eval_red_coeffs,max_degree,game_params['food_count'],game_params['board_size'])
            
            avg_return = evaluate_ga(blue_coeffs,red_coeffs,eval_py_env,True)
            returns.append(avg_return)
            print(avg_return)
        
        # Play the game
        time_step = env.reset()
        while not time_step.is_last():
            #print(time_step.observation.numpy())
            action = np.array(0, dtype=np.int32)
            time_step = env.step(action)

        # Pick best genomes for the next generation
        blue_organisms = py_environment.dead_blue_organisms
        blue_coeffs = ga_util.get_coeffs_from_best(blue_organisms, no_blues, hparams['no_best'], round(no_random), mutation_factor_range)
        red_organisms = py_environment.dead_red_organisms
        red_coeffs = ga_util.get_coeffs_from_best(red_organisms, no_reds, hparams['no_best'], round(no_random), mutation_factor_range)
        
        # Reduce the number of random organisms and mutation_factor_range
        no_random += random_step
        mutation_factor_range += mutation_factor_range_step

    plotter.plot_simple_values(y=list(returns),directory='score.jpeg')
    for single_organism_coefs in red_coeffs:
        plot_coefs(single_organism_coefs,'red-coeffs.jpeg')
    for single_organism_coefs in blue_coeffs:
        plot_coefs(single_organism_coefs,'blue-coeffs.jpeg')
    hparams['random_seed'] = random_seed
    plotter.dump_to_json(hparams,'hparams.json')

hparams = {
    'max_parameter_degree': 10, # Degree of polynomial used for function approximation
    'no_parameters': 1,
    'no_blue_organisms': 30, 
    'no_red_organisms': 30,
    'food_count': 30,
    'board_size': 30,
    'no_random_start': 5, # Number of random organisms inserted into each new generation
    'no_random_final': 0,
    'no_random_anneal_time':40, # Number of generations to anneal to final value
    'mutation_factor_range_start': [-0.1,0.1], # When mutating each coefficient will be multiplied by a random value in this range
    'mutation_factor_range_final': [0,0],
    'mutation_factor_range_anneal_time': 40,
    'no_best': 5, # Number of best organisms chosen for the next generation
    'no_generations': 15
    }
game_params = {
    'no_red_organisms': 5,
    'no_blue_organisms': 3,
    'board_size': 10,
    'food_count': 10
}
try_hparams(hparams,game_params)
