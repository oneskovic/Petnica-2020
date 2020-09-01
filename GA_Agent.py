from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time

from functools import cmp_to_key
from game_environment2D_GA import GameEnv
from plotting import Plotter
from random_util import RandomUtil
from ga_util import GaUtil

class GA_Agent:
    hparams = {}
    random_seed = 0
    eval_interval = -1
    display_moves = False
    should_log = False
    random_util = RandomUtil(0)
    plotter = Plotter('none')
    base_path = 'none'
    episode_count = 0
    def __init__(self,hyperparameters,should_log = False,eval_interval=-1,display_moves = False):
        self.hparams = hyperparameters
        self.random_seed = round(time.time())
        self.should_log = should_log
        self.display_moves = display_moves
        self.eval_interval = eval_interval

        self.random_util = RandomUtil(self.random_seed)
        self.base_path = 'results/GA-'+self.random_util.generate_random_string(6)+'/'
        self.plotter = Plotter(self.base_path)
        self.episode_count = 0

    
    def __evaluate_ga(self,blue_coeffs, red_coeffs, eval_py_env , display_moves = False, no_tests = 10):
        if display_moves:
            episode_path = 'game-drawn/episode-'+str(self.episode_count)
            if not os.path.exists(self.base_path + episode_path):
                os.makedirs(self.base_path + episode_path)
            
            time_step = eval_py_env.reset()
            picture_count = 0
            while not time_step.is_last():
                observation = time_step.observation
                self.plotter.plot_state(observation,episode_path+'/'+str(picture_count)+'.jpeg')
                picture_count += 1
                time_step = eval_py_env.step()
            
            picture_count = 0
            self.episode_count += 1

        total_return = 0.0
        for _ in range(no_tests):
            time_step = eval_py_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                #print(time_step.observation.numpy())
                time_step = eval_py_env.step()
                episode_return += time_step.reward
            total_return += episode_return

        return total_return/no_tests

    def __plot_coefs(self,single_organism_coefs, file_name):
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
        
        self.plotter.plot_simple_values(x=x_values,y=function_values,directory=file_name)

    def train(self, eval_game_params):
            hparams = self.hparams
            max_degree = hparams['max_parameter_degree']
            no_blues = hparams['no_blue_organisms']
            no_reds = hparams['no_red_organisms']
            no_parameters = hparams['no_parameters']
            coef_count = np.power(max_degree+1,no_parameters)
            self.ga_util = GaUtil(self.random_util,coef_count)
            
            blue_coeffs = self.random_util.get_random_matrix(no_blues,coef_count,[-1000,1000])
            red_coeffs = self.random_util.get_random_matrix(no_reds,coef_count,[-1000,1000])

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
                    prev_blue_organisms = env.dead_blue_organisms
                    prev_red_organisms =  env.dead_red_organisms 
                    
                env = GameEnv(blue_coeffs,red_coeffs,max_degree,hparams['food_count'],hparams['board_size'])
                #utils.validate_py_environment(py_environment, episodes=5)
                
                # Evaluate the GA
                if self.eval_interval > 0 and (generation_number+1) % self.eval_interval == 0:
                    eval_blue_coeffs = self.ga_util.get_coeffs_from_best(prev_blue_organisms, eval_game_params['no_blue_organisms'], eval_game_params['no_blue_organisms'], 0, [0,0])
                    eval_red_coeffs = self.ga_util.get_coeffs_from_best(prev_red_organisms, eval_game_params['no_red_organisms'], eval_game_params['no_red_organisms'], 0, [0,0])
                    eval_py_env = GameEnv(eval_blue_coeffs,eval_red_coeffs,max_degree,eval_game_params['food_count'],eval_game_params['board_size'])
                    
                    avg_return = self.__evaluate_ga(blue_coeffs,red_coeffs,eval_py_env,self.display_moves)
                    returns.append(avg_return)
                    if self.should_log:
                        print(avg_return)
                
                # Play the game
                time_step = env.reset()
                while not time_step.is_last():
                    #print(time_step.observation.numpy())
                    time_step = env.step()

                # Pick best genomes for the next generation
                blue_organisms = env.dead_blue_organisms
                blue_coeffs = self.ga_util.get_coeffs_from_best(blue_organisms, no_blues, hparams['no_best'], round(no_random), mutation_factor_range)
                red_organisms = env.dead_red_organisms
                red_coeffs = self.ga_util.get_coeffs_from_best(red_organisms, no_reds, hparams['no_best'], round(no_random), mutation_factor_range)
                
                # Reduce the number of random organisms and mutation_factor_range
                no_random += random_step
                mutation_factor_range += mutation_factor_range_step

            self.plotter.plot_simple_values(y=list(returns),directory='score.jpeg')
            for single_organism_coefs in red_coeffs:
                self.__plot_coefs(single_organism_coefs,'red-coeffs.jpeg')
            for single_organism_coefs in blue_coeffs:
                self.__plot_coefs(single_organism_coefs,'blue-coeffs.jpeg')
            hparams['random_seed'] = self.random_seed
            self.plotter.dump_to_json(hparams,'hparams.json')