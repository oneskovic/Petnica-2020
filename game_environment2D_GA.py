from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
from copy import deepcopy
from collections import deque

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class Organism:
    time_alive = 0
    x_pos = 0
    y_pos = 0
    energy = 0
    id = -1
    type = -1
    time_to_reproduce = 0
    coefficients = []
    polynomial_max_degree = 0
    def __multiply_params(self,parameters,times_used):
        result = 1.0
        for i in range(len(parameters)):
            result *= np.power(parameters[i],times_used[i])
        return result
        
    def __eval_function(self,parameters,times_used,position,max_degree,available_coefs):
        # Each recursive call will compute the sum of all parameter combinations
        # where each parameter is used max_degree times. Each call returns the sum of
        # all combinations starting with the given prefix - eg. if position = 3
        # the prefix is the first 3 parameters  
        if position >= len(parameters):
            return self.__multiply_params(parameters,times_used) * available_coefs.popleft()  
        
        sub_sum = 0.0
        for _ in range(max_degree+1):
            sub_sum += self.__eval_function(parameters,times_used,position+1,max_degree,available_coefs)
            times_used[position] += 1
        times_used[position] = 0
        return sub_sum
    
    def compute_function_recursive(self, parameters):
        times_used = [0]*len(parameters)
        available_coefs = deque(self.coefficients)
        return self.__eval_function(parameters,times_used,0,self.polynomial_max_degree,available_coefs)
        
    def compute_function_value(self,distance):
        distance_pow = 1
        result = 0.0
        for coef in self.coefficients:
            result += coef * distance_pow
            distance_pow *= distance
        return result

    def __init__(self, x_position, y_position, energy, type, time_to_reproduce, polynomial_degree = 0, coefficients = []):
        self.x_pos = x_position
        self.y_pos = y_position
        self.energy = energy
        self.type = type
        self.polynomial_max_degree = polynomial_degree
        self.coefficients = coefficients
        self.time_to_reproduce = time_to_reproduce
        self.time_alive = 0
        self.id = np.random.randint(-100000000,100000000)

    def to_list(self, include_coefs = False):
        if include_coefs:
            return [self.x_pos,self.y_pos,self.energy,self.time_to_reproduce,self.type,self.time_alive] + self.coefficients
        else:
            return [self.x_pos,self.y_pos,self.energy,self.time_to_reproduce,self.type,self.time_alive]

class GameEnv(py_environment.PyEnvironment):
  # Lists containing the currently alive organisms
  blue_organisms = []
  red_organisms = []

  # Lists containing the organisms that have died (used to pick the best organisms)
  dead_red_organisms = []
  dead_blue_organisms = []

  # Coefficients used when resetting (or starting) the game
  blue_starting_coefs = []
  red_starting_coefs = []
  polynomial_degree = 0

  green_organisms = []
  start_hp = 20 # HP / energy that all organisms have when the game starts
  board_length = 10 # Board will be of size board_length x board_length
  food_energy = 10 # Energy that each green organism will have
  max_moves = 200 # The maximum number of moves a game can last
  current_move_number = 0
  board_food_count = 10 # The number of green organisms always present on the board
  reproduction_cooldown = 3

  def __init__(self, blue_start_coefs, red_start_coefs, polynomial_degree, food_count = 10, board_size = 10):
    # Initialize specs
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(6*(self.board_food_count+len(blue_start_coefs)+len(red_start_coefs)),), dtype=np.int32, minimum=0, name='observation')

    self._episode_ended = False
    self.blue_organisms = []
    self.red_organisms = []
    self.dead_blue_organisms = []
    self.dead_red_organisms = []
    self.polynomial_degree = polynomial_degree
    self.board_food_count = food_count
    self.board_length = board_size

    # Initialize blue organisms
    for start_coefs in blue_start_coefs:
        x_pos = np.random.randint(0,self.board_length)
        y_pos = np.random.randint(0,self.board_length)
        self.blue_organisms.append(Organism(x_pos,y_pos,self.start_hp,1,self.reproduction_cooldown,self.polynomial_degree,start_coefs))
    
    self.blue_starting_coefs = blue_start_coefs
    
    # Initialize red organisms
    for start_coefs in red_start_coefs:
        x_pos = np.random.randint(0,self.board_length)
        y_pos = np.random.randint(0,self.board_length)
        self.red_organisms.append(Organism(x_pos,y_pos,self.start_hp,2,self.reproduction_cooldown,self.polynomial_degree,start_coefs))

    self.red_starting_coefs = red_start_coefs

    self.green_organisms = []
    # Generate green organisms
    for _ in range(self.board_food_count):
        x_pos = np.random.randint(0,self.board_length)
        y_pos = np.random.randint(0,self.board_length)

        self.green_organisms.append(Organism(x_pos,y_pos,self.food_energy,0,self.reproduction_cooldown))

    self.current_move_number = 0

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def __organisms_to_array(self,organisms_list, starting_count):
    organism_array = []
    for i in range(min(starting_count,len(organisms_list))):
        organism_array += organisms_list[i].to_list()
    for _ in range(max(0,starting_count - len(organisms_list))):
        organism_array += [0]*6
    return organism_array

  def __get_current_game_state(self):
    state = []
    
    state += self.__organisms_to_array(self.blue_organisms, len(self.blue_starting_coefs))
    state += self.__organisms_to_array(self.red_organisms, len(self.red_starting_coefs))    

    for green_organism in self.green_organisms:
        state += green_organism.to_list()
    return state  

  def _reset(self):
    self._episode_ended = False

    self.blue_organisms = []
    self.dead_blue_organisms = []
    self.red_organisms = []
    self.dead_red_organisms = []
    
    for start_coefs in self.blue_starting_coefs:
        x_pos = np.random.randint(0,self.board_length)
        y_pos = np.random.randint(0,self.board_length)
        self.blue_organisms.append(Organism(x_pos,y_pos,self.start_hp,1,self.reproduction_cooldown,self.polynomial_degree,start_coefs))
    
    for start_coefs in self.red_starting_coefs:
        x_pos = np.random.randint(0,self.board_length)
        y_pos = np.random.randint(0,self.board_length)
        self.red_organisms.append(Organism(x_pos,y_pos,self.start_hp,2,self.reproduction_cooldown,self.polynomial_degree,start_coefs))

    self.green_organisms = []
    # Generate food
    for _ in range(self.board_food_count):
        x_pos = np.random.randint(0,self.board_length)
        y_pos = np.random.randint(0,self.board_length)

        self.green_organisms.append(Organism(x_pos,y_pos,self.food_energy,0,self.reproduction_cooldown))
    self.current_move_number = 0

    self._state = self.__get_current_game_state()
    return ts.restart(np.array(self._state, dtype=np.int32))

  def __reduce_organism_energy(self, organisms):
    for i in range(len(organisms)):
        organisms[i].energy -= 1
        organisms[i].time_to_reproduce = max(0,organisms[i].time_to_reproduce-1)
        organisms[i].time_alive += 1

  def __move_organism(self, organism, action):
        organism_copy = deepcopy(organism)
        # Process organism's action
        if action == 0:
            organism_copy.x_pos = (organism.x_pos-1+self.board_length)%self.board_length
        elif action == 1:
            organism_copy.x_pos = (organism.x_pos+1)%self.board_length
        elif action == 2:
            organism_copy.y_pos = (organism.y_pos+1)%self.board_length
        elif action == 3:
            organism_copy.y_pos = (organism.y_pos-1+self.board_length)%self.board_length
        else:
            raise ValueError('`action` should be in range [0,3].')
        return organism_copy

  def __get_distance(self,organism1,organism2):
    dx = organism1.x_pos - organism2.x_pos
    dy = organism1.y_pos - organism2.y_pos
    return abs(dx) + abs(dy)
  
  def __compute_organism_action(self,organism,other_organisms):
    if len(other_organisms) == 0:
        return 0
    # Find the organism that maximises the function
    max_result_organism = other_organisms[0]
    max_result = float('-inf')
    for other_organism in other_organisms:
        if organism.id == other_organism.id:
            continue
        distance = self.__get_distance(organism,other_organism)
        #function_value = organism.compute_function_value(distance)
        function_value = organism.compute_function_recursive([distance])
        if function_value > max_result:
            max_result = function_value
            max_result_organism = other_organism
    
    dx = organism.x_pos - max_result_organism.x_pos
    dy = organism.y_pos - max_result_organism.y_pos

    best_distance = dx*dx + dy*dy
    organism_action = 0
    possible_actions = [0,1,2,3]
    for action in possible_actions:
        new_organism = self.__move_organism(organism,action)
        distance = self.__get_distance(new_organism,max_result_organism)

        if distance <= best_distance and organism.type >= max_result_organism.type:
            organism_action = action
            best_distance = distance

        if distance >= best_distance and organism.type <= max_result_organism.type:
            organism_action = action
            best_distance = distance

    return organism_action
  
  # Computes and processes actions for organisms, computing function values for each organism in other_organisms
  def __process_actions_for_organisms(self,organisms, other_organisms):
    for i in range(len(organisms)):
        action = self.__compute_organism_action(organisms[i], other_organisms)
        organisms[i] = self.__move_organism(organisms[i],action)

  # Checks if any prey should be consumed, adds energy to the adequate predator organism
  # Sets energy = 0 for any prey that is consumed - dead prey must be removed afterwards
  def __consume_prey(self,predator_organisms, prey_organisms):
    for i in range(len(prey_organisms)):
        prey = prey_organisms[i]
        for j in range(len(predator_organisms)):
            predator = predator_organisms[j]
            if predator.x_pos == prey.x_pos and predator.y_pos == prey.y_pos:
                predator_organisms[j].energy += prey.energy
                prey_organisms[i].energy = 0
                break

  def __reproduce_organisms(self,organisms):
      for i in range(len(organisms)):
          for j in range(len(organisms)):
              if i != j:
                  if organisms[i].x_pos == organisms[j].x_pos and \
                  organisms[i].y_pos == organisms[j].y_pos and \
                  organisms[i].time_to_reproduce <= 0 and organisms[j].time_to_reproduce <= 0:
                      organisms[i].time_to_reproduce = self.reproduction_cooldown
                      organisms[j].time_to_reproduce = self.reproduction_cooldown
                      child_energy = organisms[i].energy/2 + organisms[j].energy/2
                      organisms[i].energy /= 2
                      organisms[j].energy /= 2
                      coefs = [0]*len(organisms[i].coefficients)
                      for coef_index in range(len(coefs)):
                          coefs[coef_index] = (organisms[i].coefficients[coef_index] + 
                                              organisms[j].coefficients[coef_index])/2.0
                      child = Organism(organisms[i].x_pos,organisms[i].y_pos,child_energy,organisms[i].type,
                                       self.reproduction_cooldown,organisms[i].polynomial_max_degree,coefs)
                      organisms.append(child)
  def _step(self, action):
    reward = 0
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    self.__reduce_organism_energy(self.blue_organisms)
    self.__reduce_organism_energy(self.red_organisms)

    # Remove dead blue organisms
    self.dead_blue_organisms += [organism for organism in self.blue_organisms if organism.energy <= 0]
    self.blue_organisms = [organism for organism in self.blue_organisms if organism.energy > 0]
    # Remove dead red organisms
    self.dead_red_organisms += [organism for organism in self.red_organisms if organism.energy <= 0]
    self.red_organisms = [organism for organism in self.red_organisms if organism.energy > 0]

    # Make sure episodes don't go on forever.
    if self.current_move_number >= self.max_moves:
        self._episode_ended = True
        self.dead_blue_organisms += self.blue_organisms
        self.dead_red_organisms += self.red_organisms
    elif len(self.blue_organisms) == 0 or len(self.red_organisms) == 0:
        self._episode_ended = True
        if len(self.blue_organisms) > 0:
            self.dead_blue_organisms += self.blue_organisms
        if len(self.red_organisms) > 0:
            self.dead_red_organisms += self.red_organisms
        #reward = -20
    else:
        self.current_move_number += 1

    if not self._episode_ended:    
        all_organisms = self.green_organisms+self.red_organisms+self.blue_organisms
        self.__process_actions_for_organisms(self.blue_organisms,all_organisms)
        self.__process_actions_for_organisms(self.red_organisms,self.blue_organisms+self.red_organisms)
    
        reward = len(self.blue_organisms) + len(self.red_organisms)
        
        self.__consume_prey(self.blue_organisms,self.green_organisms)
        self.__consume_prey(self.red_organisms,self.blue_organisms)

        # Remove dead green and blue organisms. This must be done again 
        # for green and blue as additional organisms might have been consumed
        self.green_organisms = [organism for organism in self.green_organisms if organism.energy > 0]            
        self.dead_blue_organisms += [organism for organism in self.blue_organisms if organism.energy <= 0]
        self.blue_organisms = [organism for organism in self.blue_organisms if organism.energy > 0]            
    
        self.__reproduce_organisms(self.blue_organisms)
        self.__reproduce_organisms(self.red_organisms)
    
        # Generate new green organisms
        while len(self.green_organisms) != self.board_food_count:
            x_pos = np.random.randint(0,self.board_length)
            y_pos = np.random.randint(0,self.board_length)
            self.green_organisms.append(Organism(x_pos,y_pos,self.food_energy,0,self.reproduction_cooldown))

    self._state = self.__get_current_game_state()
    if self._episode_ended:
      return ts.termination(np.array(self._state, dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=reward, discount=0.9)