import numpy as np
import gym
from collections import deque
from copy import deepcopy
from gym import spaces
from termcolor import colored

class Organism:
    time_alive = 0
    time_to_reproduce = 0
    x_pos = 0
    y_pos = 0
    energy = 0
    id = -1
    type = -1

    def __init__(self, x_position, y_position, energy, type, time_to_reproduce):
      self.x_pos = x_position
      self.y_pos = y_position
      self.energy = energy
      self.type = type
      self.time_to_reproduce = time_to_reproduce
      self.time_alive = 0
      self.id = np.random.randint(-100000000,100000000)

    def to_list(self):
      return [self.x_pos,self.y_pos,self.energy,self.time_to_reproduce,self.type]


class GameEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1
    # Lists containing the currently alive organisms
    blue_organisms = []
    red_organisms = []

    green_organisms = []
    start_hp = 20 # HP / energy that all organisms have when the game starts
    board_length = 10 # Board will be of size board_length x board_length
    food_energy = 10 # Energy that each green organism will have
    max_moves = 200 # The maximum number of moves a game can last
    current_move_number = 0
    board_food_count = 10 # The number of green organisms always present on the board
    blue_organisms_start_count = 10
    red_organisms_start_count = 10
    #input_layer_count = 15
    reproduction_cooldown = 3

    organisms_to_move = []
    player_to_move = 'Red'

    def __init__(self, grid_size=10):
        super(GameEnv, self).__init__()    

        self.action_space = spaces.Discrete(4)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=-1, high=20,
                                            shape=(self.board_length,self.board_length,3), dtype=np.float32)

        self._episode_ended = False

        self.blue_organisms = []
        # Initialize blue organisms
        for _ in range(self.blue_organisms_start_count):
            x_pos = np.random.randint(0,self.board_length)
            y_pos = np.random.randint(0,self.board_length)
            self.blue_organisms.append(Organism(x_pos,y_pos,self.start_hp,1,self.reproduction_cooldown))
        
        self.red_organisms = []
        # Initialize red organisms
        for _ in range(self.red_organisms_start_count):
            x_pos = np.random.randint(0,self.board_length)
            y_pos = np.random.randint(0,self.board_length)
            self.red_organisms.append(Organism(x_pos,y_pos,self.start_hp,2,self.reproduction_cooldown))

        self.green_organisms = []
        # Generate green organisms
        for _ in range(self.board_food_count):
            x_pos = np.random.randint(0,self.board_length)
            y_pos = np.random.randint(0,self.board_length)

            self.green_organisms.append(Organism(x_pos,y_pos,self.food_energy,0,0))

        #self.organism_to_move = self.blue_organisms[0]
        self.organisms_to_move = deepcopy(self.blue_organisms)
        self.player_to_move = 'Blue'
        self.current_move_number = 0

    def reshape_observation(self,observation):
        reshaped = []
        for row in range(self.board_length):
            reshaped.append([0]*self.board_length)
        for col in range(self.board_length):
            reshaped[row][col] = [0]*4
            
        for row in range(self.board_length):
            for col in range(self.board_length):
                for frame in range(4):
                    reshaped[row][col][frame] = observation[frame][row][col]
        
        return reshaped

    def organisms_to_array(self, organisms_list, required_count):
        organism_array = []
        # Filter out the organism that should move next (appended to the front)
        #organisms_list = [organism for organism in organisms_list if organism.id != self.organism_to_move.id]
        
        # Attempt to add the required_count of organisms
        for i in range(int(min(required_count,len(organisms_list)))):
            organism_array += organisms_list[i].to_list()
        
        # If the required_count has not been reached fill the rest with 0 elements
        for _ in range(int(max(0,required_count - len(organisms_list)))):
            organism_array += [0]*5
        return organism_array

    def __get_distance(self, organism1, organism2):
        dx = organism1.x_pos - organism2.x_pos
        dy = organism1.y_pos - organism2.y_pos
        return abs(dx)+abs(dy)

    # Sorts organisms by their distance from target_organism
    def __sort_organisms_by_distance(self, organisms, target_organism):
        distance_dict = {}
        # Should append to list not overwrite
        for organism in organisms:
            # Dirty, temporary fix
            noise = np.random.uniform(0.0000000001,0.00001)
            distance_dict[self.__get_distance(organism, target_organism)+noise] = organism
            
        ordered_organisms = []
        for distance in sorted(distance_dict.keys()):
            current_organism = distance_dict[distance]
            ordered_organisms.append(current_organism)
            
        return ordered_organisms
        
    def __get_current_game_state(self):
        # Initialize the state array
        state = np.zeros((self.board_length,self.board_length,3))
        
        for organism in self.green_organisms:
            state[organism.y_pos][organism.x_pos][0] = 1
            state[organism.y_pos][organism.x_pos][1] = self.food_energy
            state[organism.y_pos][organism.x_pos][2] = 0 
        for organism in self.blue_organisms:
            state[organism.y_pos][organism.x_pos][0] = 2
            state[organism.y_pos][organism.x_pos][1] = organism.energy
            state[organism.y_pos][organism.x_pos][2] = 0 
        for organism in self.red_organisms:
            state[organism.y_pos][organism.x_pos][0] = 3
            state[organism.y_pos][organism.x_pos][1] = organism.energy
            state[organism.y_pos][organism.x_pos][2] = 0
        
        organism_to_move = self.organisms_to_move[0]
        state[organism_to_move.y_pos][organism_to_move.x_pos][0] = organism_to_move.type+1
        state[organism_to_move.y_pos][organism_to_move.x_pos][1] = organism_to_move.energy
        state[organism_to_move.y_pos][organism_to_move.x_pos][2] = 1
                
        #state += self.organisms_to_move[0].to_list() # Append the organism that should move to the front
        
        #ordered_blues = self.__sort_organisms_by_distance(self.blue_organisms,self.organism_to_move)
        #state += self.organisms_to_array(ordered_blues, self.input_layer_count/3)
        #ordered_reds = self.__sort_organisms_by_distance(self.red_organisms,self.organism_to_move)
        #state += self.organisms_to_array(ordered_reds,  self.input_layer_count/3)
        #ordered_greens = self.__sort_organisms_by_distance(self.green_organisms,self.organism_to_move)
        #state += self.organisms_to_array(ordered_greens, self.input_layer_count/3)
        return state

    def __update_organism_position(self, organism, action):
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
        
    def __move_organism(self, action, other_player, current_organisms, other_organisms):
        i = 0
        while i < len(self.organisms_to_move):
            found = False
            for organism_index in range(len(current_organisms)):
                if self.organisms_to_move[i].id == current_organisms[organism_index].id:
                    current_organisms[organism_index] = self.__update_organism_position(current_organisms[organism_index], action)
                    current_organisms[organism_index].energy -= 1
                    current_organisms[organism_index].time_to_reproduce = \
                        max(0,current_organisms[organism_index].time_to_reproduce-1)
                    found = True
                    break
            i+=1
            if found:
                break
            
        if i == len(self.organisms_to_move):
            self.player_to_move = other_player
            self.current_move_number += 1
            self.organisms_to_move = deepcopy(other_organisms)
        else:    
            self.organisms_to_move = self.organisms_to_move[i:]
    
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
                        energy = organisms[i].energy/2 + organisms[j].energy/2
                        organisms[i].energy /= 2
                        organisms[j].energy /= 2
                        organisms[i].time_to_reproduce = self.reproduction_cooldown
                        organisms[j].time_to_reproduce = self.reproduction_cooldown
                        new_organism = Organism(organisms[i].x_pos,organisms[j].y_pos,
                                        energy,organisms[i].type,self.reproduction_cooldown)
                        organisms.append(new_organism)

    def reset(self):
        self._episode_ended = False
        self.blue_organisms = []
        for _ in range(self.blue_organisms_start_count):
            x_pos = np.random.randint(0,self.board_length)
            y_pos = np.random.randint(0,self.board_length)
            self.blue_organisms.append(Organism(x_pos,y_pos,self.start_hp,1,self.reproduction_cooldown))
        
        self.red_organisms = []
        for _ in range(self.red_organisms_start_count):
            x_pos = np.random.randint(0,self.board_length)
            y_pos = np.random.randint(0,self.board_length)
            self.red_organisms.append(Organism(x_pos,y_pos,self.start_hp,2,self.reproduction_cooldown))

        self.green_organisms = []
        # Generate food
        for _ in range(self.board_food_count):
            x_pos = np.random.randint(0,self.board_length)
            y_pos = np.random.randint(0,self.board_length)

            self.green_organisms.append(Organism(x_pos,y_pos,self.food_energy,0,0))
            
        self.organisms_to_move = deepcopy(self.blue_organisms)
        self.player_to_move = 'Blue'
        self.current_move_number = 0

        self._state = self.__get_current_game_state()

        return np.array(self._state).astype(np.float32)

    def step(self, action):
        reward = 1
        # Make sure episodes don't go on forever.
        if self.current_move_number >= self.max_moves or len(self.blue_organisms) == 0 or len(self.red_organisms) == 0:
            self._episode_ended = True
        # if self._episode_ended:
        #     # The last action ended the episode. Ignore the current action and start
        #     # a new episode.
        #     return self.reset()

        if self.player_to_move == 'Blue':
            self.__move_organism(action,'Red',self.blue_organisms,self.red_organisms)
        else:
            self.__move_organism(action,'Blue',self.red_organisms,self.blue_organisms)
        
        # Remove dead blue organisms
        self.blue_organisms = [organism for organism in self.blue_organisms if organism.energy > 0]
        # Remove dead red organisms
        self.red_organisms = [organism for organism in self.red_organisms if organism.energy > 0]

        # Make sure episodes don't go on forever.
        if self.current_move_number >= self.max_moves or len(self.blue_organisms) == 0 or len(self.red_organisms) == 0:
            self._episode_ended = True
            #reward = -20
        else:    
            self.__consume_prey(self.blue_organisms,self.green_organisms)
            self.__consume_prey(self.red_organisms,self.blue_organisms)
            self.__reproduce_organisms(self.blue_organisms)
            self.__reproduce_organisms(self.red_organisms)

            # Remove dead green and blue organisms. This must be done again 
            # for green and blue as additional organisms might have been consumed
            self.green_organisms = [organism for organism in self.green_organisms if organism.energy > 0]            
            self.blue_organisms = [organism for organism in self.blue_organisms if organism.energy > 0]            
        
            if self.current_move_number >= self.max_moves or len(self.blue_organisms) == 0 or len(self.red_organisms) == 0:
                self._episode_ended = True
        
            # Generate new green organisms
            while len(self.green_organisms) != self.board_food_count:
                x_pos = np.random.randint(0,self.board_length)
                y_pos = np.random.randint(0,self.board_length)
                self.green_organisms.append(Organism(x_pos,y_pos,self.food_energy,0,0))

        self._state = self.__get_current_game_state()
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array(self._state).astype(np.float32), reward, self._episode_ended, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        organism_to_move = self.organisms_to_move[0]
        print('No blue organisms: ' + str(len(self.blue_organisms)))
        print('No red organisms: ' + str(len(self.red_organisms)))
        print('Organism to move is at: x={0},y={1}'.format(self.organisms_to_move[0].x_pos,self.organisms_to_move[0].y_pos))
        for row in range(self.board_length):
            for col in range(self.board_length):
                if row == organism_to_move.y_pos and col == organism_to_move.x_pos:
                    print(colored(int(self._state[row][col][0]),'yellow'),end='')
                else:
                    print(int(self._state[row][col][0]),end='')
            print('')

    def close(self):
        pass