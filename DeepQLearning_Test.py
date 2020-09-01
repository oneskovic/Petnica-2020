from __future__ import absolute_import, division, print_function

import base64
import json
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import utils
from tf_agents.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from game_environment2D import GameEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 2  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 32  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200 # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

start_epsilon = 1
end_epsilon = 0.1
epsilon_anneal_steps = 4000

nn_update_frequency = 4
td_sample_size = 4

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)
    
def plot_organisms(organisms,cmap):
  x_positions = []
  y_positions = []
  hps = []
  for i in range(len(organisms)):
    x_positions.append(organisms[i][0])
    y_positions.append(organisms[i][1])
    hps.append(organisms[i][2])
  plt.scatter(x_positions, y_positions,c = hps,cmap=cmap,vmin=0,vmax=20)
  
def plot_state(state,directory):
  i = 0
  red_organisms = []
  blue_organisms = []
  green_organisms = []
  while i < len(state):
    x = state[i]
    y = state[i+1]
    hp = state[i+2]
    type = state[i+4]
    if type == 0:
      green_organisms.append([x,y,hp])
    elif type == 1:
      blue_organisms.append([x,y,hp])
    else:
      red_organisms.append([x,y,hp])
    i += 5
  plt.cla()
  plt.xlim(0,10)
  plt.ylim(0,10)
  plot_organisms(red_organisms,'Reds')
  plot_organisms(blue_organisms,'Blues')
  plot_organisms(green_organisms,'Greens')
  plt.savefig(directory)
  plt.cla()
    
picture_count = 0
episode_count = 0

def compute_avg_return(environment, policy, num_episodes=10, display_moves = False):

  if display_moves:
    global episode_count
    episode_path = 'results/DeepQ-2/game-drawn/episode-'+str(episode_count)
    if not os.path.exists(episode_path):
      os.makedirs(episode_path)
    
    time_step = environment.reset()
    episode_return = 0.0

    global picture_count
    while not time_step.is_last():
      observation = time_step.observation.numpy()[0]
      plot_state(observation,episode_path+'/'+str(picture_count)+'.jpeg')
      picture_count += 1
      #print(time_step.observation.numpy())
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    print('Episode return: '+str(episode_return))
    picture_count = 0
    episode_count += 1

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return
    
  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

def solve_perfectly(environment, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
      time_step = environment.reset()
      while not time_step.is_last():
        dx = time_step.observation.numpy()[0][0] - time_step.observation.numpy()[0][3]
        action = np.array(0, dtype = np.int32)
        if abs(dx) > 0:
            action = np.array(0, dtype=np.int32)
            if dx < 0:
              action = np.array(1, dtype=np.int32)
        else:
            dy = time_step.observation.numpy()[0][1] - time_step.observation.numpy()[0][4]
            action = np.array(3, dtype=np.int32)
            if dy < 0:
                action = np.array(2, dtype=np.int32)

        time_step = environment.step(action)
        total_return += time_step.reward

    return total_return.numpy()[0] / num_episodes
losses = [] # losses will contain losses generated in each iteration

def try_hparams(hparams):

    # Initialize train and eval environments
    environment = GameEnv()
    utils.validate_py_environment(environment, episodes=5)
    train_py_env = GameEnv()
    eval_py_env = GameEnv()
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Initialize the QNetwork
    fc_layer_params = (hparams['layer1_count'],hparams['layer2_count'],)#hparams['layer3_count'],)
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.95, epsilon=1e-07)

    train_step_counter = tf.Variable(0)

    # Initialize the DQN Agent
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        n_step_update=td_sample_size,
        target_update_period=nn_update_frequency,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    # Collect some data using a totaly random policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)

    # Convert replay buffer to dataset
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=td_sample_size+1).prefetch(3)

    iterator = iter(dataset)

    agent.train = common.function(agent.train)
    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agents policy, random policy and optimal policy once before training
    optimal_return = 0 #solve_perfectly(eval_env,num_eval_episodes)
    random_return = compute_avg_return(eval_env,random_policy,num_eval_episodes)
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    
    returns = [avg_return] # returns will contain all average returns of the agent during training
    global losses
    epsilon = start_epsilon
    epsilon_step = (start_epsilon-end_epsilon) / epsilon_anneal_steps

    for _ in range(num_iterations):
      # Reduce epsilon
      epsilon = max(epsilon - epsilon_step, end_epsilon)
        
      # Collect a few steps using the epsilon greedy policy and save to the replay buffer.
      for _ in range(collect_steps_per_iteration):
        collect_step(train_env, EpsilonGreedyPolicy(agent.policy, epsilon), replay_buffer)

      # Sample a batch of data from the buffer and update the agent's network.
      experience, unused_info = next(iterator)
      train_loss = agent.train(experience).loss

      step = agent.train_step_counter.numpy()

      if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))
        losses.append(train_loss)
        
      if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes, True)
        print('step = {0}: Average Return = {1} Optimal policy = {2} Random policy = {3}'.format(step, avg_return, optimal_return,random_return))
        returns.append(avg_return)
    
    #plt.plot(losses)
    #plt.show()
    #plt.cla()
    return returns

layer1_counts = [100]
layer2_counts = [20]

import string
import random
def generate_random_string(length):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

for count1 in layer1_counts:
    for count2 in layer2_counts:
        hparams = {
            'layer1_count': count1,
            'layer2_count': count2
            }
        results = try_hparams(hparams)
        file_name = generate_random_string(6)
        result_file = open('results/DeepQ-2/'+str(file_name)+'.json','w+')
        hparams_map = {
          'num_iterations':num_iterations,
          'initial_collect_steps': initial_collect_steps, 
          'collect_steps_per_iteration': collect_steps_per_iteration,
          'replay_buffer_max_length': replay_buffer_max_length,

          'batch_size': batch_size,
          'learning_rate': learning_rate,
          'log_interval': log_interval,

          'num_eval_episodes': num_eval_episodes,
          'eval_interval': eval_interval,

          'start_epsilon': start_epsilon,
          'end_epsilon': end_epsilon,
          'epsilon_anneal_steps': epsilon_anneal_steps,

          'nn_update_frequency': nn_update_frequency,
          'td_sample_size': td_sample_size,
          'nn_architecture': (count1,count2)
        }
        json.dump(hparams_map,result_file,sort_keys=True, indent=4)
        result_file.close()
        plt.cla()
        plt.plot(losses)
        plt.savefig('results/DeepQ-2/Loss-'+str(file_name))
        plt.cla()
        plt.plot(results)
        plt.savefig('results/DeepQ-2/Score-'+str(file_name))
        plt.cla()
