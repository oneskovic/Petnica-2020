import gym
from game_env_gym import GameEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import time
import numpy as np

env = GameEnv()
#env = gym.make('CartPole-v1')

# model = DQN(MlpPolicy, env, verbose=1, tensorboard_log ='tb_log')
# model.learn(total_timesteps=10000000)
# model.save("deepqrcina")

model = DQN.load('deepqrcina')

obs = env.reset()
done = False
total_reward = 0.0

q_values_for_actions = [[],[],[],[]]

def predict(observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = model._is_vectorized_observation(observation, model.observation_space)

        observation = observation.reshape((-1,) + model.observation_space.shape)
        q_values = []
        with model.sess.as_default():
            actions, q_values, _ = model.step_model.step(observation, deterministic=deterministic)

        for i in range(len(q_values[0])):
            q_values_for_actions[i].append(q_values[0][i])

        if not vectorized_env:
            actions = actions[0]

        return actions, None

while not done:
    action, _states = predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render(mode="graphical")
    time.sleep(0.1)

for i in range(len(q_values_for_actions)):
    np_array = np.asarray(q_values_for_actions[i])
    np.savetxt("q_values"+str(i)+".csv", np_array, delimiter=",")
print(total_reward)