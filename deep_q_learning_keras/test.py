import gym
from game_env_gym import GameEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


env = GameEnv()
#env = gym.make('CartPole-v1')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000000)
model.save("deepqrcina")

# model = DQN.load('deepqrcina')

# obs = env.reset()
# done = False
# total_reward = 0.0
# while not done:
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     total_reward += reward
#     env.render()
# print(total_reward)