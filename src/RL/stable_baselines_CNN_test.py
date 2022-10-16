import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import matplotlib.pyplot as plt

# local
from stable_baselines_helpers import *

# main hyper parameter
training_steps = 1e4

# create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# There already exists an environment generator that will make and wrap atari environments correctly.
# exchange this for our env
env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
# Stack 4 frames in time - so no recurrence needed even though its images
env = VecFrameStack(env, n_stack=4)


# create alg and learn it for 10k steps
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=training_steps)

# Show the learned policy on the env
obs = env.reset()
for i in range(10000):
    # returns action from the agent
    action, _states = model.predict(obs, deterministic=True)

    print(type(obs), obs.shape)

    # normal environment stuff
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
