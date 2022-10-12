import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import matplotlib.pyplot as plt


# local
from stable_baselines_helpers import *

# main hyper parameter
training_steps = 1e4

# create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Logs will be saved in log_dir/monitor.csv
env = gym.make("CartPole-v1")
env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)


# create alg and learn it for 10k steps
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=training_steps, callback=callback)

# Show the learned policy on the env
obs = env.reset()
for i in range(1):
    # returns action from the agent
    action, _states = model.predict(obs, deterministic=True)

    # normal environment stuff
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()

# plot everything
results_plotter.plot_results([log_dir], training_steps, results_plotter.X_TIMESTEPS, "PPO Cartpole")
plot_results(log_dir)