#! /usr/bin/env python3
import os

import ipdb
import math
import numpy as np
import sys

from matplotlib import pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common import results_plotter
from hit_ball_env import HitBallEnv


# Control whether to do onscreen or offscreen.
# Can't do both. Can't give robot images if you do onscreen.
on_screen_render = False
# If you aren't rendering on screen, do you want to see what the robot sees? It's slow...
matplotlib_display = True and not on_screen_render

def make_env():
    return HitBallEnv(
        robots = ['IIWA'],
        env_configuration = ['default'],    # positions
        controller_configs = {'type':'OSC_POSE', 'interpolation': 'linear', 'ramp_ratio':0.6 },
        gripper_types = ['BatOneGripper'],
        use_camera_obs = True,  # True means controller will be given camera inputs
        reward_shaping = True,   # Whether to offer partial rewards for partial success
        has_renderer = False,    # True means you will see the visuals; can't be both on and off screen though.
        has_offscreen_renderer = True,    # Required if you want camera observations for the controller.
        render_camera = 'aboverobot',   # name of camera to render (None = default which the user can control)
        render_collision_mesh = False,
        render_visual_mesh = True,
        control_freq = 30,      # Hz of controller being called
        horizon = 256,          # Number of control steps in an episode (not seconds, not time steps, but control steps)
        camera_names = ['aboverobot'],   # Cameras to be used for observations to controller
        camera_heights = 160,  # 84 was default, but our ball is small and hard to see
        camera_widths = 160,
        camera_depths = True,   # True if you want RGB-D cameras
        # There are more optional args, but I didn't think them relevant.
    )

if __name__ == '__main__':

    # Create vectorized environments
    num_env = 1
    if num_env > 1:
        venv = SubprocVecEnv([make_env]*num_env)
    else:
        venv = DummyVecEnv([make_env]*1)
    env = VecFrameStack(venv, n_stack=4)


    #print("SS = ", venv.observation_space)
    #print("AS = ", venv.action_space)

    class SaveAfterEpisodeCallback(BaseCallback):
        def on_rollout_end(self):
            print('Rollout end')
            self.model.save('save_checkpoint_backup.model')
            print('Checkpoint saved')

        def _on_step(self):
            return True

    class ActionNormPrintCallback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 64 == 0:
                n = np.linalg.norm(self.locals['actions'])
                print('Action norm: {:.4}'.format(n))
            return True

    class SaveReward(BaseCallback):
        def __init__(self):
            super().__init__()
            self.my_reward = 0
            self.my_rewards = []

        def _on_step(self) -> bool:
            # print(self.locals)
            self.my_reward += self.locals["rewards"][0]
            if self.locals["dones"][0]:
                self.my_rewards.append(self.my_reward)
                self.my_reward = 0
            return True

    # Prepare agent
    load_filename = sys.argv[1] if len(sys.argv) > 1 else None

    agent = PPO("MultiInputPolicy", env, verbose=1)
    if load_filename is not None:
        agent.load(load_filename)

    # learn
    reward_record = SaveReward()
    agent.learn(1_000_000, callback=CallbackList([SaveAfterEpisodeCallback(), reward_record])) # , ActionNormPrintCallback()]))

    plt.plot(reward_record.my_rewards)
    plt.show()