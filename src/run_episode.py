#! /usr/bin/env python3

import ipdb
import math
import numpy as np
import sys
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

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
    env = make_env()
    observation, info = env.reset()
    print("SS:")
    for key in env.observation_space:
        print("\t", key, " = ", env.observation_space[key].shape)
    print("AS = ", env.action_space.shape)


    for i in range(1000):
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)

        # print("Reward ", reward)

        if done:
            obs, info = env.reset()

