#! /usr/bin/env python3

import ipdb
import math
import numpy as np
import sys
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from hit_ball_env import HitBallEnv

import torch
torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device=None, abbreviated=False))


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
        use_camera_obs = not on_screen_render,  # True means controller will be given camera inputs
        reward_shaping = True,   # Whether to offer partial rewards for partial success
        has_renderer = on_screen_render,    # True means you will see the visuals; can't be both on and off screen though.
        has_offscreen_renderer = not on_screen_render,    # Required if you want camera observations for the controller.
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

# Create vectorized environments
num_env = 5
venv = SubprocVecEnv([make_env]*num_env, 'fork')

class SaveAfterEpisodeCallback(BaseCallback):
    def on_rollout_end(self):
        print('Rollout end. Saving checkpoint.')
        self.model.save('save_checkpoint.model')

    def _on_step(self):
        return True

# Prepare agent
load_filename = sys.argv[1] if len(sys.argv) > 1 else None

agent = RecurrentPPO("MultiInputLstmPolicy", venv, verbose=1)
if load_filename is not None:
    agent.load(load_filename)
print(agent.policy)

# learn
agent.learn(10_000, callback=SaveAfterEpisodeCallback())

