#! /usr/bin/env python3

import collections
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

if __name__ == '__main__':
    # Create vectorized environments
    num_env = 4
    if num_env > 1:
        venv = SubprocVecEnv([make_env]*num_env)
    else:
        venv = DummyVecEnv([make_env]*1)

    class SaveAfterEpisodeCallback(BaseCallback):
        def on_rollout_end(self):
            print('Rollout end')
            self.model.save('save_checkpoint.model')
            print('Checkpoint saved')

        def _on_step(self):
            return True

    class ActionNormPrintCallback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 256 == 0:
                n = np.linalg.norm(self.locals['actions'])
                print('Action norm: {:.4}'.format(n))
            return True

    class RewardPrintCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.d = collections.deque([], 256*num_env*2)
        def _on_step(self):
            self.d.append(self.locals['rewards'])
            if self.num_timesteps % 256 == 0:
                n = np.mean(np.vstack(self.d)) * 256
                print('Mean reward over {:d} episodes: {:.4}'.format(int(len(self.d)/256), n))
            return True

    # Prepare agent
    load_filename = sys.argv[1] if len(sys.argv) > 1 else None

    agent = RecurrentPPO("MultiInputLstmPolicy", venv, verbose=1)
    if load_filename is not None:
        agent.load(load_filename)
    print(agent.policy)

    # learn
    expected_fps = 105
    approx_seconds_to_run = 5*60
    steps_to_run = expected_fps * approx_seconds_to_run
    agent.learn(steps_to_run, callback=CallbackList([SaveAfterEpisodeCallback(), ActionNormPrintCallback(), RewardPrintCallback()]))

