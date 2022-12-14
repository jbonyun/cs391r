#! /usr/bin/env python3

import imageio
import ipdb
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import sys

from hit_ball_env import HitBallEnv

import torch
torch.cuda.empty_cache()

model_filename = sys.argv[1] if len(sys.argv) > 1 else None
if model_filename is None:
    raise Exception('Must provide filename to load model from')


num_env = 3
control_freq = 15
horizon = 64


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
        render_camera = 'followrobot',   # name of camera to render (None = default which the user can control)
        render_collision_mesh = False,
        render_visual_mesh = True,
        control_freq = control_freq, # Hz of controller being called
        horizon = horizon,           # Number of control steps in an episode (not seconds, not time steps, but control steps)
        camera_names = ['followrobot','aboverobot'],   # Cameras to be used for observations to controller
        camera_heights = 84,  # 84 was default
        camera_widths = 84,
        camera_depths = True,   # True if you want RGB-D cameras
        camera_color = False,
        # There are more optional args, but I didn't think them relevant.
    )



class MakeVideoCallback(BaseCallback):
    def __init__(self, fname, camname, fps=30, verbose=0, num_envs=1):
        super().__init__(verbose)
        self.fname = fname
        self.camname = camname
        self.fps = fps
        self.num_envs = num_envs
        self.vid = []
    def _on_training_start(self):
        print('Vid start')
        for i in range(self.num_envs):
            #fname = self.fname
            fname = self.fname.format(i)
            self.vid.append(imageio.get_writer(fname, fps=self.fps))
    def _on_step(self):
        for i in range(self.num_envs):
            im = self.locals['infos'][i]['observer']
            self.vid[i].append_data(im)
        if self.locals['n_steps'] >= self.locals['total_timesteps'] / self.num_envs:
            print('Did all timesteps in an episode')
            return False
        return True
    def close(self):
        print('Vid end')
        for i in range(self.num_envs):
            self.vid[i].close()

if __name__ == '__main__':

    if num_env > 1:
        venv = SubprocVecEnv([make_env]*num_env, 'fork')
    else:
        venv = DummyVecEnv([make_env]*num_env)
    agent = RecurrentPPO.load(model_filename, venv, verbose=1)

    # We will "learn" for one cycle, but really we're just recording the rollout.
    vid = MakeVideoCallback('rollout_{}.mp4', 'followrobot', fps=control_freq, num_envs=num_env)
    agent.learn(horizon*num_env, callback=vid)
    vid.close()

