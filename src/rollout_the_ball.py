#! /usr/bin/env python3

import imageio
import ipdb
import numpy as np
from sb3_contrib import RecurrentPPO
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import sys

from hit_ball_env import HitBallEnv

import torch
torch.cuda.empty_cache()

model_filename = sys.argv[1] if len(sys.argv) > 1 else None
if model_filename is None:
    raise Exception('Must provide filename to load model from')

# Control whether to do onscreen or offscreen.
# Can't do both. Can't give robot images if you do onscreen.
on_screen_render = False
# If you aren't rendering on screen, do you want to see what the robot sees? It's slow...
matplotlib_display = True and not on_screen_render

env = HitBallEnv(
        robots = ['IIWA'],
        env_configuration = ['default'],    # positions
        controller_configs = {'type':'OSC_POSE', 'interpolation': 'linear', 'ramp_ratio':0.6 },
        gripper_types = ['BatOneGripper'],
        use_camera_obs = not on_screen_render,  # True means controller will be given camera inputs
        reward_shaping = True,   # Whether to offer partial rewards for partial success
        has_renderer = on_screen_render,    # True means you will see the visuals; can't be both on and off screen though.
        has_offscreen_renderer = not on_screen_render,    # Required if you want camera observations for the controller.
        render_camera = 'followrobot',   # name of camera to render (None = default which the user can control)
        render_collision_mesh = False,
        render_visual_mesh = True,
        control_freq = 30,      # Hz of controller being called
        horizon = 300,          # Number of control steps in an episode (not seconds, not time steps, but control steps)
        camera_names = ['followrobot','aboverobot'],   # Cameras to be used for observations to controller
        camera_heights = 160,  # 84 was default, but our ball is small and hard to see
        camera_widths = 160,
        camera_depths = True,   # True if you want RGB-D cameras
        # There are more optional args, but I didn't think them relevant.
    )

# learn
agent = RecurrentPPO("MultiInputLstmPolicy", env, verbose=1)
#print(agent.policy)
agent.load(model_filename)


class MakeVideoCallback(BaseCallback):
    def __init__(self, fname, camname, fps=30, verbose=0):
        super().__init__(verbose)
        self.fname = fname
        self.camname = camname
        self.fps = fps
    def _on_training_start(self):
        print('Vid start')
        self.vid = imageio.get_writer(self.fname, fps=self.fps)
    def _on_training_end(self):
        print('Vid end')
        self.vid.close()
    def _on_step(self):
        im = np.flipud(self.model.env.venv.envs[0].env.sim.render(height=512, width=1024, camera_name=self.camname))
        self.vid.append_data(im)
        if self.num_timesteps >= self.locals['total_timesteps']:
            print('Did all timesteps in an episode')
            return False
        return True

# Never got this to work. Deep down in the bowels it does things differently.
#evaluate_policy(agent, agent.env, 1, render=False, callback=callback)
# So instead we will "learn" for one cycle, but really we're just recording the rollout.
agent.learn(env.horizon, callback=MakeVideoCallback('rollout.mp4', 'followrobot', fps=env.control_freq))


