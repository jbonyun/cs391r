#! /usr/bin/env python3

import ipdb
import math
import numpy as np
from matplotlib import pyplot
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import sys

from hit_ball_env import HitBallEnv

import torch
torch.cuda.empty_cache()

model_filename = sys.argv[1]

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
        render_camera = 'followball',   # name of camera to render (None = default which the user can control)
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
menv = Monitor(env)
agent = RecurrentPPO("MultiInputLstmPolicy", menv, verbose=1)
print(agent.policy)
agent.load(model_filename)


def callback(locs, globs):
    #print('callback', locs.keys(), globs.keys())
    #print('  ', locs['observations'].keys())
    # It's wrapped really deep. This is hardcoded unravelling of the wrapping.
    im = locs['model'].env.venv.envs[0].env.sim.render(height=512, width=1024, camera_name='followrobot')
    pyplot.imshow(np.flipud(im))
    pyplot.draw()
    pyplot.pause(0.001)
    #ipdb.set_trace()

#ipdb.set_trace()
evaluate_policy(agent, agent.env, 1, render=False, callback=callback)

print('Done')
sys.exit(0)

NUM_EPISODES = 1
for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    i_step = 0
    while True:
        # Update visuals
        if env.viewer is not None: env.render()
        if i_step % 30 == 1:
            print('ball qpos', np.round(env.sim.data.get_body_xpos('ball0_main'), 4))
            print('ball qvel', np.round(env.sim.data.get_body_xvelp('ball0_main'), 4))
        # Choose an action. I'm doing this only on the first timestep and then repeating forever.
        if i_step == 0:
            action = np.random.uniform(-0.25, 0.25, (6,))  # Some random position
            #action = np.zeros((6,))  # What does zero action mean? Seems to stay still from starting pos.
        #ipdb.set_trace()
        # Execute the action and see result.
        observation, reward, done, info = env.step(action)
        if matplotlib_display and env.viewer is None and i_step % 5 == 1:
            plot_observations(observation, env.camera_names)
        if reward > 0.1:
            print('Big reward!', np.round(reward,2))
        # Stop if done.
        if done:
            print("Episode finished after {} timesteps".format(i_step + 1))
            break
        i_step = i_step + 1
