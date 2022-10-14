#! /usr/bin/env python3

import ipdb
import math
import numpy as np

from robosuite.wrappers.gym_wrapper import GymWrapper

from hit_ball_env import HitBallEnv

# For gym use, I don't think we can do any onscreen rendering.
on_screen_render = False

env = HitBallEnv(
        robots = ['IIWA'],
        env_configuration = ['default'],    # positions
        controller_configs = {'type':'OSC_POSE', 'interpolation': 'linear', 'ramp_ratio':0.6 },
        gripper_types = ['BatOneGripper'],
        use_camera_obs = not on_screen_render,  # True means controller will be given camera inputs
        reward_shaping = True,   # Whether to offer partial rewards for partial success
        has_renderer = on_screen_render,    # True means you will see the visuals; can't be both on and off screen though.
        has_offscreen_renderer = not on_screen_render,    # Required if you want camera observations for the controller.
        render_camera = 'underrobotleft',   # name of camera to render (None = default which the user can control)
        render_collision_mesh = False,
        render_visual_mesh = True,
        control_freq = 30,      # Hz of controller being called
        horizon = 300,          # Number of control steps in an episode (not seconds, not time steps, but control steps)
        camera_names = ['underrobotleft', 'underrobotright', 'aboverobot'],   # Cameras to be used for observations to controller
        camera_heights = 320, #84,  # 84 was default, but our ball is small and hard to see
        camera_widths = 320, #84,
        camera_depths = True,   # True if you want RGB-D cameras
        # There are more optional args, but I didn't think them relevant.
    )

HIGH_D_STATE = ['underrobotleft_image', 'underrobotright_image', 'aboverobot_image', 'underrobotleft_depth', 'underrobotright_depth', 'aboverobot_depth', 'robot0_proprio-state']
LOW_D_STATE = ['object-state', 'robot0_proprio-state']

gym = GymWrapper(env, LOW_D_STATE)


NUM_EPISODES = 1
for i_episode in range(NUM_EPISODES):
    observation = gym.reset()
    i_step = 0
    while True:
        if i_step % 30 == 1:
            print('ball qpos', np.round(env.sim.data.get_body_xpos('ball0_main'), 4))
            print('ball qvel', np.round(env.sim.data.get_body_xvelp('ball0_main'), 4))
        # Choose an action. I'm doing this only on the first timestep and then repeating forever.
        if i_step == 0:
            action = np.random.uniform(-0.25, 0.25, (6,))  # Some random position
            #action = np.zeros((6,))  # What does zero action mean? Seems to stay still from starting pos.
        # Execute the action and see result.
        observation,reward,done,misc = gym.step(action)
        #ipdb.set_trace()
        if reward > 0.1:
            print('Big reward!', np.round(reward,2))
        # Stop if done.
        if done:
            print("Episode finished after {} timesteps".format(i_step + 1))
            break
        i_step = i_step + 1
