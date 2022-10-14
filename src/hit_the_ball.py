#! /usr/bin/env python3

import ipdb
import math
import numpy as np


from hit_ball_env import HitBallEnv

# Control whether to do onscreen or offscreen.
# Can't do both. Can't give robot images if you do onscreen.
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
        camera_heights = 84,
        camera_widths = 84,
        camera_depths = True,   # True if you want RGB-D cameras
        # There are more optional args, but I didn't think them relevant.
    )



NUM_EPISODES = 1
for i_episode in range(NUM_EPISODES):
    if i_episode != 0: observation = env.reset()
    i_step = 0
    while True:
        # Update visuals
        if env.viewer is not None: env.render()
        if i_step % 30 == 1:
            print('ball qpos', np.round(env.sim.data.get_body_xpos('ball0_main'), 4))
            print('ball qvel', np.round(env.sim.data.get_body_xvelp('ball0_main'), 4))
        # Choose an action. I'm doing this only on the first timestep and then repeating forever.
        if i_step == 0:
            #action = np.random.uniform(-0.05, 0.05, (6,))
            action = np.zeros((6,))  # What does zero action mean?
        env.ball.set_shooter_control(env.sim, None if i_step == 0 else 0.)
        #ipdb.set_trace()
        # Execute the action and see result.
        observation, reward, done, info = env.step(action)
        if reward > 0.1:
            print('Big reward!', np.round(reward,2))
        # Stop if done.
        if done:
            print("Episode finished after {} timesteps".format(i_step + 1))
            break
        i_step = i_step + 1
