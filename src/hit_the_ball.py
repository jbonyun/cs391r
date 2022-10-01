#! /usr/bin/env python3

import ipdb
import math
import numpy as np


from hit_ball_env import HitBallEnv

env = HitBallEnv(
        robots = ['IIWA'],
        env_configuration = ['default'],    # positions
        controller_configs = {'type':'OSC_POSE', 'interpolation': 'linear', 'ramp_ratio':0.6 },
        gripper_types = ['BatOneGripper'],
        #initialization_noise = None,
        use_camera_obs = False,  # True means controller will be given camera inputs
        has_renderer = True,    # True means you will see the visuals; can't be both on and off screen though.
        has_offscreen_renderer = False,    # Required if you want camera observations for the controller.
        render_camera = None,   # name of camera to render (None = default which the user can control)
        render_collision_mesh = False,
        render_visual_mesh = True,
        control_freq = 30,      # Hz of controller being called
        horizon = 300,          # Number of control steps in an episode (not seconds, not time steps, but control steps)
        camera_names = ['all-robotview'],   # Cameras to be used for observations to controller
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
        # Update visuals (and maybe the off-screen render used for control observations?)
        env.render()
        print('ball qvel', np.round(env.sim.data.get_body_xvelp('ball0_main'), 2))
        # Choose an action. I'm doing this only on the first timestep and then repeating forever.
        if i_step == 0:
            #action = np.random.uniform(-0.05, 0.05, (6,))
            action = np.zeros((6,))  # What does zero action mean?
        env.ball.set_shooter_control(env.sim, None if i_step == 0 else 0.)
        #ipdb.set_trace()
        # Execute the action and see result.
        observation, reward, done, info = env.step(action)
        # Stop if done.
        if done:
            print("Episode finished after {} timesteps".format(i_step + 1))
            break
        i_step = i_step + 1
