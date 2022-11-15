#! /usr/bin/env python3

import collections
from datetime import datetime
import imageio
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


num_env = 3
control_freq = 15
horizon = 64
video_period = 10

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
        control_freq = control_freq, # Hz of controller being called
        horizon = horizon,           # Number of control steps in an episode (not seconds, not time steps, but control steps)
        camera_names = ['aboverobot', 'followrobot'],   # Cameras to be used for observations to controller
        camera_heights = 84,  # 84 was default, but our ball is small and hard to see
        camera_widths = 84,
        camera_depths = True,   # True if you want RGB-D cameras
        camera_color = False,
        # There are more optional args, but I didn't think them relevant.
    )


class SaveAfterEpisodeCallback(BaseCallback):
    def __init__(self, episode_len_steps, num_envs, rollouts_per_save, rollouts_per_bigsave=None):
        super().__init__()
        self.ep_len = episode_len_steps
        self.num_envs = num_envs
        self.rollouts_per_save = rollouts_per_save
        self.rollouts_per_bigsave = rollouts_per_bigsave
    def on_rollout_end(self):
        if self.rollouts_per_bigsave is not None and self.num_timesteps % (self.ep_len*self.num_envs*self.rollouts_per_bigsave) == 0:
            print('BIGSAVE')
            self.model.save('save_checkpoint_{}.model'.format(self.num_timesteps))
        if self.num_timesteps % (self.ep_len*self.num_envs*self.rollouts_per_save) == 0:
            print('Episode+Rollout end +Save')
            self.model.save('save_checkpoint.model')
            print('Checkpoint saved')
        elif self.num_timesteps % horizon == 0:
            print('Episode+Rollout end')
        else:
            print('Rollout end')

    def _on_step(self):
        return True

class ActionNormPrintCallback(BaseCallback):
    def __init__(self, episode_len_steps, num_envs):
        super().__init__()
        self.ep_len = episode_len_steps
        self.num_envs = num_envs
    def _on_step(self):
        if (self.num_timesteps / self.num_envs) % self.ep_len == 0:
            n = np.linalg.norm(self.locals['actions'])
            print('Action norm: {:.4}'.format(n))
        return True

class RewardPrintCallback(BaseCallback):
    def __init__(self, episode_len_steps, history_len_episodes, num_envs, log_fname=None, log_ep=None):
        super().__init__()
        self.d = collections.deque([], int(np.ceil(history_len_episodes * episode_len_steps / num_envs)))
        self.ep_len = episode_len_steps
        self.num_envs = num_envs
        self.log_filename = log_fname
        self.log_ep = log_ep
    def _on_step(self):
        self.d.append(self.locals['rewards'])
        if (self.num_timesteps / self.num_envs) % self.ep_len == 0:
            all_data = np.vstack(self.d)
            mean = np.mean(all_data) * self.ep_len  # across both dimensions
            num_ep = np.product(all_data.shape) / self.ep_len
            print('Mean reward over {:d} episodes: {:.4}'.format(int(num_ep), mean))
        if (self.num_timesteps / self.num_envs) % self.log_ep == 0:
            all_data = np.vstack(self.d)
            mean = np.mean(all_data) * self.ep_len  # across both dimensions
            num_ep = np.product(all_data.shape) / self.ep_len
            print('Updating reward log')
            with open(self.log_filename, 'a') as f:
                f.write('{},{},{}\n'.format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), self.num_timesteps, mean))
            print('Mean reward over {:d} episodes: {:.4}'.format(int(num_ep), mean))
        return True
    def set_log_on_n_episodes(self, fname, n_ep):
        self.log_filename = fname
        self.log_ep = n_ep


class MakeVideoCallback(BaseCallback):
    def __init__(self, fname, camname, envs, fps=30, verbose=0, num_envs=1, rollout_period=1):
        super().__init__(verbose)
        self.fname = fname
        self.camname = camname
        self.envs = envs
        self.fps = fps
        self.num_envs = num_envs
        self.rollout_period = rollout_period
        self.vid = []
        self.recording_this_episode = False
        self.rollout_count = 0
    def _on_rollout_start(self):
        if self.rollout_count % self.rollout_period == 0:
            self.start_vid()
        self.rollout_count += 1
    def _on_step(self):
        if self.recording_this_episode:
            for i in range(self.num_envs):
                if 'observer' in self.locals['infos'][i]:
                    im = self.locals['infos'][i]['observer']
                    self.vid[i].append_data(im)
                else:
                    print('observer image was not in infos')
            if self.locals['n_steps'] >= self.locals['n_rollout_steps'] - 1:
                self.end_vid()
            return True
    def start_vid(self):
        self.recording_this_episode = True
        print('Vid start')
        # Create/open the video files for each env
        for i in range(self.num_envs):
            fname = self.fname.format(i)
            self.vid.append(imageio.get_writer(fname, fps=self.fps))
        # Tell the environments to generate the image we need this time
        self.envs.env_method('set_record_observer', self.camname)
    def end_vid(self):
        self.recording_this_episode = False
        print('Vid end')
        for i in range(self.num_envs):
            self.vid[i].close()
            self.vid[i] = None
        self.vid = []

class VarianceScheduler(BaseCallback):
    def __init__(self, venv, episode_len_steps, eps_per_escal):
        super().__init__()
        self.venv = venv
        self.ep_len = episode_len_steps
        self.eps_per_escal = eps_per_escal
        self.steps_next_escal = self.eps_per_escal * self.ep_len
    def on_rollout_end(self):
        if self.num_timesteps >= self.steps_next_escal:
            self.steps_next_escal = self.num_timesteps + self.eps_per_escal * self.ep_len
            num_eps = self.num_timesteps / self.ep_len
            venv.env_method('grow_variance', num_eps)
            venv.env_method('shrink_ball', num_eps)
    def _on_step(self):
        pass


if __name__ == '__main__':
    # Create vectorized environments
    if num_env > 1:
        venv = SubprocVecEnv([make_env]*num_env)
    else:
        venv = DummyVecEnv([make_env]*1)


    # Prepare agent
    load_filename = sys.argv[1] if len(sys.argv) > 1 else None

    if load_filename is not None:
        agent = RecurrentPPO.load(load_filename, venv, verbose=1, n_steps=horizon)
    else:
        policy_kwargs = dict(features_extractor_class=CombinedExtractorDilatedCNN)
        agent = RecurrentPPO("MultiInputLstmPolicy", venv, verbose=1, n_steps=horizon, policy_kwargs=policy_kwargs)
    print(agent.policy)

    # Prepare callbacks
    vid = MakeVideoCallback('rollout_{}.mp4', 'followrobot', venv, fps=control_freq, num_envs=num_env, rollout_period=video_period)
    rew = RewardPrintCallback(horizon, num_env*25, num_env, 'reward.log', num_env*10)
    varsched = VarianceScheduler(venv, horizon, 64)
    savemod = SaveAfterEpisodeCallback(horizon,num_env,3,500)
    callbacks = [savemod, rew, vid, varsched]

    # learn
    expected_fps = {1:50, 2:60, 3:65, 4:80, 5:100, 6: 105, 24:225, 36:225}.get(num_env, 105)
    approx_seconds_to_run = 60*60*24
    steps_to_run = expected_fps * approx_seconds_to_run

    agent.learn(steps_to_run, callback=CallbackList(callbacks))

