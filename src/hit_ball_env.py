from collections import OrderedDict
import ipdb
import math
import random

import numpy as np
from xml.etree.ElementTree import Element

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models import MujocoWorldBase
from robosuite.models.tasks import Task
from robosuite.robots.robot import Robot
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.camera_utils import get_real_depth_map

from gripper.cylindrical_bat import BatOneGripper
from robosuite.models.grippers import GRIPPER_MAPPING
GRIPPER_MAPPING['BatOneGripper'] = BatOneGripper

from ping_pong_ball import PingPongBall
from ball_spawn import BallSpawner, BoxInSpace, CircleInSpace, SpeedSpawner, BallTrajectory, OneOfN
from deterministic_sampler import DeterministicSampler

from gym.spaces import Box
from gym import spaces

class HitBallEnv(SingleArmEnv):
    """
    A task to make contact with a ball.
    A work-in-progress. Copied from the Stack built-in task, and gradually modifying.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object information in the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_color=True,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        np.random.seed()
        random.seed()

        self.ball_radius = 0.10   # in meters; radius not diameter; ping pong is 0.02

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.spawner = BallSpawner()
        ball_spawn_type = 'center'   # full, one, two, center
        TARGET_RADIUS = 0.40
        if ball_spawn_type == 'full':
            self.spawner.src = BoxInSpace([2.5, 0, 0.2], None, 0.5, 0.5, 0.5)
            self.spawner.tgt = CircleInSpace((0,0,0), (1,0,0), (0,1,0), 1.*math.pi, 0.8)
            self.spawner.spd = SpeedSpawner(0.5, 0.7)
        elif ball_spawn_type == 'one':
            self.spawner.src = BoxInSpace([2.5, 0, 0.2], None, 0.0, 0.0, 0.0)  # No randomness
            self.spawner.tgt = CircleInSpace((0,-0.5,0), (1,0,0), (0,1,0), 1.*math.pi, TARGET_RADIUS)
            self.spawner.spd = SpeedSpawner(0.7, 0.7)  # No randomness
        elif ball_spawn_type == 'center':
            self.spawner.src = BoxInSpace([2.5, 0, 0], None, 0.0, 0.0, 0.0)  # No randomness
            self.spawner.tgt = CircleInSpace((0,0,0), (1,0,0), (0,1,0), 1.*math.pi, TARGET_RADIUS)
            self.spawner.spd = SpeedSpawner(0.7, 0.7)  # No randomness
        elif ball_spawn_type == 'two':
            self.spawner.src = BoxInSpace([2.5, 0, 0.2], None, 0.0, 0.0, 0.0)  # No randomness
            self.spawner.tgt = OneOfN([
                CircleInSpace((0,-0.5,0), (1,0,0), (0,1,0), 2.*math.pi, TARGET_RADIUS),
                CircleInSpace((0,0.5,0), (1,0,0), (0,1,0), 2.*math.pi, TARGET_RADIUS)
            ])
            self.spawner.spd = SpeedSpawner(0.7, 0.7)  # No randomness
        else:
            raise Exception('Bad ball_spawn_type: ' + str(ball_spawn_type))

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=None, #"default",  # default is a mounting cart, which is silly in space
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
        self.camera_color=camera_color
        self.contact_prev_step = False
        self.contact_during_step = False
        self.has_collided = False
        self.has_been_close = False
        self.record_observer = None
        self.format_spaces()
        self.metadata = {'render.modes': ['human']}

    def grow_target_radius(self, num_eps, param_set):
        if param_set is None: return
        # Example: param_set = (0.1, 0.4, 20000)
        min_radius, max_radius, halflife_eps = param_set
        new_val = max_radius - (max_radius - min_radius) * 0.5**(float(num_eps) / float(halflife_eps))
        print('Growing radius of target to {:.5f}'.format(new_val))
        if isinstance(self.spawner.tgt, OneOfN):
            for n in self.spawner.tgt.things:
                n.radius = new_val
        elif isinstance(self.spawner.tgt, CircleInSpace):
            self.spawner.tgt.radius = new_val
        else:
            print('Dont know how to grow variance')
    def shrink_ball(self, num_eps, param_set):
        if param_set is None: return
        # Example: param_set = (0.02, 0.10, 20000)
        min_size, max_size, halflife_eps = param_set
        self.ball_radius = min_size + (max_size - min_size) * 0.5**(float(num_eps) / float(halflife_eps))
        print('Ball radius set to {:.5f}'.format(self.ball_radius))

    def format_observation(self, state):
        # if no camera, do nothing
        if not self.use_camera_obs:
            #return state
            return { 'objects': state['object-state'],
                     'joints': state['robot0_proprio-state']
            }
        # else return image + joints
        obs_cam = self.render_camera
        if self.camera_depths:
            # Reprocess depth to have a max distance much closer than Mujoco's 145m.
            # This matters bc Mujoco normalizes between 0 and 1, but since we have no
            # background, it includes lots of points at 145m in the normalizing. Which
            # squashes our relevant depths into nothingness.
            dp = state[obs_cam+'_depth']
            rd = get_real_depth_map(self.sim, dp)
            MAX_DISTANCE = 6.  # In meters
            rdcap = np.minimum(rd, MAX_DISTANCE)
            dp = 1. - (rdcap / MAX_DISTANCE)
            dp = (dp*255).astype(np.uint8)
            if self.camera_color:
                im = state[obs_cam+'_image']
            else:
                # Grayscale-D
                im = np.expand_dims(np.mean(state[obs_cam+'_image'], axis=2), axis=2).astype(np.uint8)
            #print('fmtobs im range',np.max(im), np.min(im), ' dp range',np.max(dp), np.min(dp))
            concat_image = np.concatenate((im,dp), axis=2)
        else:
            concat_image = state[obs_cam + '_image']
        return { "image":concat_image.astype(np.uint8),
                 "joints":state["robot0_proprio-state"]
        }

    def format_spaces(self):
        self.action_space = Box(low=-1., high=1., shape=(6,), dtype=np.float32)

        if self.use_camera_obs:
            num_channels = 3 if self.camera_color else 1
            num_channels += (1 if self.camera_depths else 0)
            self.observation_space =  spaces.Dict({
                "image": Box(low=0, high=255, shape=(self.camera_widths[0], self.camera_heights[0], num_channels), dtype=np.uint8),
                "joints": Box(low=-1., high=1., shape=(28,), dtype=np.float32)
            })
        elif self.use_object_obs:
            self.observation_space =  spaces.Dict({
                "objects": Box(low=-1., high=1., shape=(6,), dtype=np.float32),
                "joints": Box(low=-1., high=1., shape=(28,), dtype=np.float32)
            })

    def set_record_observer(self, camname):
        self.record_observer = camname

    def step(self, action):
        """
        Overload the base class just to trigger the ball to shoot
        """
        self.contact_prev_step = self.contact_during_step
        self.contact_during_step = False  # Reset contact detection
        self.ball.set_shooter_control(self.sim, None if self.timestep == 0 else 0.)
        obs, reward, done, info = super().step(action)
        fmt_obs = self.format_observation(obs)
        if self.record_observer is not None and self.record_observer in self.camera_names:
            # Don't want to do this work unless we are told we will use it.
            if self.record_observer == self.render_camera:
                # Special case where we will render it exactly as seen by algo
                obs_im = fmt_obs['image']
                if obs_im.shape[2] == 1:
                    info['observer'] = np.flipud(obs_im)
                elif obs_im.shape[2] == 2:
                    info['observer'] = np.concatenate((np.flipud(obs_im[:,:,0]), np.flipud(obs_im[:,:,-1])), axis=0)
                elif obs_im.shape[2] == 3:
                    info['observer'] = np.flipud(obs_im)
                elif obs_im.shape[2] == 4:
                    info['observer'] = np.concatenate((np.flipud(obs_im[:,:,0:3]), np.tile(np.expand_dims(np.flipud(obs_im[:,:,-1]), axis=2),(1,1,3))), axis=0)
            else:
                # Render it pretty
                info['observer'] = np.flipud(self.sim.render(height=512, width=1024, camera_name=self.record_observer))
        info['success'] = self.has_collided
        return fmt_obs, reward, done, info

    def reset(self):
        """
        Overload the base class to process to observations.
        """
        obs = super().reset()
        if not self.has_collided:
            print('No contact')  # So we conclusively know it missed making contact
        self.contact_prev_step = False
        self.contact_during_step = False
        self.has_collided = False
        self.has_been_close = False
        self.record_observer = None
        return self.format_observation(obs)

    def reward(self, action):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward_pieces = self.staged_rewards()
        if self.reward_shaping:
            reward = sum(reward_pieces)
        else:
            reward = reward_pieces[-1]  # Last one is contact

        if self.reward_scale is not None:
            reward *= self.reward_scale   # WTF? Why divide by 2?  / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for direction of motion
                - (float): reward for proximity
                - (float): reward for contact
        """
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        ball_pos = self.sim.data.body_xpos[self.ball_body_id]

        reward_direction_scale = 0.0 #1.0
        if reward_direction_scale != 0.0:
            # Calculate direction from gripper to ball
            direction = ball_pos - gripper_site_pos

            # normalize. Account for 0 norm
            norm = np.linalg.norm(direction)
            unit_direction = direction / norm

            # find gripper velocity. Norm it
            gripper_velocity = self.sim.data.site_xvelp[self.robots[0].eef_site_id]
            gripper_velocity_norm = np.linalg.norm(gripper_velocity)
            unit_gripper_velocity = gripper_velocity / gripper_velocity_norm

            # take dot product. IF either norm is 0, thne the result is undefined. Use 0.0 as reward
            reward_direction = np.dot(unit_direction,unit_gripper_velocity)
            if norm == 0.0 or gripper_velocity_norm == 0.0:
                reward_direction = 0.0
            reward_direction *= reward_direction_scale
        else:
            reward_direction = 0.

        # Proximity to the ball
        # Was from the stacking task; scale 0.25 to 20
        prox_dist_scale = 2.0 #10.0  # Seems to be in meters, higher means sharper tanh slope
        prox_mult_scale = 0.005 #0.25
        dist = np.linalg.norm(gripper_site_pos - ball_pos)
        r_prox = (1 - np.tanh(prox_dist_scale * dist)) * prox_mult_scale

        # Ball position being in starting direction.
        # To encourage it to hit the ball back in starting direction (+x)
        r_ball_x = 0.
        ball_pos = self.sim.data.body_xpos[self.ball_body_id]
        ball_x = ball_pos[0]
        CLOSE_THRESHOLD = 1.5  # m
        if self.has_been_close and ball_x > CLOSE_THRESHOLD:
            # Was inside thresh, is now outside thresh: it bounced off something 
            ball_x_mult = 0.15
            r_ball_x = (ball_x - CLOSE_THRESHOLD) * ball_x_mult
        elif ball_x < CLOSE_THRESHOLD:
            self.has_been_close = True

        # give big points for contact
        r_contact = 0.
        #made_contact = self.check_contact(self.ball, self.robots[0].gripper)
        made_contact = self.contact_during_step
        if made_contact:
            if not self.has_collided:
                if not self.check_contact(self.ball, self.robots[0].gripper):
                    print('Contact! (between control steps)')
                else:
                    print('Contact!')
                r_contact = 1.0
                self.has_collided = True
            elif self.contact_prev_step:
                print('No reward contact (continued)')
            else:
                print('No reward contact (repeated discrete)')

        reward_tuple = reward_direction, r_prox, r_ball_x, r_contact
        #print('rewards', reward_tuple)
        return reward_tuple

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        mujoco_arena = MujocoWorldBase()
        from robosuite.models.base import MujocoXML
        xml = MujocoXML('empty_space.xml')
        mujoco_arena.merge(xml)

        super()._load_model()

        mujoco_objects = MujocoWorldBase()
        # Make a ball. It merges itself to the world you give it.
        self.ball = PingPongBall(self.spawner.random(), 1./self.control_freq, radius=self.ball_radius)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.ball)
        else:
            self.placement_initializer = DeterministicSampler(
                name="BallSampler",
                mujoco_objects=self.ball,
                ensure_valid_placement=False,
                reference_pos=self.ball.trajectory.origin  # Center around the spawner's chosen place
            )

        self.model = Task(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.ball]
        )

        # Tweak it in ways that can't be done before the merging done in Task class.
        # Disable gravity by setting it to zero acceleration in x/y/z
        self.model.root.find('option').attrib['gravity'] = '0 0 0'
        self.model.root.find('option').attrib['density'] = '0'
        self.model.root.find('option').attrib['viscosity'] = '0'
        self.model.actuator.append(self.ball.create_shooter())
        # Add a focal point for the camera
        site_el = Element('body', attrib={'name':'observertarget', 'pos': '0.5 0 0.5'})
        self.model.worldbody.append(site_el)
        # DEBUGING: this is what will be instantiated by the mujoco, so a good time to print it out to review
        #for l in self.model.get_xml().split('\n'): print(l)


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.ball_body_id = self.sim.model.body_name2id(self.ball.get_obj().get('name')) #root_body)
        self.ball.actuator_id = self.sim.model.actuator_name2id('ball{}_shooter'.format(self.ball.name_suffix))

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position of the ball
            @sensor(modality=modality)
            def ball_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.ball_body_id])

            # distance from gripper to ball
            @sensor(modality=modality)
            def gripper_to_ball(obs_cache):
                return (
                    obs_cache["ball_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "ball_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [ball_pos, gripper_to_ball]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _update_observables(self, force=False):
        """
        Runs after every mujoco timestep (not control timestep).
        A good place to keep track of contact!
        """
        super()._update_observables(force)
        curr_ball_contact = self.check_contact(self.ball, self.robots[0].gripper)
        self.contact_during_step = self.contact_during_step or curr_ball_contact

    def _check_success(self):
        """
        Check if contact has been made

        Returns:
            bool: True if contact is made
        """
        reward_pieces = self.staged_rewards()
        r_contact = reward_pieces[-1]  # last one is contact
        return r_contact > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the ball.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the ball
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.ball)

