#! /usr/bin/env python3

import ipdb
import math
import numpy as np
import xml.etree.ElementTree as ET

from robosuite.models import MujocoWorldBase
from robosuite.utils.mjcf_utils import array_to_string

from gripper import BatOneGripper
from ball_spawn import BallSpawner, BoxInSpace, CircleInSpace, SpeedSpawner, BallTrajectory

world = MujocoWorldBase()

from robosuite.models.base import MujocoXML
xml = MujocoXML('empty_space.xml')
world.merge(xml)

def remove_xml_element_named(root: ET.Element, name: str):
    """Recursively search an xml Element object and remove any children named `name`"""
    for child in root:
        if child.get('name') == name:
            root.remove(child)
        else:
            remove_xml_element_named(child, name)


# Put a robot in the world
from robosuite.models.robots import IIWA
mujoco_robot = IIWA()
mujoco_robot.set_base_xpos([0, 0, 0])  # Robot is at 0,0,0 in world coords.
# Remove cameras that come bundled with the robot arm, because I don't want them.
remove_xml_element_named(mujoco_robot.root, 'robot0_eye_in_hand')
remove_xml_element_named(mujoco_robot.root, 'robot0_robotview')
# Add a gripper
gripper = BatOneGripper()
mujoco_robot.add_gripper(gripper)
world.merge(mujoco_robot)


# Disable gravity by setting it to zero acceleration in x/y/z
world.option.attrib['gravity'] = '0 0 0'

from robosuite.models.objects import BallObject
class Ball(BallObject):
    MASS = 0.0027    # kg, official ping pong ball = 2.7g
    RADIUS = 0.02    # m, official ping pong ball = 40mm diameter
    ball_count = 0   # global count of the number of Ball objects we have, so they can have unique names.
    def __init__(self, world, trajectory):
        self.index = Ball.ball_count  # Remember and update the global object count.
        Ball.ball_count = self.index + 1
        super().__init__(
            name='ball{}'.format(self.index),
            size=[Ball.RADIUS],
            rgba=[0, 0.5, 0.5, 1],
            solref=[-10000., -7.],  # set bouncyness as negative numbers. first is stiffness, second is damping.
            density=self.density(),
            )
        self.trajectory = trajectory
        self.get_obj().set('pos', array_to_string(self.trajectory.origin))  # Set initial position.
        self.shooter = ET.Element('general', attrib={'name': 'ball{}_shooter'.format(self.index), 'site': 'ball{}_default_site'.format(self.index), 'gear': array_to_string(self.trajectory.velocity_vector) + ' 0 0 0'})
        self.actuator_id = len(world.actuator)  # Assumes that assigne actuator id is where it appears in the xml list.
        world.actuator.append(self.shooter)     # Add the shooter to the xml.
        world.worldbody.append(self.get_obj())  # Add the ball object to the xml.
    def volume(self):
        """Volume of the ball, m^3"""
        return Ball.RADIUS**3 * math.pi * 4./3.
    def density(self):
        """Density of the ball, kg/m^3"""
        return Ball.MASS / self.volume()
    def shooter_force(self):
        """How hard (in Newtons) the initial force must push for one frame time to instill initial velocity."""
        return self.trajectory.speed * Ball.MASS / float(world.option.get('timestep'))
    def set_shooter_control(self, sim, set_to=None):
        """Apply the shooter_force to the actuator that will push this ball"""
        sim.data.ctrl[self.actuator_id] = (self.shooter_force() if set_to is None else set_to)

spawner = BallSpawner()
spawner.src = BoxInSpace([5, 0, 2], None, 2, 4, 3)
spawner.tgt = CircleInSpace((0,0,1), (1,0,0), (0,1,0), 1.*math.pi, 1.)
spawner.spd = SpeedSpawner(1.0, 2.0)

#ball = Ball(world, spawner.random())
NUM_BALLS = 25
balls = [Ball(world, spawner.random()) for bi in range(NUM_BALLS)]

# Convert the xml that robosuite has managed into a real mujoco object
model = world.get_model(mode="mujoco_py")

#ipdb.set_trace()

from mujoco_py import MjSim, MjViewer
sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

#ipdb.set_trace()

for i in range(10000):
    # Choose a constant joint effort for each robot joint at random. Only on first frame.
    if i == 0:
        for joint_i in range(7):
            sim.data.ctrl[sim.model.actuator_name2id('robot0_torq_j{}'.format(joint_i+1))] = np.random.normal(0, 0.2)
    # Set ball actuation force. Zero except for first frame.
    for ball in balls:
        ball.set_shooter_control(sim, ball.shooter_force() if i == 0 else 0.)
    sim.step()
    viewer.render()
