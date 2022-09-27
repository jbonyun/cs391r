#! /usr/bin/env python3

import ipdb
import math
import xml.etree.ElementTree as ET

from robosuite.models import MujocoWorldBase
from robosuite.utils.mjcf_utils import array_to_string

from ball_spawn import BallSpawner, BoxInSpace, CircleInSpace, SpeedSpawner, BallTrajectory

world = MujocoWorldBase()

from robosuite.models.base import MujocoXML
xml = MujocoXML('empty_space.xml')
world.merge(xml)

# Put a robot in the world
from robosuite.models.robots import IIWA
mujoco_robot = IIWA()
mujoco_robot.set_base_xpos([0, 0, 0])  # Robot is at 0,0,0 in world coords.
world.merge(mujoco_robot)

# Disable gravity by setting it to zero acceleration in x/y/z
world.option.attrib['gravity'] = '0 0 0'

from robosuite.models.objects import BallObject
class Ball(BallObject):
    MASS = 0.0027    # kg, official ping pong ball = 2.7g
    RADIUS = 0.02    # m, official ping pong ball = 40mm diameter
    ball_count = 0   # global count of the number of Ball objects we have, so they can have unique names.
    def __init__(self, world, trajectory):
        self.index = Ball.ball_count
        Ball.ball_count = self.index + 1
        super().__init__(
            name='ball{}'.format(self.index),
            size=[Ball.RADIUS],
            rgba=[0, 0.5, 0.5, 1],
            solref=[-10000., -7.],  # set bouncyness as negative numbers. first is stiffness, second is damping.
            density=self.density(),
            )
        self.trajectory = trajectory
        self.get_obj().set('pos', array_to_string(self.trajectory.origin))
        self.shooter = ET.Element('general', attrib={'name': 'ball{}_shooter'.format(self.index), 'site': 'ball{}_default_site'.format(self.index), 'gear': array_to_string(self.trajectory.velocity_vector) + ' 0 0 0'})
        self.actuator_id = len(world.actuator)
        world.actuator.append(self.shooter)
        world.worldbody.append(self.get_obj())
    def volume(self):
        return Ball.RADIUS**3 * math.pi * 4./3.  # m^3
    def density(self):
        return Ball.MASS / self.volume()  # kg/m^3
    def shooter_force(self):
        return self.trajectory.speed * Ball.MASS / float(world.option.get('timestep')) # N
    def set_shooter_control(self, sim):
        sim.data.ctrl[self.actuator_id] = self.shooter_force()

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

for i in range(10000):
  if sim.data.ctrl is not None: sim.data.ctrl[:] = 0  # I think this makes all control efforts 0
  if i == 0:
    # Set initial forces to create starting velocities. Only happens on frame 0.
    for ball in balls:
        ball.set_shooter_control(sim)
  sim.step()
  viewer.render()
