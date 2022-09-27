#! /usr/bin/env python3

import ipdb
import math

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
    def __init__(self, trajectory):
        super().__init__(
            name='ball',
            size=[Ball.RADIUS],
            rgba=[0, 0.5, 0.5, 1],
            solref=[-10000., -7.],  # set bouncyness as negative numbers. first is stiffness, second is damping.
            density=self.density(),
            )
        self.trajectory = trajectory
        self.get_obj().set('pos', array_to_string(self.trajectory.origin))
    def volume(self):
        return Ball.RADIUS**3 * math.pi * 4./3.  # m^3
    def density(self):
        return Ball.MASS / self.volume()  # kg/m^3
    def create_shooter(self):
        import xml.etree.ElementTree as ET
        return ET.Element('general', attrib={'name': 'shooter', 'site': 'ball_default_site', 'gear': array_to_string(self.trajectory.velocity_vector) + ' 0 0 0'})
    def init_force(self):
        return self.trajectory.speed * Ball.MASS / float(world.option.get('timestep')) # N

spawner = BallSpawner()
spawner.src = BoxInSpace([5, 0, 2], None, 2, 4, 3)
spawner.tgt = CircleInSpace((0,0,1), (1,0,0), (0,1,0), 1.*math.pi, 1.)
spawner.spd = SpeedSpawner(1.0, 3.0)

ball = Ball(spawner.random())
world.worldbody.append(ball.get_obj())

# Create the force the propels the ball to its initial velocity
sphere_shooter = ball.create_shooter()
world.actuator.append(sphere_shooter)

# Convert the xml that robosuite has managed into a real mujoco object
model = world.get_model(mode="mujoco_py")

#ipdb.set_trace()

from mujoco_py import MjSim, MjViewer
sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  if sim.data.ctrl is not None: sim.data.ctrl[:] = 0  # I think this makes all control efforts 0
  if i == 0: sim.data.ctrl[model.actuator_name2id('shooter')] = ball.init_force()  # Initial force only happens on frame 0
  sim.step()
  viewer.render()
