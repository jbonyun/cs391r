#! /usr/bin/env python3

import ipdb
import math

from robosuite.models import MujocoWorldBase

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

# Insert a ball
BALL_MASS = 0.0027    # kg, official ping pong ball mass
BALL_RADIUS = 0.02    # m, a ping pong ball is 40mm in diameter = 20mm radius
BALL_VELOCITY = -0.2  # m/s, chosen to meet our simulation goals
BALL_VOLUME = BALL_RADIUS**3 * math.pi  # m^3
BALL_DENSITY = BALL_MASS / BALL_VOLUME  # kg/m^3
SHOOTER_FORCE = BALL_VELOCITY * BALL_MASS / float(world.option.get('timestep')) # N
from robosuite.models.objects import BallObject
sphere = BallObject(
    name='ball',
    size=[BALL_RADIUS],
    rgba=[0, 0.5, 0.5, 1],
    solref=[-10000., -7.],  # set bouncyness as negative numbers. first is stiffness, second is damping.
    density=BALL_DENSITY,
    ).get_obj()
sphere.set('pos', '0.5 0 1.0')  # table is something under z=1, so z=1 will be above the table
world.worldbody.append(sphere)

# Create the force the propels the ball to its initial velocity
import xml.etree.ElementTree as ET
sphere_shooter = ET.Element('general', attrib={'name': 'shooter', 'site': 'ball_default_site'})
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
  if i == 0: sim.data.ctrl[model.actuator_name2id('shooter')] = SHOOTER_FORCE  # Initial force only happens on frame 0
  sim.step()
  viewer.render()
