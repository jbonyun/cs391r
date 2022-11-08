#! /usr/bin/env python3

import ipdb
import math
import numpy as np
import xml.etree.ElementTree as ET

from robosuite.models import MujocoWorldBase
from robosuite.utils.mjcf_utils import array_to_string

from gripper.cylindrical_bat import BatOneGripper
from ball_spawn import BallSpawner, BoxInSpace, CircleInSpace, SpeedSpawner, BallTrajectory
from ping_pong_ball import PingPongBall

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
#world.option.attrib['gravity'] = '0 0 0'

spawner = BallSpawner()
use_random_spawn = True  # False means a deterministic point and path, for testing.
if use_random_spawn:
    spawner.src = BoxInSpace([2.5, 0, 0], None, 0.5, 0.5, 0.5)
    spawner.tgt = CircleInSpace((0,0,0), (1,0,0), (0,1,0), 1.*math.pi, 0.8)
    spawner.spd = SpeedSpawner(0.5, 0.7)
else:
    spawner.src = BoxInSpace([2.5, 0, 0], None, 0.0, 0.0, 0.0)  # No randomness
    #spawner.tgt = CircleInSpace((0,0,0), (1,0,0), (0,1,0), 1.*math.pi, 0.0)  # No randomness
    spawner.tgt = OneOfN([CircleInSpace((0,-0.5,0), (1,0,0), (0,1,0), 1.*math.pi, 0.0),
                               CircleInSpace((0,0.5,0), (1,0,0), (0,1,0), 1.*math.pi, 0.0)])
    spawner.spd = SpeedSpawner(0.7, 0.7)  # No randomness

NUM_BALLS = 25
balls = [PingPongBall(spawner.random(), timestep, str(bi)) for bi in range(NUM_BALLS)]
for ball in balls:
    world.actuator.append(ball.create_shooter())

site_el = ET.Element('body', attrib={'name':'observertarget', 'pos': '0.5 0 0.5'})
world.worldbody.append(site_el)

# Convert the xml that robosuite has managed into a real mujoco object
model = world.get_model(mode="mujoco_py")

ipdb.set_trace()

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
        ball.set_shooter_control(sim, None if i == 0 else 0.)
    sim.step()
    viewer.render()
