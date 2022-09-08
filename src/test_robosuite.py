#! /usr/bin/env python3
import ipdb

from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()

#from robosuite.models.base import MujocoXML
#xml = MujocoXML('example.xml')
#world.merge(xml)

from robosuite.models.robots import Panda,IIWA

mujoco_robot = IIWA()  # Iiwa will stay balanced in straight up direction
#mujoco_robot = Panda()  # Panda will humorously flop over if you don't give it control

from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])  # Not sure what this is doing
world.merge(mujoco_arena)

from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

sphere = BallObject(
    name='ball',
    size=[0.02],  # ping pong is 40mm diameter = 2cm radius
    rgba=[0, 0.5, 0.5, 1],
    solref=[-10000., -7.],  # set bouncyness as negative numbers. first is stiffness, second is damping.
    ).get_obj()
sphere.set('pos', '1.0 0 1.0')  # table is something under z=1, so z=1 will be above the table
world.worldbody.append(sphere)

model = world.get_model(mode="mujoco_py")

ipdb.set_trace()

# How can we set initial velocity of the ball?
# In raw mujoco you can set the qvel property of mjData object.
#sphere.set('qvel', '0.5 0.5 0')
print(model.bodies)

from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()
