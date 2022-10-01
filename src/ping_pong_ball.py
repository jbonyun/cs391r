import math
import xml.etree.ElementTree as ET
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import array_to_string
from ball_spawn import BallTrajectory

class PingPongBall(BallObject):
    MASS = 0.0027    # kg, official ping pong ball = 2.7g
    RADIUS = 0.02    # m, official ping pong ball = 40mm diameter
    ball_count = 0   # global count of the number of Ball objects we have, so they can have unique names.
    def __init__(self, trajectory: BallTrajectory, timestep):
        self.index = PingPongBall.ball_count  # Remember and update the global object count.
        PingPongBall.ball_count = self.index + 1
        super().__init__(
            name='ball{}'.format(self.index),
            size=[PingPongBall.RADIUS],
            rgba=[0, 0.5, 0.5, 1],
            solref=[-10000., -7.],  # set bouncyness as negative numbers. first is stiffness, second is damping.
            density=self.density(),
            )
        self.trajectory = trajectory
        print('Ball traj', self.trajectory)
        self.get_obj().set('pos', array_to_string(self.trajectory.origin))  # Set initial position.
        self.timestep = timestep
    def volume(self):
        """Volume of the ball, m^3"""
        return PingPongBall.RADIUS**3 * math.pi * 4./3.
    def density(self):
        """Density of the ball, kg/m^3"""
        return PingPongBall.MASS / self.volume()
    def create_shooter(self):
        shooter_xml = ET.Element('general', attrib={'name': 'ball{}_shooter'.format(self.index), 'site': 'ball{}_default_site'.format(self.index), 'gear': array_to_string(self.trajectory.velocity_vector) + ' 0 0 0'})
        # Needs to be appended to the world's actuators list
        # TODO: somehow we need to find our self.actuator_id so we can activate it
        return shooter_xml
    def shooter_force(self):
        """How hard (in Newtons) the initial force must push for one frame time to instill initial velocity."""
        return self.trajectory.speed * PingPongBall.MASS / self.timestep
    def set_shooter_control(self, sim, set_to=None):
        """Apply the shooter_force to the actuator that will push this ball"""
        sim.data.ctrl[self.actuator_id] = (self.shooter_force() if set_to is None else set_to)
