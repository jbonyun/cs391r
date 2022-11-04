import math
import xml.etree.ElementTree as ET
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import array_to_string
from ball_spawn import BallTrajectory

class PingPongBall(BallObject):
    MASS = 0.0027    # kg, official ping pong ball = 2.7g
    RADIUS = 0.20    # m, official ping pong ball = 40mm diameter
    def __init__(self, trajectory: BallTrajectory, timestep, name_suffix='0'):
        self.name_suffix = str(name_suffix)
        super().__init__(
            name='ball{}'.format(self.name_suffix),
            size=[PingPongBall.RADIUS],
            rgba=[0, 0.5, 0.5, 1],
            solref=[-10000., -7.],  # set bouncyness as negative numbers. first is stiffness, second is damping.
            density=self.density(),
            )
        self.trajectory = trajectory
        print('Ball traj', self.trajectory)
        self.get_obj().set('pos', array_to_string(self.trajectory.origin))  # Set initial position.
        self.timestep = timestep
        self.actuator_id = None
    def volume(self):
        """Volume of the ball, m^3"""
        return PingPongBall.RADIUS**3 * math.pi * 4./3.
    def density(self):
        """Density of the ball, kg/m^3"""
        return PingPongBall.MASS / self.volume()
    def create_shooter(self):
        # Needs to be appended to the world's actuators list
        return ET.Element('general', attrib={'name': 'ball{}_shooter'.format(self.name_suffix), 'joint': 'ball{}_joint0'.format(self.name_suffix), 'gear': array_to_string(self.trajectory.velocity_vector) + ' 0 0 0'})
    def shooter_force(self):
        """How hard (in Newtons) the initial force must push for one frame time to instill initial velocity."""
        return self.trajectory.speed * PingPongBall.MASS / self.timestep
    def set_shooter_control(self, sim, set_to=None):
        """Apply the shooter_force to the actuator that will push this ball"""
        if self.actuator_id is None:
            raise Exception('You didnt set the actuator_id in the PingPongBall')
        sim.data.ctrl[self.actuator_id] = (self.shooter_force() if set_to is None else set_to)
