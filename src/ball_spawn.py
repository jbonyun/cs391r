#! /usr/bin/env python3

import math
import numpy as np
import random

import matplotlib.pyplot as plt  # For testing and visualizing


class BoxInSpace():
    # Defines a box in space.
    # It is defined by a central point and a normal that orients the direction of one face, plus
    # a constraint that the face the normal is identifying has its bottom edge parallel with xy plane.
    def __init__(self, center_xyz, normal_vector_of_close_side, height, width, depth):
        # Center is the x/y/z tuple of the location of the middle of the box, wrt world coords.
        # Normal vector is the direction x/y/z tuple that points from center point to center of face closest to origin.
        # height is length of front face that is perpendicular to xy plane.
        # width is length of front face that is parallel to xy plane
        # depth is the length away from the normal face.
        # All lengths are the full distance, so the center is halfway along the full length.
        self.size = (height, width, depth)
        # Find transformation from box-space to world-space.
        t = np.zeros([4,4])
        # Translation part is just the center point's location.
        t[0:3, 3] = center_xyz
        t[3,3] = 1
        if normal_vector_of_close_side is None:
            # Make it rectilinear in world space, because easier
            t[0,0] = 1
            t[1,1] = 1
            t[2,2] = 1
        else:
            # TODO: Not convinced this math is right...
            raise Exception('Math is probably wrong here. Use boxes that are square to space.')
            # New x axis is the normal
            t[0:3, 0] = normal_vector_of_close_side
            # New y axis has zero world-z component by our constraints.
            # That leaves the world-x and world-y to dictate it, and have it perp to x and perp to perp to xy
            t[0, 1] = np.sqrt(1 / (1 + (t[0,0] / t[1,0])**2))
            t[1, 1] = -np.sqrt(1 - t[0,1]**2)
            t[1, 2] = 0.
            # New z is whatever is perp to both of those
            t[0:3, 2] = np.cross(t[0:3,0], t[0:3,1])
        # Save that in a better variable name
        self.transform_box_wrt_world = t
    def random(self):
        # Select a random point in the box.
        box_point = np.ones([4, 1])  # 4x1 bc homogenous coordinates, 4th point is 1, others will be overwritten
        box_point[0:3,0] = np.array([random.uniform(-measure/2, measure/2) for measure in self.size])
        # Transform the point to world-space
        world_point = np.matmul(self.transform_box_wrt_world, box_point)
        return world_point[0:3,0].transpose()



def random_point_in_semicircle(minmax_rads, minmax_dist):
    # Picks a random point in a circle, or part of a circle
    # Choose an angle between min and max angles; (0,2*pi) would be a whole circle
    angle = random.uniform(minmax_rads[0], minmax_rads[1])
    # Choose a distance between 0 and max distance
    dist = random.uniform(0., 1.)
    dist = math.sqrt(dist)  # I think this corrects for the higher density near the center?
    dist = minmax_dist[0] + dist * (minmax_dist[1] - minmax_dist[0])
    return (angle, dist)

def point_on_plane(plane_center_point, plane_normal_vector, plane_zero_vector, rads_on_plane, dist_on_plane):
    # assert that plane_zero_vector and plane_normal_vector are perpendicular
    assert abs(np.dot(plane_normal_vector, plane_zero_vector)) < 1e-8, 'Zero vector must be on plane'
    # vector that is perp to both plane_normal_vector and plane_zero_vector
    plane_90_vector = np.cross(plane_normal_vector, plane_zero_vector)
    # find vector on plane that is rads_on_plane from plane_zero_vector
    v = np.cos(rads_on_plane) * plane_zero_vector + np.sin(rads_on_plane) * plane_90_vector
    # add dist_on_plane along that vector to the plane_center_point
    return plane_center_point + v * dist_on_plane

def random_point_on_plane(plane_center_point, plane_normal_vector, plane_zero_vector, angle_bounds, distance_bounds):
    rads, dist = random_point_in_semicircle(angle_bounds, distance_bounds)
    pt = point_on_plane(plane_center_point, plane_normal_vector, plane_zero_vector, rads, dist) 
    dist_ans = np.linalg.norm(plane_center_point - pt)
    #DEBUG print(rads, dist, pt, dist_ans)
    return pt

class CircleInSpace():
    def __init__(self, origin_point, plane_normal, zero_angle_vector, angle_size, radius):
        self.origin_point = np.array(origin_point)
        self.plane_normal = np.array(plane_normal)
        self.zero_angle_vector = np.array(zero_angle_vector)
        self.angle_size = angle_size
        self.radius = radius
        assert abs(np.dot(self.plane_normal, zero_angle_vector)) < 1e-8, 'Zero vector must be on plane'
        assert angle_size >= 0., 'Angle must be positive'
        assert radius >= 0., 'Radius must be positive'
    def random(self):
        return random_point_on_plane(self.origin_point, self.plane_normal, self.zero_angle_vector, (0., self.angle_size), (0., self.radius))




class SpeedSpawner():
    """Generates random speeds in a given range"""
    def __init__(self, min_vel, max_vel):
        self.min_vel = min_vel
        """Minimum allowed velocity (m/s)"""
        self.max_vel = max_vel
        """Maximum allowed velocity (m/s)"""
    def random(self):
        """Generate a random velocity"""
        return random.uniform(self.min_vel, self.max_vel)



class BallTrajectory():
    """Describes a ball's trajectory"""
    def __init__(self, origin, velocity, target=None):
        self.origin_ = np.array(origin)
        self.velocity_ = np.array(velocity)
        self.target_ = target
    @property
    def origin(self):
        """Starting point in x/y/z space"""
        return self.origin_
    @property
    def velocity(self):
        """Velocity vector with x/y/z components"""
        return self.velocity_
    @property
    def speed(self):
        """Scalar speed, without direction"""
        return np.linalg.norm(self.velocity_)
    @property
    def velocity_vector(self):
        """Normalized velocity vector, with unit length"""
        return self.velocity_ / self.speed
    @property
    def target(self):
        """Optional (can be None) destination of the ball in x/y/z space"""
        return self.target_
    def position(self, t):
        """Calculated position at time t"""
        #DEBUG print('orig', self.origin_, 't', t, 'vel', self.velocity_)
        return self.origin_ + t * self.velocity_
    def total_time(self):
        """The number of seconds it will take to reach target. None if target is None"""
        if self.target_ is None:
            return None
        path = self.target_ - self.origin
        times = np.divide(path, self.velocity_)
        # In theory these should all be the same... but some could be nan?
        return np.nanmax(times)
    


class BallSpawner():
    """A wrapper that holds everything necessary to generate a BallTrajectory"""
    def __init__(self, source_spawner=None, target_spawner=None, speed_spawner=None):
        self.src = source_spawner
        self.tgt = target_spawner
        self.spd = speed_spawner
    def random(self):
        """Generate a random BallTrajectory

        This gravity-free version creates a straight-line trajectory.
        (A more complicated calculation could accommodate gravity and still hit the target.)
        """
        o = self.src.random()
        t = self.tgt.random()
        s = self.spd.random()
        path = t - o  # Vector between source and target
        vel = s * path / np.linalg.norm(path)  # Normalize the path length, then multiply by speed
        return BallTrajectory(o, vel, target=t)



############### TESTING #############

def test_boxinspace():
    box_center = np.array([10, 10, 10])
    #source_box = BoxInSpace(box_center, -box_center, 1, 5, 1)
    source_box = BoxInSpace(box_center, None, 1, 1, 5)
    points = np.zeros([1000, 3])
    for i in range(1000):
        source_point = source_box.random()
        points[i, :] = source_point[0:3, :].transpose()
        #print('Source', source_point)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_zlim(0)
    ax.view_init(elev=15, azim=-130)
    plt.show()

def test_circleinspace():
    robot_circle =  CircleInSpace((0,0,0), (1,0,0), (0,1,0), 1.*math.pi, 1.)

    #source_circle = CircleInSpace((3,1,1), (3,1,1), (0,1,0), 2.*math.pi, 4.)
    zero_vec = (-np.sqrt(1 - roots[1]**2), roots[1], 0)
    print('zero vec', zero_vec, 'norm', np.linalg.norm(zero_vec), 'dot norm', np.dot((3,1,1), zero_vec))
    source_circle = CircleInSpace((3,1,1), (3,1,1), zero_vec, 2.*math.pi, 4.)
    for i in range(10):
        source_point = source_circle.random()
        target_point = robot_circle.random()
        print('Source', source_point, 'Target', target_point)


def plot_trajectory(traj, ax=None):
    total_time = traj.total_time()
    step_time = 0.1
    num_points = math.floor(total_time / step_time)
    points = np.zeros((num_points,3))
    for ip in range(num_points):
        t = 0. + step_time * ip
        points[ip, :] = traj.position(t)
    end_points = np.zeros((2,3))
    end_points[0, :] = traj.origin
    end_points[1, :] = traj.target
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=15, azim=-130)
    ax.scatter(points[:,0], points[:,1], points[:,2], 'bo', s=1)
    ax.scatter(end_points[:,0], end_points[:,1], end_points[:,2], 'rx')
    #ax.set_xlim(0)
    #ax.set_ylim(0)
    #ax.set_zlim(0)
    return ax

def test_spawner():
    spawner = BallSpawner()
    box_center = np.array([5, 0, 2])
    spawner.src = BoxInSpace(box_center, None, 2, 4, 3)
    spawner.tgt = CircleInSpace((0,0,1), (1,0,0), (0,1,0), 1.*math.pi, 1.)
    spawner.spd = SpeedSpawner(0.5, 2.0)
    ax = None
    for i in range(25):
        traj = spawner.random()
        ax = plot_trajectory(traj, ax)
    plt.show()
    

if __name__ == '__main__':
    #test_boxinspace()
    test_spawner()
