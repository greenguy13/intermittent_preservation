#!/usr/bin/python

"""
Taken from
"""

import rospy
from geometry_msgs.msg import Pose, PointStamped
from tf import TransformListener
import numpy as np
from scipy.ndimage import minimum_filter
import project_utils as pu
from bresenham import bresenham

FREE = 0.0
OCCUPIED = 90.0
INDEX_FOR_X = 0
INDEX_FOR_Y = 1

class Grid:
    """Occupancy Grid."""

    def __init__(self, map_msg):
        self.header = map_msg.header
        self.origin_translation = [map_msg.info.origin.position.x,
                                   map_msg.info.origin.position.y, map_msg.info.origin.position.z]
        self.origin_quaternion = [map_msg.info.origin.orientation.x,
                                  map_msg.info.origin.orientation.y,
                                  map_msg.info.origin.orientation.z,
                                  map_msg.info.origin.orientation.w]
        self.grid = np.reshape(map_msg.data,
                               (map_msg.info.height,
                                map_msg.info.width))  # shape: 0: height, 1: width.
        self.resolution = map_msg.info.resolution  # cell size in meters.

        self.tf_listener = TransformListener()  # Transformation listener.

        self.transformation_matrix_map_grid = self.tf_listener.fromTranslationRotation(
            self.origin_translation,
            self.origin_quaternion)

        self.transformation_matrix_grid_map = np.linalg.inv(self.transformation_matrix_map_grid)

        # Array to check neighbors.
        self.cell_radius = int(10 / self.resolution)  # TODO parameter
        self.footprint = np.ones((self.cell_radius + 1, self.cell_radius + 1))

    def cell_at(self, x, y):
        """Return cell value at x (column), y (row)."""
        return self.grid[int(y), int(x)]

    def is_free(self, x, y):
        if self.within_boundaries(x, y):
            return 0 <= self.cell_at(x, y) < 50  # TODO: set values.
        else:
            return False

    def is_obstacle(self, x, y):
        if self.within_boundaries(x, y):
            return self.cell_at(x, y) >= 50  # TODO: set values.
        else:
            return False

    def is_unknown(self, x, y):
        if self.within_boundaries(x, y):
            return self.cell_at(x, y) < 0  # TODO: set values.
        else:
            return True

    def within_boundaries(self, x, y):
        if 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]:
            return True
        else:
            return False

    def convert_coordinates_i_to_xy(self, i):
        """Convert coordinates if the index is given on the flattened array."""
        x = i % self.grid.shape[1]  # col
        y = i / self.grid.shape[1]  # row
        return x, y

    def wall_cells(self):
        """
        Return only *wall cells* -- i.e. obstacle cells that have free or
            unknown cells as neighbors -- as columns and rows.
        """

        # Array to check neighbors.
        window = np.asarray([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        # Return all cells that are obstacles and satisfy the neighbor_condition
        neighbor_condition = self.grid > minimum_filter(self.grid,
                                                        footprint=window, mode='constant', cval=10000)  # TODO: value.
        obstacles = np.nonzero((self.grid >= OCCUPIED) & neighbor_condition)
        obstacles = np.stack((obstacles[1], obstacles[0]),
                             axis=1)  # 1st column x, 2nd column, y.

        return obstacles

    def is_frontier(self, previous_cell, current_cell,
        distance):
        """current_cell a frontier?"""
        v = previous_cell - current_cell
        u = v/np.linalg.norm(v)

        end_cell = current_cell - distance * u

        x1, y1 = current_cell.astype(int)
        x2, y2 = end_cell.astype(int)

        for p in list(bresenham(x1, y1, x2, y2)):
            if self.is_unknown(*p):
                return True
            elif self.is_obstacle(*p):
                return False
        return False

    def unknown_area_approximate(self, cell):
        """Approximate unknown area with the robot at cell."""
        cell_x = int(cell[INDEX_FOR_X])
        cell_y = int(cell[INDEX_FOR_Y])

        min_x = np.max((0, cell_x - self.cell_radius))
        max_x = np.min((self.grid.shape[1], cell_x + self.cell_radius + 1))
        min_y = np.max((0, cell_y - self.cell_radius))
        max_y = np.min((self.grid.shape[0], cell_y + self.cell_radius + 1))

        return (self.grid[min_y:max_y, min_x:max_x] < FREE).sum()

    def unknown_area(self, cell):  # TODO orientation of the robot if fov is not 360 degrees
        """Return unknown area with the robot at cell"""
        unknown_cells = set()

        shadow_angle = set()
        cell_x = int(cell[INDEX_FOR_X])
        cell_y = int(cell[INDEX_FOR_Y])
        for d in np.arange(1, self.cell_radius):  # TODO orientation
            for x in range(cell_x - d, cell_x + d + 1):  # go over x axis
                for y in range(cell_y - d, cell_y + d + 1):  # go over y axis
                    if self.within_boundaries(x, y):
                        angle = np.around(np.rad2deg(pu.theta(cell, [x, y])), decimals=1)  # TODO parameter
                        if angle not in shadow_angle:
                            if self.is_obstacle(x, y):
                                shadow_angle.add(angle)
                            elif self.is_unknown(x, y):
                                unknown_cells.add((x, y))

        return len(unknown_cells)

    def pose_to_grid(self, pose):
        """Pose (x,y) in header.frame_id to grid coordinates"""
        # Get transformation matrix map-occupancy grid.
        return (self.transformation_matrix_grid_map.dot([pose[0], pose[1], 0, 1]))[
               0:2] / self.resolution  # TODO check int.

    def grid_to_pose(self, grid_coordinate):
        """Pose (x,y) in grid coordinates to pose in frame_id"""
        # Get transformation matrix map-occupancy grid.
        return (self.transformation_matrix_map_grid.dot(
            np.array([grid_coordinate[0] * self.resolution,
                      grid_coordinate[1] * self.resolution, 0, 1])))[0:2]

    def get_explored_region(self):
        """ Get all the explored cells on the grid map"""

        # TODO refactor the code, right now hardcoded to quickly solve the problem of non-common areas.
        # TODO more in general use matrices.
        def nearest_multiple(number, res=0.2):
            return np.round(res * np.floor(round(number / res, 2)), 1)

        p_in_sender = PointStamped()
        p_in_sender.header = self.header

        poses = set()
        self.tf_listener.waitForTransform("robot_0/map",
                                          self.header.frame_id, rospy.Time(),
                                          rospy.Duration(4.0))
        p_in_sender.header.stamp = rospy.Time()
        for x in range(self.grid.shape[1]):
            for y in range(self.grid.shape[0]):
                if self.is_free(x, y):
                    p = self.grid_to_pose((x, y))
                    p_in_sender.point.x = p[0]
                    p_in_sender.point.y = p[1]

                    p_in_common_ref_frame = self.tf_listener.transformPoint("robot_0/map", p_in_sender).point
                    poses.add((nearest_multiple(p_in_common_ref_frame.x), nearest_multiple(p_in_common_ref_frame.y)))
        return poses