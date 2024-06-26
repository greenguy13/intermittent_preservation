#!/usr/bin/env python

import math
import os

import numpy as np
import pickle
from os import path
import rospy
import tf
import shapely.geometry as sg
from scipy import stats
from sklearn.metrics import mean_squared_error
from geometry_msgs.msg import Point, Pose, PoseStamped



TOTAL_COVERAGE = 1
MAXIMUM_EXPLORATION_TIME = 2
COMMON_COVERAGE = 3
FULL_COVERAGE = 4

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SMALL = 0.00000001
PRECISION = 0
FREE = 0
OCCUPIED = 100
UNKNOWN = -1
SCALE = 10.0

# navigation states
ACTIVE_STATE = 1  # This state shows that the robot is collecting messages
PASSIVE_STATE = -1  # This state shows  that the robot is NOT collecting messages
ACTIVE = 1  # The goal is currently being processed by the action server
SUCCEEDED = 3  # The goal was achieved successfully by the action server (Terminal State)
ABORTED = 4  # The goal was aborted during execution by the action server due to some failure (Terminal State)
LOST = 9  # An action client can determine that a goal is LOST. This should not be sent over the wire by an action


def add_entries_dicts(dict1, dict2):
    """
    Adds the entries of the dicts
    :param dicts:
    :return:
    """
    #If the keys are identical, we sum the values right away
    if list(dict1.keys()) == list(dict2.keys()):
        sum_array = np.array(list(dict1.values())) + np.array(list(dict2.values()))
        sum_array = sum_array.tolist()
        sum_dict = dict(zip(dict1.keys(), sum_array))

    #Else, we do a pairing to sum respective values
    else:
        sum_dict = dict()
        for key in dict1:
            sum_dict[key] = dict1[key] + dict2[key]

    return sum_dict

def save_data(data, file_name):
    saved_data = []
    if path.exists(file_name):
        os.unlink(file_name)
    f = open(file_name, "wb")
    f.close()
    with open(file_name, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()

def dump_data(recorded_data, filename):
    """
    Pickle dumps recorded chosen optimal decisions
    :return:
    """
    with open('{}.pkl'.format(filename), 'wb') as f:
        pickle.dump(recorded_data, f)
        f.close()

def load_data_from_file(file_name):
    data_dict = []
    if path.exists(file_name) and path.getsize(file_name) > 0:
        with open(file_name, 'rb') as fp:
            # try:
            data_dict = pickle.load(fp)
            fp.close()
            # except Exception as e:
            #     rospy.logerr("error saving data: {}".format(e))
    return data_dict


def pose2pixel(pose, origin_x, origin_y, resolution):
    x = round((pose[INDEX_FOR_X] - origin_y) / resolution, PRECISION)
    y = round((pose[INDEX_FOR_Y] - origin_x) / resolution, PRECISION)
    position = [0.0] * 2
    position[INDEX_FOR_X] = x
    position[INDEX_FOR_Y] = y
    return tuple(position)


def pixel2pose(point, origin_x, origin_y, resolution):
    new_p = [0.0] * 2
    new_p[INDEX_FOR_X] = origin_x + point[INDEX_FOR_X] * resolution
    new_p[INDEX_FOR_Y] = origin_y + point[INDEX_FOR_Y] * resolution
    return tuple(new_p)

def convert_coords_to_PoseStamped(coords, frame='map'):
    """
    Converts x,y coords to PoseStampled wrt frame
    :param coord:
    :return:
    """
    pose = PoseStamped()
    pose.header.seq = 0
    pose.header.frame_id = frame
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = coords[0]
    pose.pose.position.y = coords[1]
    pose.pose.orientation.w = 1.0

    return pose


def get_vector(p1, p2):
    xv = p2[INDEX_FOR_X] - p1[INDEX_FOR_X]
    yv = p2[INDEX_FOR_Y] - p1[INDEX_FOR_Y]
    v = [0] * 2
    v[INDEX_FOR_X] = xv
    v[INDEX_FOR_Y] = yv
    v = tuple(v)
    return v


def theta(p, q):
    dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
    dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
    return math.atan2(dy, dx)


def D(p, q):
    # rospy.logerr("Params: {}, {}".format(p, q))
    dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
    dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
    return math.sqrt(dx ** 2 + dy ** 2)


def euclidean_distance(p, q):
    # p and q are two-element arrays.
    # l2 norm.
    return np.linalg.norm(p - q)


def angle_pq_line(p, q):
    pq = p - q
    return np.arctan2(pq[1], pq[0])


def get_closest_point(point, set_of_points):
    """Get the closest point to a set of points."""

    dist_2 = np.sum((set_of_points - point) ** 2, axis=1)
    closest_point_id = np.argmin(dist_2)

    return closest_point_id, dist_2[closest_point_id]


def T(p, q):
    return D(p, q) * math.cos(theta(p, q))


def W(p, q):
    return abs(D(p, q) * math.sin(theta(p, q)))


def slope(p, q):
    dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
    dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
    if dx == 0:
        dx = SMALL
        if dy < 0:
            return -1 / dx
        return 1 / dx
    return dy / dx


def get_slope(p, q):
    return stats.linregress([p[INDEX_FOR_Y], q[INDEX_FOR_Y]],
                            [p[INDEX_FOR_X], p[INDEX_FOR_X]])[0]


def get_line(stacked_points):
    slope, intercept, r_value, p_value, std_err = stats.linregress(stacked_points)
    y_predict = intercept + slope * stacked_points[:, 0]

    return slope, intercept, np.sqrt(mean_squared_error(stacked_points[:, 1], y_predict))


def get_vector(p1, p2):
    xv = p2[INDEX_FOR_X] - p1[INDEX_FOR_X]
    yv = p2[INDEX_FOR_Y] - p1[INDEX_FOR_Y]
    v = [0] * 2
    v[INDEX_FOR_X] = xv
    v[INDEX_FOR_Y] = yv
    v = tuple(v)
    return v


def get_ridge_desc(ridge):
    p1 = ridge[0][0]
    p2 = ridge[0][1]
    q1 = ridge[1][0]
    q2 = ridge[1][1]
    return p1, p2, q1, q2


def line_points(p1, p2, parts):
    x_min = int(round(min([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])))
    y_min = int(round(min([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])))
    x_max = int(round(max([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])))
    y_max = int(round(max([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])))
    pts = zip(np.linspace(x_min, x_max, parts), np.linspace(y_min, y_max, parts))
    points = []
    for p in pts:
        point = [0.0] * 2
        point[INDEX_FOR_X] = round(p[0], 2)
        point[INDEX_FOR_Y] = round(p[1], 2)
        points.append(tuple(point))
    return points


def compute_similarity(v1, v2, e1, e2):
    dv1 = np.sqrt(v1[INDEX_FOR_X] ** 2 + v1[INDEX_FOR_Y] ** 2)
    dv2 = np.sqrt(v2[INDEX_FOR_X] ** 2 + v2[INDEX_FOR_Y] ** 2)
    dotv1v2 = v1[INDEX_FOR_X] * v2[INDEX_FOR_X] + v1[INDEX_FOR_Y] * v2[INDEX_FOR_Y]
    v1v2 = dv1 * dv2
    if v1v2 == 0:
        v1v2 = 1
    if abs(dotv1v2) == 0:
        dotv1v2 = 0
    cos_theta = round(dotv1v2 / v1v2, PRECISION)
    sep = separation(e1, e2)
    return cos_theta, sep


def get_linear_points(intersections, lidar_fov):
    linear_ridges = []
    for intersect in intersections:
        p1 = intersect[0][0]
        p2 = intersect[0][1]
        p3 = intersect[1][1]
        if D(p2, p3) > lidar_fov and collinear(p1, p2, p3):
            linear_ridges.append(intersect)
    return linear_ridges


def collinear(p1, p2, p3, width, bias):
    s1 = slope(p1, p2)
    s2 = slope(p2, p3)
    if bias >= abs(s1 - s2) and 2 * W(p2, p3) <= width:
        return True
    return False


def scale_up(pose, scale):
    p = [0.0] * 2
    p[INDEX_FOR_X] = round(pose[INDEX_FOR_X] * scale, PRECISION)
    p[INDEX_FOR_Y] = round(pose[INDEX_FOR_Y] * scale, PRECISION)
    p = tuple(p)
    return p


def scale_down(pose, scale):
    p = [0.0] * 2
    p[INDEX_FOR_X] = pose[INDEX_FOR_X] / scale
    p[INDEX_FOR_Y] = pose[INDEX_FOR_Y] / scale
    p = tuple(p)
    return p


def process_edges(edges):
    x_pairs = []
    y_pairs = []
    edge_list = list(edges)
    for edge in edge_list:
        xh, yh = reject_outliers(list(edge))
        if len(xh) == 2:
            x_pairs.append(xh)
            y_pairs.append(yh)
    return x_pairs, y_pairs


def separation(e1, e2):
    p1 = e1[0]
    p2 = e1[1]
    p3 = e2[0]
    p4 = e2[1]
    p2_p3 = W(p2, p3)
    return p2_p3
    # c1 = p1[INDEX_FOR_Y] - slope(p1, p2) * p1[INDEX_FOR_X]
    # c2 = p4[INDEX_FOR_Y] - slope(p3, p4) * p4[INDEX_FOR_X]
    # return abs(c1 - c2)


def is_free(p, pixel_desc):
    rounded_pose = get_point(p)
    return rounded_pose in pixel_desc and pixel_desc[rounded_pose] == FREE


def is_unknown(p, pixel_desc):
    rounded_pose = get_point(p)
    return rounded_pose in pixel_desc and pixel_desc[rounded_pose] == UNKNOWN


def is_obstacle(p, pixel_desc):
    new_p = get_point(p)
    return new_p in pixel_desc and pixel_desc[new_p] == OCCUPIED


def get_point(p):
    xc = round(p[INDEX_FOR_X], PRECISION)
    yc = round(p[INDEX_FOR_Y], PRECISION)
    new_p = [0.0] * 2
    new_p[INDEX_FOR_X] = xc
    new_p[INDEX_FOR_Y] = yc
    new_p = tuple(new_p)
    return new_p


def bresenham_path(p1, p2):
    points = []
    x1 = p1[INDEX_FOR_X]
    y1 = p1[INDEX_FOR_Y]
    x2 = p2[INDEX_FOR_X]
    y2 = p2[INDEX_FOR_Y]
    x = x1
    y = y1
    dx = x2 - x1
    dy = y2 - y1
    p = 2 * dx - dy
    while (x <= x2):
        points.append((x, y))
        x += 1
        if p < 0:
            p = p + 2 * dy
        else:
            p = p + 2 * dy - 2 * dx
            y += 1
    return points


def reject_outliers(data):
    raw_x = [v[INDEX_FOR_X] for v in data]
    raw_y = [v[INDEX_FOR_Y] for v in data]
    # rejected_points = [v for v in raw_x if v < 0]
    # indexes = [i for i in range(len(raw_x)) if raw_x[i] in rejected_points]
    # x_values = [raw_x[i] for i in range(len(raw_x)) if i not in indexes]
    # y_values = [raw_y[i] for i in range(len(raw_y)) if i not in indexes]
    # return x_values, y_values
    return raw_x, raw_y


def log_msg(type, id, msg, debug=True):
    if debug:
        if type == 'robot':
            rospy.logwarn("Robot {}: {}".format(id, msg))
        elif type == 'area':
            rospy.logwarn("Area {}: {}".format(id, msg))

def in_range(point, polygon):
    x = point[INDEX_FOR_X]
    y = point[INDEX_FOR_Y]
    return polygon[0][INDEX_FOR_X] <= x <= polygon[2][INDEX_FOR_X] and polygon[0][INDEX_FOR_Y] <= y <= polygon[2][
        INDEX_FOR_Y]


def creat_polygon(leaf, parent, width, radius):
    x = leaf[0]
    y = leaf[1]

    opp = width / 2.0
    adj = radius
    hyp = np.sqrt(opp ** 2 + adj ** 2)
    theta1 = theta(parent, leaf)
    angle_sum = (np.pi / 2) + theta1
    cos_val = opp * np.cos(angle_sum)
    sin_val = opp * np.sin(angle_sum)

    top_left_x = x + cos_val
    top_left_y = y + sin_val

    bottom_left_x = x - cos_val
    bottom_left_y = y - sin_val

    lx = x + hyp * np.cos(theta1)
    ly = y + hyp * np.sin(theta1)

    top_right_x = lx + cos_val
    top_right_y = ly + sin_val

    bottom_right_x = lx - cos_val
    bottom_right_y = ly - sin_val

    point = Point(test_point[0], test_point[1])
    polygon = Polygon([(bottom_left_x, bottom_left_y), (top_left_x, top_left_y), (top_right_x, top_right_y),
                       (bottom_right_x, bottom_right_y)])
    print(polygon.contains(point))

    points = [parent, leaf, (lx, ly), (bottom_left_x, bottom_left_y), (top_left_x, top_left_y),
              (top_right_x, top_right_y), (bottom_right_x, bottom_right_y)]
    return points


def there_is_unknown_region(p1, p2, pixel_desc, min_ratio=4.0):
    x_min = min([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])
    y_min = min([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])
    x_max = max([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])
    y_max = max([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])
    min_points = max([abs(x_max - x_min), abs(y_max - y_min)])
    bbox = sg.box(x_min, y_min, x_max, y_max)
    point_count = 0
    for p, v in pixel_desc.items():
        if v == UNKNOWN:
            p = sg.Point(p[INDEX_FOR_X], p[INDEX_FOR_Y])
            if bbox.contains(p):
                point_count += 1
    return point_count >= min_points


def get_robot_pose(listener, rid):
    robot_pose = None
    while not robot_pose:
        try:
            listener.waitForTransform("map".format(rid),
                                           "robot_{}/base_link".format(rid),
                                           rospy.Time(0),
                                           rospy.Duration(4.0))
            (robot_loc_val, rot) = listener.lookupTransform("map".format(rid),
                                                                 "robot_{}/base_link".format(rid),
                                                                 rospy.Time(0))
            robot_pose = robot_loc_val[0:2]
        except:
            rospy.sleep(1)
            pass
    robot_pose = np.array(robot_pose)
    return robot_pose
