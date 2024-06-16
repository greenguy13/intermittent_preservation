#!/usr/bin/env python3

import os
import pickle
from enum import Enum
import random as rd
import numpy as np
import rospy
import actionlib
from loss_fcns import *
from pruning import *
import tf
import project_utils as pu
from scipy.spatial import distance, Voronoi
import igraph
from grid import Grid
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import Int8, Float32
from visualization_msgs.msg import Marker
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from status import areaStatus, battStatus, robotStatus

"""
Tasks
    T1: Fine tune move_base params to effectively maneuver obstacles
    T2: Modularize some aspects of the scripts, particularly the sampling of nodes from decision methods
    T3: Save data by rosbag: Identify info/data for summary
    T4: Replicate the experiments (multiple experiments with diff seeds)
    T5: Visualize real-time changes in F-measure in areas
"""

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3  # GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html


class Robot:
    def __init__(self, node_name):
        """

        :param node_name:
        :param areas:
        :param est_distance_matrix:
        :param est_batt_consumption_matrix:
        """

        rospy.init_node(node_name, anonymous=True)
        self.listener = tf.TransformListener()

        # Parameters
        self.robot_id = rospy.get_param("~robot_id")
        self.debug_mode = rospy.get_param("/debug_mode")  # 0 or 1
        self.robot_velocity = rospy.get_param(
            "/robot_velocity")  # Linear velocity of robot; we assume linear and angular are relatively equal
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery")  # Max battery
        self.fsafe, self.fcrit = rospy.get_param("/f_thresh")  # (safe, crit)
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = rospy.get_param(
            "/batt_consumed_per_time")  # (travel, restoration)
        self.decay_rates_dict = rospy.get_param("/decay_rates_dict")  # Decay rate of areas
        self.areas = list(self.decay_rates_dict.keys())
        self.areas = [int(i) for i in self.areas]  # list of int area IDs
        self.dec_steps = rospy.get_param("/dec_steps")
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.robot_radius = rospy.get_param("/robot_radius")  # for grid cell computation
        self.nsample_areas = rospy.get_param("/area_count")  # Sample nodes from voronoi equal to area count
        self.seed = 100 + 10 * rospy.get_param("/run")
        self.degree_criterion_node_selection = rospy.get_param("/degree_criterion_node_selection")
        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation

        # Initialize variables
        charging_station_coords = rospy.get_param("/charging_station_coords")
        charging_pose_stamped = self.convert_coords_to_PoseStamped(charging_station_coords)
        self.sampled_nodes_poses = [charging_pose_stamped]  # list container for sampled nodes of type PoseStamped
        self.charging_station = 0
        self.curr_loc = self.charging_station  # Initial location robot is the charging station
        self.battery = self.max_battery  # Initialize battery at max, then gets updated by subscribed battery topic
        self.optimal_path = []
        self.graph, self.dist_matrix = None, None
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True
        self.curr_fmeasures = dict()  # container of current F-measure of areas
        self.decision_results = []

        # Publishers/Subscribers
        # Service request to move_base to get plan : make_Plan
        server = '/robot_' + str(self.robot_id) + '/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        rospy.Subscriber('/map', OccupancyGrid, self.static_map_callback)
        rospy.Subscriber('/robot_{}/battery_status'.format(self.robot_id), Int8, self.battery_status_cb)
        rospy.Subscriber('/robot_{}/battery'.format(self.robot_id), Float32, self.battery_level_cb)

        for area in self.areas:
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb,
                             area)  # REMARK: Here we assume that we have live measurements of the F-measures
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb)

        self.marker_pub = rospy.Publisher('voronoi', Marker, queue_size=0)
        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)

        # Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base',
                                                              MoveBaseAction)
        self.robot_goal_client.wait_for_server()

        """
        On charging:
            Robot's mission area is 0. It then changes its status to CHARGING once it reaches the charging station.
            The charging station, which subscribes to robot's status, charges up the battery.
            Here, we assume there is only one charging station.

            If the robot_status is other than CHARGING, the battery status is DEPLETING.

        On area restoration:
            Robot's mission area is a specific area. If reaches the area, it changes its status to RESTORING_F.
            Now, the current mission area, which subscribes to both robot_status and robot_mission_area topics, will restore F; while,
                those other areas not the mission area will have their F continually decay 
        """

    # MAP/NAVIGATION METHODS
    def static_map_callback(self, data):
        """
        Callback for grid
        :param data:
        :return:
        """
        self.latest_map = Grid(data)

        self.min_range_radius = 8 / self.latest_map.resolution
        self.min_edge_length = self.robot_radius / self.latest_map.resolution
        self.compute_gvg()
        if len(self.sampled_nodes_poses) < self.nsample_areas + 1:
            self.sample_nodes_from_voronoi()
            self.build_dist_matrix()
            pu.log_msg('robot', self.robot_id, "Nodes to preserve: " + str(self.sampled_nodes_poses), self.debug_mode)

    def compute_gvg(self):
        """Compute GVG for exploration."""

        start_time_clock = rospy.Time.now().to_sec()
        # Get only wall cells for the Voronoi.
        obstacles = self.latest_map.wall_cells()
        end_time_clock = rospy.Time.now().to_sec()
        pu.log_msg('robot', self.robot_id, "generate obstacles2 {}".format(end_time_clock - start_time_clock),
                   self.debug_mode)

        start_time_clock = rospy.Time.now().to_sec()

        # Get Voronoi diagram.
        vor = Voronoi(obstacles)
        end_time_clock = rospy.Time.now().to_sec()
        pu.log_msg('robot', self.robot_id, "voronoi {}".format(end_time_clock - start_time_clock), self.debug_mode)
        start_time_clock = rospy.Time.now().to_sec()

        # Initializing the graph.
        self.graph = igraph.Graph()
        # Correspondance between graph vertex and Voronoi vertex IDs.
        voronoi_graph_correspondance = {}

        # Simplifying access of Voronoi data structures.
        vertices = vor.vertices
        ridge_vertices = vor.ridge_vertices
        ridge_points = vor.ridge_points

        edges = []
        weights = []
        # Create a graph based on ridges.
        for i in range(len(ridge_vertices)):
            ridge_vertex = ridge_vertices[i]
            # If any of the ridge vertices go to infinity, then don't add.
            if ridge_vertex[0] == -1 or ridge_vertex[1] == -1:
                continue
            p1 = vertices[ridge_vertex[0]]
            p2 = vertices[ridge_vertex[1]]

            # Obstacle points determining the ridge.
            ridge_point = ridge_points[i]
            q1 = obstacles[ridge_point[0]]
            q2 = obstacles[ridge_point[1]]

            # If the vertices on the ridge are in the free space
            # and distance between obstacle points is large enough for the robot
            if self.latest_map.is_free(p1[INDEX_FOR_X], p1[INDEX_FOR_Y]) and \
                    self.latest_map.is_free(p2[INDEX_FOR_X], p2[INDEX_FOR_Y]) and \
                    distance.euclidean(q1, q2) > self.min_edge_length:

                # Add vertex and edge.
                graph_vertex_ids = [-1, -1]  # temporary for finding verted IDs.

                # Determining graph vertex ID if existing or not.
                for point_id in range(len(graph_vertex_ids)):
                    if ridge_vertex[point_id] not in voronoi_graph_correspondance:
                        # if not existing, add new vertex.
                        graph_vertex_ids[point_id] = self.graph.vcount()
                        self.graph.add_vertex(coord=vertices[ridge_vertex[point_id]])
                        voronoi_graph_correspondance[ridge_vertex[point_id]] = graph_vertex_ids[point_id]
                    else:
                        # Otherwise, already added before.
                        graph_vertex_ids[point_id] = voronoi_graph_correspondance[ridge_vertex[point_id]]

                # Add edge.
                self.graph.add_edge(graph_vertex_ids[0], graph_vertex_ids[1], weight=distance.euclidean(p1, p2))
            else:
                # Otherwise, edge not added.
                continue

        # Take only the largest component.
        cl = self.graph.clusters()
        self.graph = cl.giant()
        self.prune_leaves()
        end_time_clock = rospy.Time.now().to_sec()
        pu.log_msg('robot', self.robot_id, "ridge {}".format(end_time_clock - start_time_clock), self.debug_mode)

        # Publish GVG.
        self.publish_edges()

    def prune_leaves(self):
        current_vertex_id = 0  # traversing graph from vertex 0.
        self.leaves = {}
        self.intersections = {}
        while current_vertex_id < self.graph.vcount():
            if self.graph.degree(current_vertex_id) == 1:
                neighbor_id = self.graph.neighbors(current_vertex_id)[0]
                neighbor = self.graph.vs["coord"][neighbor_id]
                current_vertex = self.graph.vs["coord"][current_vertex_id]
                if not self.latest_map.is_frontier(neighbor, current_vertex, self.min_range_radius):
                    self.graph.delete_vertices(current_vertex_id)
                else:
                    self.leaves[current_vertex_id] = self.latest_map.unknown_area_approximate(current_vertex)
                    current_vertex_id += 1
            else:
                if self.graph.degree(current_vertex_id) > 2:
                    self.intersections[current_vertex_id] = self.graph.vs["coord"][current_vertex_id]
                current_vertex_id += 1

    def publish_edges(self):
        """For debug, publishing of GVG."""

        # Marker that will contain line sequences.
        m = Marker()
        m.id = 0
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.LINE_LIST
        m.color.a = 1.0
        m.color.r = 1.0
        m.scale.x = 0.1

        # Plot each edge.
        for edge in self.graph.get_edgelist():
            for vertex_id in edge:
                # Grid coordinate for the vertex.
                p = self.graph.vs["coord"][vertex_id]
                p_t = self.latest_map.grid_to_pose(p)
                p_ros = Point(x=p_t[0], y=p_t[1])

                m.points.append(p_ros)

        # Publish marker.
        self.marker_pub.publish(m)

    def sample_nodes_from_voronoi(self):
        """
        Samples nodes from potential nodes generated by voronoi
        Potential nodes are filtered from the entire tree by self.degree_criterion_node_selection
        Called in static_map_callback
        :return:
        """
        rd.seed(self.seed + 1)
        potential_nodes = self.graph.vs.select(_degree=self.degree_criterion_node_selection)
        sampled_nodes = rd.sample(potential_nodes.indices, self.nsample_areas)

        # Establish the coordinate dictionary here
        for node in sampled_nodes:
            p = self.graph.vs[node]["coord"]
            p_t = self.latest_map.grid_to_pose(p)
            p_ros = (p_t[0], p_t[1])
            pose_stamped = self.convert_coords_to_PoseStamped(p_ros)
            self.sampled_nodes_poses.append(pose_stamped)

    def convert_coords_to_PoseStamped(self, coords, frame='map'):
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