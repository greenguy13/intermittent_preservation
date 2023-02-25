#!/usr/bin/env python

"""
Environment: Determines the areas (charging station plus areas to monitor) in a map, and then sends those information out
as a server to the respective areas, which are its clients. If charging station: pose; if an area to monitor: pose + decay rate.

Subscribes to iGraph created by gvg
Server to poses of areas
Server to decay rates of areas

#Gets the parameters dict_rates

"""

import random as rd
import numpy as np
import rospy
import actionlib
import tf
import project_utils as pu
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Pose
from scipy.spatial import distance, Voronoi
import igraph
from grid import Grid
from visualization_msgs.msg import Marker

INDEX_FOR_X = 0
INDEX_FOR_Y = 1

class Environment:

    def __init__(self):
        """

        :param node_name:
        :param areas:
        :param est_distance_matrix:
        :param est_batt_consumption_matrix:
        """

        rospy.init_node("environment", anonymous=True)
        # self.listener = tf.TransformListener()

        #Parameters
        self.decay_rates_dict = rospy.get_param("/decay_rates_dict")
        self.robot_radius = rospy.get_param("/robot_radius")
        #How can we get the params of the areas?

        rospy.Subscriber('/map', OccupancyGrid, self.static_map_callback)
        self.marker_pub = rospy.Publisher('voronoi', Marker, queue_size=0)
        self.graph = None



    # Graph of the map
    def compute_gvg(self):
        """Compute GVG for exploration."""

        #start_time_clock = rospy.Time.now().to_sec()
        # Get only wall cells for the Voronoi.
        obstacles = self.latest_map.wall_cells()
        #end_time_clock = rospy.Time.now().to_sec()
        #pu.log_msg(self.robot_id,"generate obstacles2 {}".format(#end_time_clock - #start_time_clock),self.debug_mode)

        #start_time_clock = rospy.Time.now().to_sec()

        # Get Voronoi diagram.
        vor = Voronoi(obstacles)
        #end_time_clock = rospy.Time.now().to_sec()
        #pu.log_msg(self.robot_id,"voronoi {}".format(#end_time_clock - #start_time_clock),self.debug_mode)
        #start_time_clock = rospy.Time.now().to_sec()

        # Initializing the graph.
        self.graph = igraph.Graph()
        # Correspondance between graph vertex and Voronoi vertex IDs.
        voronoi_graph_correspondance = {}

        # Simplifying access of Voronoi data structures.
        vertices = vor.vertices
        ridge_vertices = vor.ridge_vertices
        ridge_points = vor.ridge_points

        # edges = []
        # weights = []
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
                graph_vertex_ids = [-1, -1] # temporary for finding verted IDs.

                # Determining graph vertex ID if existing or not.
                for point_id in range(len(graph_vertex_ids)):
                    if ridge_vertex[point_id] not in voronoi_graph_correspondance:
                        # if not existing, add new vertex.
                        graph_vertex_ids[point_id] = self.graph.vcount()
                        self.graph.add_vertex(
                            coord=vertices[ridge_vertex[point_id]])
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
        #end_time_clock = rospy.Time.now().to_sec()
        #pu.log_msg(self.robot_id,"ridge {}".format(#end_time_clock - #start_time_clock),self.debug_mode)

        # Publish GVG.
        self.publish_edges()

    def prune_leaves(self):
        current_vertex_id = 0 # traversing graph from vertex 0.
        self.leaves = {}
        self.intersections = {}
        while current_vertex_id < self.graph.vcount():
            if self.graph.degree(current_vertex_id) == 1:
                neighbor_id = self.graph.neighbors(current_vertex_id)[0]
                neighbor = self.graph.vs["coord"][neighbor_id]
                current_vertex = self.graph.vs["coord"][current_vertex_id]
                if not self.latest_map.is_frontier(
                    neighbor, current_vertex, self.min_range_radius): # TODO parameter.
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
        # TODO constant values set at the top.
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

    def run_operation(self):
        while not rospy.is_shutdown():
            if self.graph != None:
                print(self.graph.vs.indices)
                break
            print("No graph")
            rospy.sleep(1)

if __name__ == '__main__':
    Environment().run_operation()