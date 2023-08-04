#!/usr/bin/env python

"""
Environment: Determines the temporal areas in a map by sampling n nodes from a Voronoi graph that satisfy the condition of a corridor (node degree=3).
This list is then pickled

Subscribes to iGraph created by gvg
Server to poses of areas
Server to decay rates of areas

#Gets the parameters dict_rates

"""

import pickle
import random as rd
import numpy as np
import rospy
import tf
import project_utils as pu
from scipy.spatial import distance, Voronoi
import igraph
from grid import Grid
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Pose, PoseStamped
from visualization_msgs.msg import Marker
from reset_simulation import *

INDEX_FOR_X = 0
INDEX_FOR_Y = 1

class SampleAreaPoses:
    def __init__(self, node_name):
        """

        :param node_name:
        :param areas:
        :param est_distance_matrix:
        :param est_batt_consumption_matrix:
        """

        rospy.init_node(node_name, anonymous=True)

        # Parameters
        self.world = rospy.get_param("~world")
        self.nareas = rospy.get_param("~nareas")  # Sample nodes from voronoi equal to area count #STAR
        self.nplacements = rospy.get_param("~nplacements")

        self.debug_mode = rospy.get_param("/debug_mode")  # 0 or 1
        self.robot_radius = rospy.get_param("/robot_radius")  # for grid cell computation
        self.areas = [int(i + 1) for i in range(self.nareas)]  # list of int area IDs
        self.degree_criterion_node_selection = rospy.get_param("/degree_criterion_node_selection")

        # Initialize variables
        self.sampled_nodes_poses = dict()  # dict container for sampled nodes
        self.graph = None

        # Publishers/Subscribers
        self.robot_id = 'null'
        rospy.Subscriber('/map', OccupancyGrid, self.static_map_callback)
        self.marker_pub = rospy.Publisher('voronoi', Marker, queue_size=0)

    # MAP METHODS
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

    def sample_nodes_from_voronoi(self, nsample, seed):
        """
        Samples nodes from potential nodes generated by voronoi
        Potential nodes are filtered from the entire tree by self.degree_criterion_node_selection
        Called in static_map_callback
        :return:
        """
        rd.seed(seed)
        potential_nodes = self.graph.vs.select(_degree=self.degree_criterion_node_selection)
        sampled_nodes = rd.sample(potential_nodes.indices, nsample)

        # Establish the coordinate dictionary here
        for node in sampled_nodes:
            p = self.graph.vs[node]["coord"]
            p_t = self.latest_map.grid_to_pose(p)
            p_coords = (p_t[0], p_t[1])
            self.sampled_nodes_poses.append(p_coords)


    # Methods: Run operation
    """
    While the number of sampled nodes is less than the length of self.nareas+1, sleep.
    We pickle the sampled nodes as a list using filename.
    We then kill the node.
    """

    def sample_node_poses(self):
        """
        :return:
        """

        rate = rospy.Rate(1)
        while self.graph is None:
            self.debug("Waiting for graph to register...")
            rate.sleep()  # Data for decay rates haven't registered yet

        """
        Now for a given world, we will sample nareas (which is a loop in itself), and place them randomly in the world p times
        PO structure:
            > Migrate this to be a function of its own. We sha
        """
        for n in self.nareas:
            for p in range(self.nplacements):
                seed = n*1000 + (p+1)*100 + p*10
                sampled_nodes = self.sample_nodes_from_voronoi(n, seed)
                self.sampled_nodes_poses['n{}_p{}'.format(n, p+1)] = sampled_nodes

        #Pickle dump the sampled nodes poses
        filename = '{}_n{}_p{}_sampled_nodes_poses_dict'.format(self.world, self.nareas, self.nplacements)
        pu.dump_data(self.sampled_nodes_poses, filename)

        #Shutdown all nodes
        self.debug("Done sampling: {}".format(filename))
        self.shutdown(sleep=10)

    def dump_data(self, data, filename):
        """
        Pickle dumps recorded chosen optimal decisions
        :return:
        """
        with open('{}.pkl'.format(filename), 'wb') as f:
            pickle.dump(data, f)

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Sampling nodes done. Shutting down...")
        kill_nodes(sleep)


if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/int_preservation/results')
    SampleAreaPoses('sample_areas').sample_node_poses()