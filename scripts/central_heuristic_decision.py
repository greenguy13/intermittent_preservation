#!/usr/bin/env python

"""
Implements Clustered BFVG
1. Cluster the areas in the environment based on some attributes
    > In clustering we use K-means clustering algorithm
    > Q: What do you think would these attributes be if we are to construct a simulation?

2. We have available robots and unassigned clusters
    > What defines an available/un-assigned robot? Actually, un-assigned robot would be better
        + If the robot has no currently assigned cluster, either it is heading toward or parked in the charging station
    > What defines an unassigned cluster?
        + If there is no robot assigned to preserve that cluster of areas
    > How do we make the assignment?
        + We evaluate the cluster's value, we then assign them greedily, whichever has the highest value
        + For each cluster, we evaluate among the robots (whose current task is not to charge up) based on their battery level
            and current location. Among those, we evaluate the

Algorithm sketch:
    Among the areas in the environment, cluster them into n-clusters based on their attributes, where n is the number of robots
    Among the unassigned clusters, we evaluate their score, and then assign an available robot to it
        > Q1: Which unassigned clusters gets assignment first?
        > Q2: Which available robot gets assigned to an unassigned cluster?

data = inputs
clusters = Cluster(data) #clusters would be a list containing cluster of areas, where the number of clusters is the number of robots

PO: Evaluate the value of the unassigned clusters, and then store in a priority queue
    + Evaluation would be a forecast of the expected opportunity cost for a given number of future visits
if there is one unassigned cluster, we consider re-planning/re-assignment:
    + Note that this means there is one un-assigned robot whose task is to charge up or just parked
    + PO: Average distance within the cluster, Current location of each robot and their remaining battery,
        and whether their battery level can cover the forecasted number of future visits

Consider re-plan is triggered only when one robot is heading to a charging station

for cluster in unassigned clusters with priority:
    for robot in robots:
        evalute their score for that unassigned cluster
    assign the cluster to the robot with highest score

Note: We assume that each robot will have an assigned cluster. We first assume that we have oracle knowledge.

Initial assignment to robots
while operation:
    if at least one area is unassigned:
        Consider re-assignment of clusters among robots who are not charging up

Robot:
    Input, cluster of areas to be monitored
    d = best decision
    if d = 0:
        broadcast unassigned status

Okay. Does this cover everything/all cases?
What about if we have uncertainty? Perhaps we need to insert the assignment block inside? Yes, even the clustering part.

How do we evaluate a cluster and its assignment to potential robots?
    > Average/expected opportunity cost
    > The location of the robot to get there plus the cost would be inversely proportional
    > Or could even be the marginal opportunity cost / marginal battery consumption to get there, something like that
"""

import rospy
from time import process_time
import pickle
import numpy as np
import actionlib
from pruning import *
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from std_msgs.msg import Int8, Float32
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from int_preservation.srv import clusterAssignment
from status import centralStatus, battStatus, robotStatus, robotAssignStatus
from reset_simulation import *
from heuristic_fcns import *
from loss_fcns import *

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3  # GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html
SHUTDOWN_CODE = 99


class TaskScheduler:
    def __init__(self, node_name):
        """

        :param node_name:
        :param areas:
        :param est_distance_matrix:
        :param est_batt_consumption_matrix:
        """

        rospy.init_node(node_name, anonymous=True)

        # Parameters
        self.nrobots = rospy.get_param("/nrobots")
        self.robot_ids = [i for i in range(self.nrobots)]
        self.debug_mode = rospy.get_param("/debug_mode")
        self.robot_velocity = rospy.get_param("/robot_velocity")  # Linear velocity of robot; we assume linear and angular are relatively equal
        self.gamma = rospy.get_param("/gamma")  # discount factor
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery")  # Max battery
        self.battery_reserve = rospy.get_param("/battery_reserve")  # Battery reserve

        f_thresh = rospy.get_param("/f_thresh")
        self.fsafe, self.fcrit = f_thresh  # (safe, crit)

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = batt_consumed_per_time  # (travel, restoration)

        self.dec_steps = rospy.get_param("/dec_steps")  # STAR
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.nareas = rospy.get_param("/nareas")  # Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i + 1) for i in range(self.nareas)]  # list of int area IDs
        self.debug("Hello there!")
        self.debug("Nareas {}. Areas list: {}".format(self.nareas, self.areas))
        # self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data

        # Initialize variables/containers
        self.status = centralStatus.IDLE.value

        self.mission_areas = dict() #Mission areas of robots
        for robot_id in self.robot_ids:
            self.mission_areas[robot_id] = None

        self.assign_statuses = dict() #Assignment statuses of robots
        for robot_id in self.robot_ids:
            self.assign_statuses[robot_id] = None

        self.tlapses = dict() #Tlapses of areas
        for area in self.areas:
            self.tlapses[area] = 0

        self.robot_statuses = dict() #Robot statuses
        for robot_id in self.robot_ids:
            self.robot_statuses[robot_id] = None

        self.clusters = None #Clustering of areas
        self.clusters_assignment = dict() #Assignment of clusters (keys) to robots (values)
        self.robots_assignment = dict() #Assignment of robots (keys) to clusters (values)
        self.unassigned_clusters = list() #List of unassigned clusters
        self.unassigned_robots = list() #List of unassigned robots

        # Server
        # TODO: About robot's accomplishment of restoring an area. Needed for tlapse reset for that area

        # Publishers/Subscribers
        for robot_id in self.robot_ids:
            rospy.Subscriber('/robot_{}/assignment_status'.format(robot_id), Int8, self.assign_status_cb, robot_id)
            rospy.Subscriber('/robot_{}/mission_area'.format(robot_id), Int8, self.mission_area_cb, robot_id)
            rospy.Subscriber('/robot_{}/robot_status'.format(robot_id), Int8, self.robot_status_cb, robot_id)

    def assign_status_cb(self, msg, robot_id):
        """
        Updates the assignment status of robot from subscribed topic
        :param msg:
        :param robot_id:
        :return:
        """
        self.assign_statuses[robot_id] = int(msg.data)

    def mission_area_cb(self, msg, robot_id):
        """
        Updates the mission area of robot from subscribed topic
        :param msg:
        :param robot_id:
        :return:
        """
        self.mission_areas[robot_id] = int(msg.data)

    def robot_status_cb(self, msg, robot_id):
        """
        Updates the robots statuses.
        Moreover, updates the tlapses of areas based on robot's status and mission area/assignment status
        :param msg:
        :param robot_id:
        :return:
        """
        self.robot_statuses[robot_id] = int(msg.data)

    def create_clusters(self):
        clusters = dict()
        interval = self.nareas // self.nrobots
        areas = self.areas.copy()
        start = 0
        self.debug("Areas: {}, {}, Interval: {}".format(self.areas, areas, interval))
        for i in range(self.nrobots):
            clusters[i] = areas[start:start+interval]
            start = start + interval
        return clusters

    def assign_clusters(self, clusters):

        for robot_id in range(self.nrobots):
            self.robots_assignment[robot_id] = clusters[robot_id] #Assignment of robot to a cluster
            rospy.wait_for_service("/cluster_assignment_server_" + str(robot_id))
            try:
                cluster_assign = rospy.ServiceProxy("/cluster_assignment_server_" + str(robot_id), clusterAssignment)
                resp = cluster_assign(clusters[robot_id]) #TODO: Include decay_rates and tlapses of assigned areas
                self.debug("Robot: {}. Assigned: {}, {}".format(robot_id, clusters[robot_id], resp))
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        self.debug("Assignment of robots to clusters: {}".format(self.robots_assignment))

    def update_tlapses_areas(self):
        """
        Updates the tlapses of areas based on robot's status and mission area/assignment status
        :return:
        """
        for robot_id in self.robot_ids:
            if self.assign_statuses[robot_id] == robotAssignStatus.ASSIGNED.value and (self.robot_statuses[robot_id] == robotStatus.IDLE.value or self.robot_statuses[robot_id] == robotStatus.READY.value):
                cluster = self.clusters_assignment[robot_id]
                for area in self.clusters[cluster]:
                    self.tlapses[area] += 1


    def run_operation(self, filename, freq=1):
        rospy.sleep(10)

        self.sim_t = 0
        while self.sim_t < self.t_operation:
            if self.status == centralStatus.IDLE.value and self.clusters == None:
                self.clusters = self.create_clusters()
                self.assign_clusters(self.clusters)
                self.status = centralStatus.IN_MISSION.value

            elif self.status == centralStatus.IN_MISSION.value:
                pass

            elif self.status == centralStatus.CONSIDER_REPLAN.value:
                pass

            self.sim_t += 1
            rospy.sleep(1)
        # TODO: Save central data if any
        # TODO: Shutdown node

    def debug(self, msg):
        pu.log_msg(type='task_scheduler', id=None, msg=msg, debug=self.debug_mode)



if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation/results')
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    TaskScheduler('central_heuristic_decision').run_operation(filename)
