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
from int_preservation.srv import missionAccomplishment
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
        self.charging_station = 0 #charging station index

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

        self.robots_location = dict() #Robots location
        for robot_id in self.robot_ids:
            self.robots_location[robot_id] = None

        self.robots_battery = dict() #Robots battery
        for robot_id in self.robot_ids:
            self.robots_battery[robot_id] = None

        self.decay_rates = dict() #Decay rates
        for area in self.areas:
            self.decay_rates[area] = None

        self.clusters = None #Clustering of areas #TODO: Proper naming of clusters, with their keys and values
        self.clusters_assignment = dict() #Assignment of clusters (keys) to robots (values)
        self.robots_assignment = dict() #Assignment of robots (keys) to clusters (values)
        self.unassigned_clusters = list() #List of unassigned clusters #TODO: Initialize cluster keys, then store them here
        self.unassigned_robots = list() #List of unassigned robots #TODO: Initialize robots, then store here

        # Server
        # TODO: About robot's accomplishment of restoring an area. Needed for tlapse reset for that area. DONE
        self.mission_accomplishment_server = rospy.Service("/mission_accomplishment_server", missionAccomplishment, self.mission_accomplishment_cb)

        # Publishers/Subscribers
        self.central_status_pub = rospy.Publisher('/central_status', Int8, queue_size=1)

        for robot_id in self.robot_ids:
            rospy.Subscriber('/robot_{}/assignment_status'.format(robot_id), Int8, self.assign_status_cb, robot_id)
            rospy.Subscriber('/robot_{}/mission_area'.format(robot_id), Int8, self.mission_area_cb, robot_id)
            rospy.Subscriber('/robot_{}/robot_status'.format(robot_id), Int8, self.robot_status_cb, robot_id)
            rospy.Subscriber('/robot_{}/location'.format(robot_id), Int8, self.robot_location_cb, robot_id)
            rospy.Subscriber('/robot_{}/battery'.format(robot_id), Int8, self.robot_battery_cb, robot_id)

    def robot_location_cb(self, msg, robot_id):
        """
        Receives and stores robot location
        :param msg:
        :param robot_id:
        :return:
        """
        curr_loc = msg.data
        self.robots_location[robot_id] = int(curr_loc)

    def robot_battery_cb(self, msg, robot_id):
        """
        Receives and stores robot battery
        :param msg:
        :param robot_id:
        :return:
        """
        battery = msg.data
        self.robots_battery[robot_id] = int(battery)

    def mission_accomplishment_cb(self, msg):
        """
        Receives mission accomplishment from robots. Central then updates tlapse for that area
        :return:
        """
        #Tlapse reset if area is mission area
        robot_id = msg.robot_id
        area_id = msg.area_accomplished
        self.tlapses[area_id] = 0
        self.debug("Robot: {} restored Area: {}. Tlapse reset: {}".format(robot_id, area_id, self.tlapses[area_id]))

    def assign_status_cb(self, msg, robot_id):
        """
        Updates the assignment status of robot from subscribed topic
        :param msg:
        :param robot_id:
        :return:
        """
        assign_status = msg.data
        self.assign_statuses[robot_id] = int(assign_status)

    def mission_area_cb(self, msg, robot_id):
        """
        Updates the mission area of robot from subscribed topic
        :param msg:
        :param robot_id:
        :return:
        """
        mission_area = msg.data
        self.mission_areas[robot_id] = int(mission_area)

    def robot_status_cb(self, msg, robot_id):
        """
        Updates the robots statuses.
        Moreover, updates the tlapses of areas based on robot's status and mission area/assignment status
        :param msg:
        :param robot_id:
        :return:
        """
        robot_status = msg.data
        self.robot_statuses[robot_id] = int(robot_status)

    def decay_rate_cb(self, msg, area_id):
        """
        Store decay rate
        :param msg:
        :param area_id:
        :return:
        """
        if self.decay_rates[area_id] == None:
            self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates[area_id] = msg.data

    def create_clusters(self):
        """
        Creates clusters
        :return:
        """

        clusters = dict()
        interval = self.nareas // self.nrobots
        areas = self.areas.copy()
        start = 0
        self.debug("Areas: {}, {}, Interval: {}".format(self.areas, areas, interval))
        for i in range(self.nrobots):
            clusters['C' + str(i+1)] = areas[start:start+interval]
            start = start + interval
        return clusters

    def assign_clusters(self, clusters):
        """
        Assigns clusters to robots
        :param clusters:
        :return:
        """

        for robot_id in range(self.nrobots): #TODO: Should be unassigned robots. Initially, all robots are among the unassigned
            self.robots_assignment[robot_id] = 'C' + str(robot_id+1) #Assignment of robot to a cluster #TODO: For now, the assignment is 1:1, not yet K-means cluster
            # TODO: Remove unassigned clusters and robots if any
            rospy.wait_for_service("/cluster_assignment_server_" + str(robot_id))
            try:
                cluster_assign = rospy.ServiceProxy("/cluster_assignment_server_" + str(robot_id), clusterAssignment)
                areas_assigned = self.clusters[self.robots_assignment[robot_id]]
                # tlapses_areas = self.retrieve_tlapses(areas_assigned) #TODO: Supply here
                # decay_rates = self.retrieve_decay_rates(areas_assigned) #TODO: Supply here
                resp = cluster_assign(areas_assigned) #TODO: Include decay_rates and tlapses of assigned areas
                self.debug("Robot: {}. Assigned: {}, Response: {}".format(robot_id, areas_assigned, resp.result))
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        self.debug("Assignment of robots to clusters: {}".format(self.robots_assignment))

    def retrieve_tlapses(self, areas):
        """
        Retrieves the tlapses of areas
        :param areas:
        :return:
        """
        tlapses = list()
        for area in areas:
            tlapses.append(self.tlapses[area])
        return tlapses

    def retrieve_decay_rates(self, areas):
        """
        Retrieves the decay rates of areas
        :param areas:
        :return:
        """
        decay_rates = list()
        for area in areas:
            decay_rates.append(self.decay_rates[area])
        return decay_rates

    def update_tlapses_areas(self):
        """
        Updates the tlapses of areas based on robot's status and mission area/assignment status
        :return:
        """
        if self.status != centralStatus.IDLE.value and self.status != centralStatus.CONSIDER_REPLAN.value:
            for robot_id in self.robot_ids:
                #Case 1: Elapse time when robots are assigned and not idle/ready and central is not thinking
                if self.assign_statuses[robot_id] == robotAssignStatus.ASSIGNED.value and (self.robot_statuses[robot_id] != robotStatus.IDLE.value and self.robot_statuses[robot_id] != robotStatus.READY.value):
                    cluster = self.robots_assignment[robot_id]
                    for area in self.clusters[cluster]:
                        self.tlapses[area] += 1

                #Case 2: Elapse time for unassigned areas when robot is charging and central is not thinking
                # elif self.assign_statuses[robot_id] == robotAssignStatus.UNASSIGNED.value and (self.robot_statuses[robot_id] != robotStatus.IDLE.value and self.robot_statuses[robot_id] != robotStatus.READY.value) and self.mission_areas[robot_id] == self.charging_station:
                #     cluster = self.robots_assignment[robot_id]
                #     del self.clusters_assignment[cluster]
                #     del self.robots_assignment[robot_id]
                #     self.unassigned_clusters.append(cluster)
                #     self.unassigned_robots.append(robot_id)
                #
                #     for area in self.clusters[cluster]:
                #         self.tlapses[area] += 1



    def run_operation(self, filename, freq=1):
        rospy.sleep(10)

        wait_registry = True
        while (wait_registry is True) and (len(self.sampled_nodes_poses) != self.nareas + 1):
            na_count = 0
            for area in self.decay_rates:
                if self.decay_rates[area] is None:
                    na_count += 1
            if na_count > 0:
                # self.debug("Insufficient data. Decay rates: {}/{}. Sampled nodes poses: {}/{}".format(na_count, self.nareas,
                #                                                                                       len(self.sampled_nodes_poses), self.nareas+1))
                rospy.sleep(1)  # Data for decay rates haven't registered yet
            else:
                wait_registry = False
        self.debug("Sufficent data. Decay rates: {}".format(self.decay_rates))

        self.sim_t = 0
        while self.sim_t < self.t_operation:
            self.central_status_pub.publish(self.status) # TODO: Publish status. DONE
            self.print_state()
            if self.status == centralStatus.IDLE.value and self.clusters == None:
                self.debug("Idle central state")
                self.clusters = self.create_clusters() #TODO: The clusters should be correct keys and values
                self.assign_clusters(self.clusters)
                self.update_central_status(centralStatus.IN_MISSION)

            elif self.status == centralStatus.IN_MISSION.value:
                self.debug("Central in mission...")
                self.update_tlapses_areas() #TODO: Po, place here. Or could also be at the end of the while loop

            elif self.status == centralStatus.CONSIDER_REPLAN.value:
                self.debug("Central considers re-assignment...")

            self.sim_t += 1
            rospy.sleep(1)
        # TODO: Save central data if any
        # TODO: Shutdown node

    def print_state(self):
        """
        Prints current state of the environment based on info collected by central
        :return:
        """
        state = (self.sim_t, self.robots_curr_loc, self.robots_battery, self.tlapses, self.decay_rates)
        self.debug("State: {}".format(state))

    def update_central_status(self, status):
        """
        Updates task scheduler status
        :param status:
        :return:
        """
        self.status = status.value

    def debug(self, msg):
        pu.log_msg(type='task_scheduler', id=None, msg=msg, debug=self.debug_mode)



if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation/results')
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    TaskScheduler('central_heuristic_decision').run_operation(filename)
