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
from int_preservation.srv import assignmentAccomplishment
from int_preservation.srv import registerRobot, registerRobotResponse
from int_preservation.srv import pauseSimulation
from status import centralStatus, battStatus, robotStatus, robotAssignStatus
from reset_simulation import *
from heuristic_fcns import *
from loss_fcns import *

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3  # GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html
SHUTDOWN_CODE = 99


class CentralPlanner:
    def __init__(self, node_name):
        """

        :param node_name:
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
        self.tolerance = rospy.get_param("/move_base_tolerance")
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

        # self.check_pause()

        self.debug("Nareas {}. Areas list: {}".format(self.nareas, self.areas))
        # self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data

        #Sampled nodes poses
        # charging_station_coords = (0, 0) #rospy.get_param("~initial_pose_x"), rospy.get_param("~initial_pose_y")  # rospy.get_param("/charging_station_coords")
        # charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        # self.nodes_poses = [charging_pose_stamped]  # list container for sampled nodes of type PoseStamped, where 0 is the charging station for that robot

        # Pickle load the sampled area poses
        self.areas_poses = list()
        with open('{}.pkl'.format(rospy.get_param("/file_sampled_areas")), 'rb') as f:
            sampled_areas_coords = pickle.load(f)
        for area_coords in sampled_areas_coords['n{}_p{}'.format(self.nareas, rospy.get_param("/placement"))]:
            pose_stamped = pu.convert_coords_to_PoseStamped(area_coords)
            self.areas_poses.append(pose_stamped)

        self.dist_matrices = dict() #Initialize distance matrices of each robot #TODO: Distance matrices of registered robots
        for robot in range(self.nrobots):
            self.dist_matrices[robot] = None

        self.charging_station = 0

        # Initialize variables/containers
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

        self.clusters = None #Clustering of areas
        self.clusters_assignment = dict() #Assignment of clusters (keys) to robots (values)
        self.robots_assignment = dict() #Assignment of robots (keys) to clusters (values)
        self.unassigned_clusters = list() #List of unassigned clusters
        self.unassigned_robots = list() #List of unassigned robots

        # Server
        self.robots_registry_server = rospy.Service('/robots_registry_server', registerRobot, self.register_robots_cb)
        self.assignment_accomplishment_server = rospy.Service('/assignment_accomplishment_server', assignmentAccomplishment, self.assignment_accomplishment_cb)

        # Publishers/Subscribers
        self.central_status_pub = rospy.Publisher('/central_status', Int8, queue_size=1)

        # Service request to move_base to get plan : make_Plan
        server = '/robot_0/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        for robot_id in self.robot_ids:
            rospy.Subscriber('/robot_{}/assignment_status'.format(robot_id), Int8, self.assign_status_cb, robot_id)
            rospy.Subscriber('/robot_{}/mission_area'.format(robot_id), Int8, self.mission_area_cb, robot_id)
            rospy.Subscriber('/robot_{}/robot_status'.format(robot_id), Int8, self.robot_status_cb, robot_id)
            rospy.Subscriber('/robot_{}/location'.format(robot_id), Int8, self.robot_location_cb, robot_id)
            rospy.Subscriber('/robot_{}/battery'.format(robot_id), Float32, self.robot_battery_cb, robot_id)

        #Here: It is assumed oracle knoweldge of decay rates
        for area in self.areas:
            rospy.Subscriber('/area_{}/decay_rate'.format(area), Float32, self.decay_rate_cb, area)

    def register_robots_cb(self, msg):
        """
        Register robots id
        :return:
        """
        #TODO: UPNEXT. Sanity check
        robot_id = msg.robot_id #robot id for registration
        init_x = msg.init_x
        init_y = msg.init_y

        self.debug("Registry request received (id, x, y): {}, {}, {}".format(robot_id, init_x, init_y))

        #Build distance matrix for that robot
        self.dist_matrices[robot_id] = self.build_dist_matrix(robot_id, float(init_x), float(init_y)) #TODO

        #Debug that robot has been registered
        self.debug("Robot registered: {}. Dist matrix: {}".format(robot_id, self.dist_matrices[robot_id]))

        return registerRobotResponse(True)

    # METHODS: Node poses and distance matrix
    def get_plan_request(self, start_pose, goal_pose, tolerance):
        """
        Sends a request to GetPlan service to create a plan for path from start to goal without actually moving the robot
        :param start_pose:
        :param goal_pose:
        :param tolerance:
        :return:
        """
        req = GetPlan()
        req.start = start_pose
        req.goal = goal_pose
        req.tolerance = tolerance
        server = self.get_plan_service
        result = server(req.start, req.goal, req.tolerance)
        path = result.plan.poses
        return path

    def decouple_path_poses(self, path):
        """
        Decouples a path of PoseStamped poses; returning a list of x,y poses
        :param path: list of PoseStamped
        :return:
        """
        list_poses = list()
        for p in path:
            x, y = p.pose.position.x, p.pose.position.y
            list_poses.append((x, y))
        return list_poses

    def compute_path_total_dist(self, list_poses):
        """
        Computes the total path distance
        :return:
        """
        total_dist = 0
        for i in range(len(list_poses) - 1):
            dist = math.dist(list_poses[i], list_poses[i + 1])
            total_dist += dist
        return total_dist

    def compute_dist_bet_areas(self, area_i, area_j, tolerance):
        """
        Computes the distance between area_i and area_j:
            1. Call the path planner between area_i and area_j
            2. Decouple the elements of path planning
            3. Compute the distance then total distance
        :param area_i: PoseStamped
        :param area_j: PoseStamped
        :return:
        """
        path = self.get_plan_request(area_i, area_j, tolerance) #TODO: Which robot movebase shall we use to compute this? Could it be any robot? Could be robot_1
        list_poses = self.decouple_path_poses(path)
        total_dist = self.compute_path_total_dist(list_poses)
        return total_dist

    def build_dist_matrix(self, robot_id, init_x, init_y):
        """
        Builds the distance matrix among areas
        :return:
        """
        #TODO: Concatenate charging station with the area nodes
        charging_station_coords = (init_x, init_y)
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        nodes_poses = [charging_pose_stamped]
        nodes_poses.extend(self.areas_poses)
        self.debug("Robot: {}. Nodes_poses: {}".format(robot_id, nodes_poses))

        n = len(nodes_poses)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                area_i, area_j = nodes_poses[i], nodes_poses[j]
                if area_i != area_j:
                    dist = self.compute_dist_bet_areas(area_i, area_j, self.tolerance)
                    dist_matrix[i, j] = dist

        # self.debug("Dist matrix: {}".format(self.dist_matrix))
        return dist_matrix

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

    def assignment_accomplishment_cb(self, msg):
        """
        Receives area accomplishment from robots. Central then updates tlapse for that area
        :return:
        """
        #Tlapse reset if area is assigned area
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
        self.debug("Areas: {}, No. of Robots: {}, Interval: {}".format(areas, self.nrobots, interval))
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

        """
        This is where we would assign clusters to robots. We do a greedy assignment where we choose the highly urgent more distant clusters first,
            assigning them with the best robot that is closest and has more coverage capability
        """

        self.debug("Unassigned robots: {}".format(self.unassigned_robots))
        self.debug("Clusters: {}".format(clusters))
        assigned = 0
        for robot_id in self.unassigned_robots:
            self.robots_assignment[robot_id] = 'C' + str(robot_id+1) #Assignment of robot to a cluster #TODO: For now, the assignment is 1:1, not yet K-means cluster
            self.unassigned_robots.remove(robot_id)
            rospy.wait_for_service("/cluster_assignment_server_" + str(robot_id))
            try:
                cluster_assign = rospy.ServiceProxy("/cluster_assignment_server_" + str(robot_id), clusterAssignment)
                areas_assigned = clusters[self.robots_assignment[robot_id]]
                tlapses_areas = self.retrieve_tlapses(areas_assigned)
                decay_rates = self.retrieve_decay_rates(areas_assigned)
                resp = cluster_assign(areas_assigned, tlapses_areas, decay_rates)
                self.debug("Robot: {}. Assigned: {}, Robot availability: {}".format(robot_id, areas_assigned, resp.availability))
                self.debug("Remaining unassigned: {}".format(self.unassigned_robots))
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
            assigned += 1
        self.debug("Assignment of {} robots to clusters: {}".format(assigned, self.robots_assignment))

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
                        self.tlapses[area] += 1 #TODO: Update tlapses here

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
    #TODO: A method to consider re-assignment of robots
    def intra_cluster_closeness(self, distance_matrix, cluster_areas):
        """
        Calculate the closeness centrality of each node within a given cluster.

        Inputs:
        distance_matrix (2D np.array): The distance matrix representing the whole network.
        cluster_areas (list): The list of areas that belong to the cluster.

        Returns:
        dict: A dictionary where keys are node indices and values are their closeness centrality within the cluster.
        """
        # Extract the submatrix for the cluster
        cluster_submatrix = distance_matrix[np.ix_(cluster_areas, cluster_areas)]

        # Number of nodes in the cluster
        num_nodes_in_cluster = cluster_submatrix.shape[0]

        # Dictionary to store closeness centrality for nodes in the cluster
        cluster_centrality = {}

        # Iterate over each node in the cluster
        for idx, node in enumerate(cluster_areas):
            total_distance = np.sum(cluster_submatrix[idx])

            # Avoid division by zero for isolated nodes
            if total_distance > 0:
                cluster_centrality[node] = (num_nodes_in_cluster - 1) / total_distance
            else:
                cluster_centrality[node] = 0.0  # For isolated nodes

        return cluster_centrality

    def closeness_robot_to_cluster(self, distance_matrix, robot_location, cluster_nodes):
        """
        Calculate the closeness of a node to a cluster of nodes.

        Inputs:
        distance_matrix (2D np.array): Distance matrix for the whole graph.
        node (int): The node whose closeness to the cluster is to be measured.
        cluster_nodes (list): The list of nodes representing the cluster.

        Returns:
        float: Closeness of the node to the cluster (lower values indicate closer proximity).
        """
        avg_distance = np.mean([distance_matrix[robot_location][i] for i in cluster_nodes])

        # PO: the minimum distance from the node to any node in the cluster
        # min_distance = min(distance_matrix[robot_location][i] for i in cluster_nodes)
        return avg_distance


    def compute_clusters_scores_then_sort(self):
        """
        Computes cluster scores
        :return:
        """
        #Compute losses score
            #Total losses of areas within the cluster
            # Measure the loss which is a function of the tlapse of the areas within that cluster and their corresponding decay rate

        """
        We call on the clusters with their corresponding areas. For each cluster, we measure the losses of the areas.
            We do this by retrieving their tlapses and decay rates to compute the decay fmeasure,
                then pair this with the max fmeasure to get the loss
            We sum up the losses of the areas
            We store in a dictionary wherein the cluster loss as the value while the cluster itself is the key
            
            PO: We may have a second score on which to sort the dictionary
            We then sort the dictionary by value
            We return the sorted keys of the sorted dictionary
        """
        clusters = dict()
        for cluster in self.clusters:
            total_losses = 0
            for area in cluster:
                #Measure losses
                decayed_f = decay(self.decay_rates[area], self.tlapses[area], self.max_fmeasure)
                loss = loss_fcn(self.max_fmeasure, decayed_f)
                total_losses += loss

            #Measure intra-cluster closeness
            intra_closeness = self.intra_cluster_closeness(self.dist_matrix, self.clusters)
        #Sort the clusters by losses descendingly and closeness descendingly

        #Return the sorted clusters

    def compute_robots_scores_then_sort(self):
        """
        Computes robots scores
        :return:
        """
        #Compute robots scores
            # Measure closeness to the cluster

            # Retrieve remaining battery

        #Sort the robots by closeness ascendingly then remaining battery descendingly
        #Return sorted robots

    def consider_reassignment(self):
        """
        Considers re-assignment of robots available whenever a cluster is unassigned. An unassigned cluster occurs
            when a robot decides to charge up
        """
        # Get unassigned robots
        # Create clusters based on the number of unassigned robots
        # Measure the cluster scores then sort
        # For each cluster in sorted clusters
            # Find the best unassigned robot for that cluster
            # If cluster assignment is not the cluster, update cluster assignment for that robot


    def wait_nodes_to_register(self):
        """
        Wait for area nodes to register
        :return:
        """
        na_counts = True
        while na_counts is True:
            decay_rates, dist_matrices = list(self.decay_rates.values()), list(self.dist_matrices.values())
            self.debug("Decay rates: {}. Dist matrices: {}".format(decay_rates, dist_matrices))
            is_notna_decay_rates = all(element is not None for element in decay_rates)
            is_notna_dist_matrices = all(element is not None for element in dist_matrices)
            if is_notna_decay_rates is True and is_notna_dist_matrices is True:
                na_counts = False
            rospy.sleep(1)
        self.debug("Sufficent data. Decay rates: {}. Dist matrices: {}".format(self.decay_rates, self.dist_matrices))

    def run_operation(self, filename, freq=1):
        rospy.sleep(10)
        self.wait_nodes_to_register()
        self.unassigned_robots = self.robot_ids
        self.status = centralStatus.IDLE.value
        self.sim_t = 0
        while not rospy.is_shutdown() and self.sim_t < self.t_operation:
            self.central_status_pub.publish(self.status)
            self.print_state()

            if self.status == centralStatus.IDLE.value:
                self.debug("Idle central state. Creating and assigning clusters")

                # TODO: Send pause simulation request here
                self.clusters = self.create_clusters() #Thinking. Pause
                self.assign_clusters(self.clusters)
                if len(self.unassigned_robots) == 0:
                    self.update_central_status(centralStatus.IN_MISSION)

            elif self.status == centralStatus.IN_MISSION.value:
                self.debug("Central in mission...")

                self.update_tlapses_areas() #TODO: Update tlapses here. Yea this is correct should be +=1. Or could be from time. Yea we can use from time

            elif self.status == centralStatus.CONSIDER_REPLAN.value:
                self.debug("Central considers re-assignment...")

            self.sim_t += 1 #TODO: Update tlapses here
            rospy.sleep(1)
        # TODO: Save central data if any
        self.shutdown(sleep=10)

    def check_pause(self):
        """
        Checks the pause request to Stage
        :return:
        """
        #Check time whether it is ticking
        #Then check time again after pausing the simulation and whether it is likewise ticking
        time_new = rospy.get_time()
        for i in range(5):
            time_prev = time_new
            time_new = rospy.get_time()
            tlapse = time_new - time_prev
            self.debug("Ticking instance: {}. Rospy time: {}. Tlapse: {}".format(i, time_new, tlapse))

            rospy.sleep(1)

        #We do the pausing here
        # is_pause = False
        # if is_pause is False:
        #     self.request_pause()
        #     is_pause = True
        self.request_pause()

        time_prev = time_new
        time_new = rospy.get_time()
        tlapse = time_new - time_prev
        self.debug("Ticking instance: {}. Rospy time: {}. Tlapse: {}".format(i, time_new, tlapse))
        sum = 1 + 1
        self.debug("Sum: {}".format(sum))

        self.request_unpause()

        self.check_pause()

    def request_pause(self, is_pause=True):
        """
        Sends pause request to pause_simulation
        :return:
        """
        rospy.wait_for_service('/pause_queue_server')
        try:
            pause_request = rospy.ServiceProxy('/pause_queue_server', pauseSimulation)
            agent_id = 999
            resp = pause_request(is_pause, agent_id)
            return resp.pause_result
        except rospy.ServiceException as e:
            rospy.logerr(f"Pause service call failed: {e}")

    def request_unpause(self):
        """
        Sends unpause request to simulation
        :param is_pause:
        :return:
        """
        self.request_pause(is_pause=False)

    def print_state(self):
        """
        Prints current state of the environment based on info collected by central
        :return:
        """
        state = (self.sim_t, self.robots_location, self.robots_battery, self.tlapses, self.decay_rates)
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

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)


if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation/results')
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    CentralPlanner('central_planner').run_operation(filename)
    # CentralPlanner('central_planner').check_pause()
