#!/usr/bin/env python

"""
Correlated Multi-armed UCB

"""
import rospy
import actionlib
import json
import math
from time import process_time
import pickle
import numpy as np
from loss_fcns import *
from pruning import *
import project_utils as pu
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8, Float32
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.srv import GetPlan
from int_preservation.srv import flevel, flevelRequest
from status import areaStatus, battStatus, robotStatus
from reset_simulation import *

def request_fmeasure(area, msg=True):
    """
    Service request for F-measure
    :param msg:
    :return: (tlapse, F) #tlapse here is time lapsed since decay model of area has been updated
    """
    rospy.wait_for_service("/flevel_server_" + str(area))
    flevel_service = rospy.ServiceProxy("/flevel_server_" + str(area), flevel)
    request = flevelRequest(msg)
    result = flevel_service(request)
    return result.fmeasure


INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3  # GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html
SHUTDOWN_CODE = 99


class Robot:
    def __init__(self, node_name):
        """

        :param node_name:
        :param areas:
        :param est_distance_matrix:
        :param est_batt_consumption_matrix:
        """

        rospy.init_node(node_name, anonymous=True)

        # Parameters
        self.robot_id = rospy.get_param("~robot_id")
        self.debug_mode = rospy.get_param("/debug_mode")
        self.robot_velocity = rospy.get_param("/robot_velocity")  # Linear velocity of robot; we assume linear and angular are relatively equal
        self.gamma = rospy.get_param("/gamma")  # discount factor
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery")  # Max battery
        self.battery_reserve = rospy.get_param("/battery_reserve")  # Battery reserve

        f_thresh = rospy.get_param("/f_thresh")
        self.fsafe, self.fcrit = f_thresh

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = batt_consumed_per_time

        self.dec_steps = rospy.get_param("/dec_steps")  # STAR
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.nareas = rospy.get_param("/nareas")  # Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i + 1) for i in range(self.nareas)]  # list of int area IDs
        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data

        # Initialize variables
        charging_station_coords = rospy.get_param("~initial_pose_x"), rospy.get_param("~initial_pose_y")  # rospy.get_param("/charging_station_coords")
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        self.sampled_nodes_poses = [charging_pose_stamped]  # list container for sampled nodes of type PoseStamped

        # Pickle load the sampled area poses
        with open('{}.pkl'.format(rospy.get_param("/file_sampled_areas")), 'rb') as f:
            sampled_areas_coords = pickle.load(f)
        for area_coords in sampled_areas_coords['n{}_p{}'.format(self.nareas, rospy.get_param("/placement"))]:
            pose_stamped = pu.convert_coords_to_PoseStamped(area_coords)
            self.sampled_nodes_poses.append(pose_stamped)

        self.x, self.y = 0.0, 0.0  # Initialize robot pose
        self.charging_station = 0
        self.curr_loc = self.charging_station  # Initial location robot is the charging station
        self.battery = self.max_battery  # Initialize battery at max, then gets updated by subscribed battery topic
        self.best_decision = None
        self.dist_matrix = None
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True
        self.curr_fmeasures = dict()  # container of current F-measure of areas
        self.decay_rates_dict = dict()  # dictionary for decay rates
        self.recorded_fdata = dict()  # dictionary of recorded data collected during mission per area
        self.tlapses = dict()  # dictionary containing tlapse for each area since last restoration

        for area in self.areas:
            self.curr_fmeasures[area] = self.max_fmeasure
            self.decay_rates_dict[area] = None
            self.recorded_fdata[area] = list()
            self.tlapses[area] = 0

        self.decay_rates_counter = 0  # counter for stored decay rates; should be equal to number of areas
        self.decisions_made, self.decisions_accomplished, self.status_history = [], [], []  # record of data
        self.total_dist_travelled = 0  # total distance travelled
        self.process_time_counter = []  # container for time it took to come up with decision

        # Variables for UCB
        self.inference = rospy.get_param("/inference")
        self.exploration = rospy.get_param("/exploration")
        self.mean_losses = dict()
        self.recorded_losses = dict()
        self.counts_visited = np.zeros(self.nareas) #array of counts of number times area has been visited; zero-indexed
        for area in self.areas:
            self.mean_losses[area] = 0.0  # Initiate at 0
            self.recorded_losses[area] = list()
        self.correlation_info = rospy.get_param("/correlation_info")
        self.correlation_matrix = self.build_correlation_matrix()  # Initialize correlation matrix.

        # We sum this up
        self.environment_status = dict()
        for node in range(self.nareas + 1):
            self.environment_status[node] = 999

        # Publishers/Subscribers
        rospy.Subscriber('/robot_{}/odom'.format(self.robot_id), Odometry, self.distance_travelled_cb, queue_size=1)

        # Service request to move_base to get plan : make_Plan
        server = '/robot_' + str(self.robot_id) + '/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        rospy.Subscriber('/robot_{}/battery_status'.format(self.robot_id), Int8, self.battery_status_cb)
        rospy.Subscriber('/robot_{}/battery'.format(self.robot_id), Float32, self.battery_level_cb)

        for area in self.areas:
            rospy.Subscriber('/area_{}/decay_rate'.format(area), Float32, self.decay_rate_cb, area)
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb, area)  # REMARK: Here we assume that we have live measurements of the F-measures
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb, area)

        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)

        # Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base',
                                                              MoveBaseAction)
        self.robot_goal_client.wait_for_server()

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
        path = self.get_plan_request(area_i, area_j, tolerance)
        list_poses = self.decouple_path_poses(path)
        total_dist = self.compute_path_total_dist(list_poses)
        return total_dist

    def build_dist_matrix(self):
        """
        Builds the distance matrix among areas
        :return:
        """
        n = len(self.sampled_nodes_poses)
        self.dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                area_i, area_j = self.sampled_nodes_poses[i], self.sampled_nodes_poses[j]
                if area_i != area_j:
                    dist = self.compute_dist_bet_areas(area_i, area_j, self.tolerance)
                    self.dist_matrix[i, j] = dist

        self.debug("Dist matrix: {}".format(self.dist_matrix))

    # METHODS: Send robot to area
    def go_to_target(self, goal_idx):
        """
        Action client to move_base to move to target goal
        Goal is PoseStamped msg
        :param: goal_idx, index of goal in sampled_nodes_poses list
        :return:
        """
        goal = self.sampled_nodes_poses[goal_idx]
        self.send_robot_goal(goal)

    def send_robot_goal(self, goal):
        """
        Sends robot to goal via action client
        :param robot:
        :param goal: PoseStamped object
        :return:
        """
        movebase_goal = MoveBaseGoal()
        movebase_goal.target_pose = goal
        self.available = False
        action_goal_cb = (lambda state, result: self.action_send_done_cb(state, result, self.robot_id))
        self.robot_goal_client.send_goal(movebase_goal, done_cb=action_goal_cb, active_cb=self.action_send_active_cb)

    def action_send_active_cb(self):
        """
        Sets robot as unavailable when pursuing goal
        :return:
        """
        self.available = False
        self.update_robot_status(robotStatus.IN_MISSION)

    def action_send_done_cb(self, state, result, robot_id):
        """

        :param msg:
        :return:
        """
        if state == SUCCEEDED:
            self.debug("Arrived at mision_area: {}".format(self.mission_area))
            self.curr_loc = self.mission_area
            self.decisions_accomplished.append(self.mission_area)
            self.best_decision = None
            if self.mission_area == self.charging_station:
                self.update_robot_status(robotStatus.CHARGING)
            else:
                data = request_fmeasure(self.curr_loc)
                measured_f = float(data)
                self.recorded_fdata[self.curr_loc].append(measured_f)
                loss = loss_fcn(self.max_fmeasure, measured_f)
                self.recorded_losses[self.curr_loc].append(loss)
                self.counts_visited[self.curr_loc-1] += 1 #zero-indexed array

                self.update_robot_status(robotStatus.RESTORING_F)

    def update_mean_loss(self, area):
        """
        Updates the mean loss of the area (or arm) using the recorded_loss
        :param area:
        :param recorded_loss:
        :return:
        """
        mean_loss = self.mean_losses[area]
        new_mean_loss = np.mean(self.recorded_losses[area])

        self.debug('Recorded losses: {}. Current mean: {}. New mean: {}'.format(self.recorded_losses[area], mean_loss,
                                                                                new_mean_loss))
        self.mean_losses[area] = new_mean_loss

    def mean_duration_decay(self, duration_matrix, area):
        """
        Measures the average duration that an area decays (when robot commits to another decision than restoring area)
        Note that each column of the duration matrix represents that the corresponding area is the decision that the robot is committing to.
        We thus delete the column of the area, and then take the average to measure the average duration.
        :param duration_matrix:
        :param area:
        :return:
        """
        submatrix = np.delete(duration_matrix, area, axis=1)
        return np.mean(submatrix)

    def estimate_battery_params(self, decision, curr_battery, curr_loc, fmeasures, noise):
        """
        Measures battery consumption and feasible battery
        :param decision:
        :return:
        """
        # Battery consumed travel and preserve area (if not charging station)
        battery_consumption = self.consume_battery(start_area=curr_loc, next_area=decision,
                                                   curr_measure=fmeasures[decision],
                                                   noise=noise)
        # Battery consumed travel back to charging station
        battery_consumption_backto_charging_station = self.consume_battery(start_area=decision,
                                                                           next_area=self.charging_station,
                                                                           curr_measure=None,
                                                                           noise=noise)
        total_battery_consumption = battery_consumption + battery_consumption_backto_charging_station
        # Feasible batt = current batt - (consumption of decision + consumption to travel back to charging station)
        feasible_battery = curr_battery - total_battery_consumption

        return total_battery_consumption, feasible_battery


    def build_correlation_matrix(self):
        """
        Builds (nareas x nareas) correlation matrix given info on correlation between areas and magnitude.
        Note that correlation with own self is 1.0q
        Input:
            correlation_info: list of tuples where each tuple, [area_1, area_2, correlation], contains correlation between areas
        """
        corr_matrix = np.eye(self.nareas)

        #Correlation based on correlation info
        for corr in self.correlation_info:
            area_1, area_2 = corr[0]-1, corr[1]-1 #zero-indexed matrix
            corr_matrix[area_1, area_2] = corr[2]

        return corr_matrix

    def get_most_visited_areas(self):
        """
        Returns the most visited areas
        """
        most_visited = list()
        for area in self.areas:
            if self.counts_visited[area-1] >= math.floor(np.sum(self.counts_visited)/self.nareas):
                most_visited.append(area)
        return most_visited

    def get_max_empirical_mean(self, most_visited):
        """
        Returns max empirical mean among the most visited areas
        """
        mean_losses = list()
        for area in most_visited:
            mean_losses.append(self.mean_losses[area])
        max_mean = max(mean_losses)
        return max_mean

    def empirical_pseudo_reward(self, area, comp_area):
        """
        Returns pseudo-reward
        """
        reward = self.mean_losses[area] * self.correlation_matrix[area-1, comp_area-1]
        return reward

    def greedy_best_decision(self):
        """
        Part 1: Find the competitive arms
        #1. Find set S, most visited areas
            > prereq a counting mechanism for each area
                + can be counted by the length of each recorded data
                + or simply a counter    
            
        #2. Measure max empirical mean among S, max_emp_mean
            > measure the mean of each arm in S
            > find the max mean
        #3. Init A, list of competitive arms             
        #4. Populate A
        for j in J areas:
            Init pseudo_rewards as list
            for l in S where l is correlated with j (corr_matrix != 0):
                measure empirical pseudo-reward, z, and append to pseudo_rewards
                    > method to measure empirical pseduo-reward
            if len(pseudo_rewards) > 0:
                min = min(pseudo_rewards) #Note: Ensure that pseudo_rewards is non-empty
                if min >= max_emp_mean:
                    append j in A
        #5. Sanity check: A should never be non-empty            
        """

        """
        Part 1: Find competitive arms
        """
        most_visited_areas = self.get_most_visited_areas()
        max_empirical_mean = self.get_max_empirical_mean(most_visited_areas)
        self.debug("Counts of areas visited: {}".format(self.counts_visited))
        self.debug("Most visited arms: {}. Max empirical mean: {}".format(most_visited_areas, max_empirical_mean))
        competitive_arms = list()

        for area in self.areas:
            pseudo_rewards_list = list()
            for comp_area in most_visited_areas:
                if self.correlation_matrix[area-1, comp_area-1] != 0: #Note: zero-indexed matrix
                    z = self.empirical_pseudo_reward(area, comp_area)
                    pseudo_rewards_list.append(z)
                if len(pseudo_rewards_list) > 0:
                    max_z = max(pseudo_rewards_list)
                    self.debug("Area: {}. Min pseudo-reward: {}. <= max mean: {}".format(area, max_z, max_z <= max_empirical_mean))
                    if max_z <= max_empirical_mean:
                        competitive_arms.append(area)
        competitive_arms = list(set(competitive_arms))
        self.debug("Competitive arms: {}".format(competitive_arms))

        """
        Part 2: Apply UCB among the competitive arms, picking the one with least mean loss
        """
        # Measure duration matrix
        duration_matrix = self.dist_matrix / self.robot_velocity

        # Measure the average duration an area decays
        # Estimates the time/duration it takes to areas
        mean_duration_decay_dict = dict()
        for area in self.areas:
            mean_duration_decay_dict[area] = self.mean_duration_decay(duration_matrix, area)

        # Evaluate decision among competitive arms
        decision_array = []
        for decision in competitive_arms:
            # Battery consumption
            battery_consumption, feasible_battery = self.estimate_battery_params(decision, self.battery, self.curr_loc,
                                                                                 self.curr_fmeasures, self.noise)
            self.debug("Batt consumption: {}. Feasible batt: {}".format(battery_consumption, feasible_battery))

            if not prune(self.battery, battery_consumption, self.battery_reserve) and decision != self.curr_loc:
                bound = np.sqrt(2 * np.log(sum(self.counts_visited) + 1) / (self.counts_visited[decision-1] + 1e-5))
                ucb_value = self.mean_losses[decision] - self.exploration * bound
                self.debug("Feasible decision, Mean loss, Feasible battery: {}, {}, {}".format(decision, ucb_value,
                                                                                               feasible_battery))
                decision_array.append((decision, ucb_value, feasible_battery))

        best_decision = self.charging_station

        if len(decision_array) > 0:
            best_decision = self.get_best_decision(decision_array)

        return best_decision

    def compute_duration(self, start_area, next_area, curr_measure, restoration, noise):
        """
        Computes (time) duration of operation, which includes travelling distance plus restoration, if any
        :param distance:
        :param restoration: restore a measure (if not None) back to full measure per second
        :param noise: expected noise in distance travelled
        :return:
        """

        # Travel distance
        distance = self.dist_matrix[int(start_area), int(next_area)]
        distance += noise * distance  # distance + noise
        time = (distance / self.robot_velocity)

        # If next area is not the charging station: the restoration is the f-measure; else, the restoration is the battery level
        if next_area != self.charging_station:
            max_restore = self.max_fmeasure
        else:
            max_restore = self.max_battery

        # Restoration time: If there is need for restoration
        if (curr_measure is not None) and (restoration is not None):
            restore_time = (max_restore - curr_measure) / restoration
            time += restore_time

        return time

    def consume_battery(self, start_area, next_area, curr_measure, noise):
        """
        Consumes curr_battery for the duration of the operation.
        This duration includes the distance plus F-measure restoration, if any
        :param curr_battery:
        :param duration:
        :return:
        """

        # Batt consumed in travel
        distance = self.dist_matrix[int(start_area), int(next_area)]
        distance += noise * distance
        travel_time = (distance / self.robot_velocity)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time

        if next_area != self.charging_station:
            battery_consumed += self.batt_consumed_per_restored_f * (self.max_fmeasure - curr_measure)

        return battery_consumed

    def adjust_fmeasures(self, fmeasures, visit_area, duration):
        """
        Adjusts the F-measures of all areas in robot's mind. The visit area will be restored to max, while the other areas will decay for
        t duration. Note that the charging station is not part of the areas to monitor. And so, if the visit_area is the
        charging station, then all of the areas will decay as duration passes by.
        :param fmeasures:
        :param visit_area:
        :param t:
        :return:
        """
        for area in self.areas:
            if area == visit_area:
                fmeasures[area] = self.max_fmeasure
            else:
                tlapse_decay = get_time_given_decay(self.max_fmeasure, fmeasures[area],
                                                    self.decay_rates_dict[area]) + duration
                fmeasures[area] = decay(self.decay_rates_dict[area], tlapse_decay, self.max_fmeasure)

        return fmeasures

    def compute_net_loss(self, fmeasures):
        """
        Computes the net loss, (i.e., the sum of losses) of the fmeasures, which is a consequence of a decision
        Steps:
            1. Computes the loss for each of the F-measure of the areas
            2. Sums up the losses to get the cost of the decision
        :param fmeasures:
        :return:
        """
        netloss = compute_cost_fmeasures(fmeasures, self.fsafe, self.fcrit)
        return netloss

    def get_best_decision(self, dec_arr):
        """
        Returns the best decision in an array by sorting forecasted_loss ascendingly first then by remaining feasible battery.
        :param tree:
        :return:
        """
        # Sort the decisions: the cost is key while the value is decision
        sorted_decisions = sorted(dec_arr, key=lambda x: (x[-2], -x[-1]))
        self.debug("Decisions sorted by mean loss, feasible batt: {}".format(sorted_decisions))
        self.debug("Best decision (branch info): {}".format(sorted_decisions[0]))
        best_decision = sorted_decisions[0][0]  # pick the decision with least net loss and most available feasible battery
        return best_decision

    def update_tlapses_areas(self):
        """
        Lapses all time elapsed for each area
        :return:
        """
        for area in self.areas:
            self.tlapses[area] += 1
        self.debug("Time elapsed since last restored: {}".format(self.tlapses))

    # Methods: Run operation
    def run_operation(self, filename, freq=1):
        """
        :return:
        """

        if self.robot_id == 0:
            rate = rospy.Rate(freq)
            while self.decay_rates_counter < self.nareas and len(self.sampled_nodes_poses) != self.nareas + 1:
                self.debug("Insufficient data. Decay rates: {}/{}. Sampled nodes poses: {}/{}".format(
                    self.decay_rates_counter, self.nareas,
                    len(self.sampled_nodes_poses), self.nareas + 1))
                rate.sleep()  # Data for decay rates haven't registered yet

            self.debug("Sufficent data. Decay rates: {}. Sampled nodes poses: {}".format(self.decay_rates_dict,
                                                                                         self.sampled_nodes_poses))  # Prior knowledge of decay rates
            self.build_dist_matrix()
            t = 0
            while not rospy.is_shutdown() and t < self.t_operation:
                self.robot_status_pub.publish(self.robot_status)
                self.status_history.append(self.robot_status)
                if self.robot_status == robotStatus.IDLE.value:
                    self.debug('Robot idle')
                    if self.dist_matrix is not None:
                        self.update_robot_status(robotStatus.READY)

                elif self.robot_status == robotStatus.READY.value:
                    self.debug('Robot ready')
                    think_start = process_time()
                    self.think_decisions()
                    think_end = process_time()
                    think_elapsed = self.time_elapsed(think_start, think_end)
                    self.process_time_counter.append(think_elapsed)
                    self.debug('Best decision: {}. Process time: {}s'.format(self.best_decision, think_elapsed))
                    self.update_robot_status(robotStatus.IN_MISSION)

                elif self.robot_status == robotStatus.IN_MISSION.value:
                    self.debug('Robot in mission. Total distance travelled: {}'.format(self.total_dist_travelled))
                    if self.available:
                        self.commence_mission()

                elif self.robot_status == robotStatus.CHARGING.value:
                    self.debug('Waiting for battery to charge up')

                elif self.robot_status == robotStatus.RESTORING_F.value:
                    self.debug('Restoring F-measure')

                elif self.robot_status == robotStatus.CONSIDER_REPLAN.value:
                    self.debug('Consider re-plan...')
                    self.debug("Mission area: {}. Current mean losses: {}".format(self.mission_area,
                                                                                  self.mean_losses))
                    self.update_mean_loss(self.mission_area)
                    self.update_robot_status(robotStatus.IN_MISSION)  # Verified

                if len(self.decisions_made)>0 or (self.robot_status != robotStatus.IDLE.value) and (
                        self.robot_status != robotStatus.READY.value) and (
                        self.robot_status != robotStatus.CONSIDER_REPLAN.value):
                    self.update_tlapses_areas()  # Update the tlapse per area
                    self.compute_curr_fmeasures()
                t += 1
                rate.sleep()

            # Store results
            self.update_robot_status(robotStatus.SHUTDOWN)
            self.robot_status_pub.publish(self.robot_status)
            self.status_history.append(self.robot_status)

            # Wait before all other nodes have finished dumping their data
            if self.save:
                if self.inference is not None:
                    pu.dump_data((self.recorded_fdata, self.recorded_losses),
                                 '{}_robot{}_recorded_data'.format(filename, self.robot_id))

                pu.dump_data(self.process_time_counter, '{}_robot{}_process_time'.format(filename, self.robot_id))
                pu.dump_data(self.decisions_made, '{}_robot{}_decisions'.format(filename, self.robot_id))
                pu.dump_data((self.decisions_accomplished, self.total_dist_travelled),
                             '{}_robot{}_decisions_acc_travel'.format(filename, self.robot_id))
                pu.dump_data(self.status_history, '{}_robot{}_status_history'.format(filename, self.robot_id))
                self.debug("Dumped all data.".format(self.robot_id))
            self.shutdown(sleep=10)

    def think_decisions(self):
        """
        Thinks of the best decision before starting mission
        :return:
        """
        self.best_decision = self.greedy_best_decision()  # So inside here we can

    def time_elapsed(self, think_start, think_end):
        """
        Process time it took to process
        :param think_end:
        :return:
        """
        return think_end - think_start

    def commence_mission(self):
        """
        Commences mission
        :return:
        """
        if self.send2_next_area() == 0:
            self.update_robot_status(robotStatus.IDLE)

    def send2_next_area(self):
        """
        Sends the robot to the next area in the optimal path/decision:
        :return:
        """
        if self.best_decision is not None:
            self.mission_area = self.best_decision
            self.mission_area_pub.publish(self.mission_area)
            self.debug('Heading to: {}. {}'.format(self.mission_area, self.sampled_nodes_poses[self.mission_area]))
            self.decisions_made.append(self.mission_area)  # store decisions made
            self.go_to_target(self.mission_area)
            return 1
        return 0

    def update_robot_status(self, status):
        """
        Updates robot status
        :param status:
        :return:
        """
        self.robot_status = status.value

    def distance_travelled_cb(self, msg):
        # Updates total distance travelled
        # Sets curr robot pose
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.total_dist_travelled += math.dist((self.x, self.y), (x, y))
        self.x, self.y = x, y

    def battery_level_cb(self, msg):
        """
        Callback for battery level
        :param msg:
        :return:
        """
        self.battery = msg.data

    def battery_status_cb(self, msg):
        """

        :param msg:
        :return:
        """
        self.environment_status[self.charging_station] = msg.data
        if msg.data == battStatus.FULLY_CHARGED.value:
            if self.robot_id == 0: self.debug("Fully charged!")
            self.available = True
            self.update_robot_status(robotStatus.IN_MISSION)

    def area_status_cb(self, msg, area_id):
        """

        :param msg:
        :return:
        """
        self.environment_status[area_id] = msg.data
        if msg.data == areaStatus.RESTORED_F.value:
            self.available = True
            self.tlapses[area_id] = 0
            if self.robot_id == 0: self.debug("Area {} fully restored! tlapse reset...".format(area_id))

            if (self.inference is not None) and (self.inference != 'oracle'):
                self.update_robot_status(robotStatus.CONSIDER_REPLAN)
            else:
                self.update_robot_status(robotStatus.IN_MISSION)

    def decay_rate_cb(self, msg, area_id):
        """
        Store decay rate
        :param msg:
        :param area_id:
        :return:
        """
        # Store the decay rates at instance, (prior knowledge)
        if self.decay_rates_dict[area_id] == None and msg.data is not None:
            if self.robot_id == 0: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates_dict[area_id] = msg.data
            self.decay_rates_counter += 1
        else:
            # If we are now on mission and oracle, we immediately update the decay rates for any evolution
            if self.inference == 'oracle':
                if self.decay_rates_dict[area_id] != msg.data: self.debug(
                    "Oracle knowledge, change in decay in area {}: {}".format(area_id, msg.data))
                self.decay_rates_dict[area_id] = msg.data  # A subscribed topic. Oracle knows exactly the decay rate happening in area

    def area_fmeasure_cb(self, msg, area_id):
        """
        Updates fmeasure of area
        :param msg:
        :param area_id:
        :return:
        """
        if self.inference == 'oracle':
            self.curr_fmeasures[area_id] = msg.data

    def compute_curr_fmeasures(self):
        """
        Computes current fmeasures based on tlapse and decay rates
        :return:
        """
        for area in self.areas:
           self.curr_fmeasures[area] = decay(self.decay_rates_dict[area], self.tlapses[area], self.max_fmeasure)
        self.debug("Used for computation. Tlapses: {}. Decay rates: {}".format(self.tlapses, self.decay_rates_dict))
        self.debug("Computed current f-measures: {}".format(self.curr_fmeasures))

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)


if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    Robot('correlated_ucb').run_operation(filename)