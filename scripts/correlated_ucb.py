#!/usr/bin/env python

"""
Correlated Multi-armed UCB
Paper reference: S. Gupta, S. Chaudhari, G. Joshi, and O. Ya˘gan, “Multi-armed
bandits with correlated arms,” IEEE Transactions on Information
Theory, 2021. DOI: 10.1109/TIT.2021.3081508.

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
from status import areaStatus, battStatus, robotStatus
from reset_simulation import *
from heuristic_fcns import *
from loss_fcns import *
from int_preservation.srv import clusterAssignment, clusterAssignmentResponse
from int_preservation.srv import areaAssignment
from int_preservation.srv import assignmentAccomplishment
from int_preservation.srv import registerRobot
from int_preservation.srv import pauseSimulation
from int_preservation.srv import flevel, flevelRequest
import random

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3  # GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html
SHUTDOWN_CODE = 99


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
        self.robot_velocity = rospy.get_param(
            "/robot_velocity")  # Linear velocity of robot; we assume linear and angular are relatively equal
        self.gamma = rospy.get_param("/gamma")  # discount factor
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery")  # Max battery
        self.battery_reserve = rospy.get_param("/battery_reserve")  # Battery reserve

        f_thresh = rospy.get_param("/f_thresh")
        self.fsafe, self.fcrit = f_thresh  # (safe, crit)

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = batt_consumed_per_time  # (travel, restoration)

        self.dec_steps = rospy.get_param("/dec_steps")
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data
        self.task_scheduler = rospy.get_param("/task_scheduler")  # task scheduler

        # Initialize variables
        self.init_x, self.init_y = rospy.get_param("~initial_pose_x"), rospy.get_param(
            "~initial_pose_y")  # Initialize robot pose
        self.debug("Init x, y {}".format((self.init_x, self.init_y)))
        charging_station_coords = self.init_x, self.init_y  # rospy.get_param("/charging_station_coords")
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        self.nodes_poses = [
            charging_pose_stamped]  # list container for sampled nodes of type PoseStamped, where 0 is the charging station for that robot

        # Pickle load the sampled area poses
        with open('{}.pkl'.format(rospy.get_param("/file_sampled_areas")), 'rb') as f:
            sampled_areas_coords = pickle.load(f)
        nareas = rospy.get_param("/nareas")
        for area_coords in sampled_areas_coords['n{}_p{}'.format(nareas, rospy.get_param("/placement"))]:
            pose_stamped = pu.convert_coords_to_PoseStamped(area_coords)
            self.nodes_poses.append(pose_stamped)

        self.x, self.y = self.init_x, self.init_y  # Initialize robot pose
        self.charging_station = 0
        self.curr_loc_idx = self.charging_station  # Initial location robot is the charging station
        self.battery = self.max_battery  # Initialize battery at max, then gets updated by subscribed battery topic
        self.best_decision_idx = None
        self.dist_matrix = None
        # self.sampled_nodes_poses = None
        self.mission_area_idx = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True

        self.state = list()  # list of states

        self.decisions_made, self.decisions_accomplished, self.status_history = [], [], []  # record of data
        self.total_dist_travelled = 0  # total distance travelled
        self.process_time_counter = []  # container for time it took to come up with decision
        self.assigned_areas = None

        self.requested_pause = False  # indicator variable whether requested Stage to pause simulation

        # Publishers/Subscribers
        rospy.Subscriber('/robot_{}/odom'.format(self.robot_id), Odometry, self.distance_travelled_cb, queue_size=1)

        # Service request to move_base to get plan : make_Plan
        server = '/robot_' + str(self.robot_id) + '/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        rospy.Subscriber('/robot_{}/battery_status'.format(self.robot_id), Int8, self.battery_status_cb)
        rospy.Subscriber('/robot_{}/battery'.format(self.robot_id), Float32, self.battery_level_cb)

        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_idx_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)
        self.location_pub = rospy.Publisher('/robot_{}/location'.format(self.robot_id), Int8, queue_size=1)

        # Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base',
                                                              MoveBaseAction)
        self.robot_goal_client.wait_for_server()

        # Server for assigned cluster to monitor/preserve
        self.cluster_assignment_server = rospy.Service("/cluster_assignment_server_" + str(self.robot_id),
                                                       clusterAssignment, self.cluster_assignment_cb)

        # Corr UCB variables
        self.inference = rospy.get_param("/inference")
        self.exploration = rospy.get_param("/exploration")
        self.correlation_info = rospy.get_param("/correlation_info")

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

    def register_to_central(self):
        """
        Register robot info to central
        :return:
        """
        if self.task_scheduler == "central_planner":
            rospy.wait_for_service('/robots_registry_server')
            try:
                register_request = rospy.ServiceProxy('/robots_registry_server', registerRobot)
                self.debug("Registering robot info to central (id, x, y): {}, {}, {}".format(self.robot_id, self.init_x,
                                                                                             self.init_y))
                resp = register_request(self.robot_id, self.init_x, self.init_y)
                self.debug("Registry to central success: {}".format(resp.registered))
            except rospy.ServiceException as e:
                rospy.logerr(f"Register to central service call failed: {e}")

    def area_assignment_notice(self, assigned_areas):
        """
        Tags areas with the assigned robot
        :param assigned_areas:
        :return:
        """
        for area_id in assigned_areas:
            rospy.wait_for_service("/area_assignment_server_" + str(area_id))
            try:
                area_assign = rospy.ServiceProxy("/area_assignment_server_" + str(area_id), areaAssignment)
                resp = area_assign(self.robot_id)
                self.debug("Noted assigned Area {}".format(resp))
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")

    def cluster_assignment_cb(self, msg):
        """
        Sets the cluster assignment as the areas for restoration/preservation
        :param msg:
        :return:
        """
        self.robot_status = robotStatus.IDLE.value  # Halts all operations of the robot to re-consider new assignment
        assigned_areas, decay_rates, tlapses = msg.cluster, msg.decay_rates, msg.tlapses
        assigned_areas, decay_rates, tlapses = list(assigned_areas), list(decay_rates), list(tlapses)
        self.debug(
            "Cluster assignment: {}, {}. Decay rates: {}. Tlapses: {}".format(type(assigned_areas), assigned_areas,
                                                                              decay_rates, tlapses))
        self.instantiate_variables(assigned_areas=assigned_areas, decay_rates=decay_rates, tlapses=tlapses)
        self.is_assigned = True
        return clusterAssignmentResponse(self.is_assigned)

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
            self.curr_loc_idx = self.mission_area_idx
            mission_area = self.get_assigned_area_id(self.mission_area_idx)
            self.debug("Arrived at mision_area: {}".format(mission_area))
            self.decisions_accomplished.append(mission_area)
            self.best_decision_idx = None

            if self.mission_area_idx == self.charging_station:
                self.update_robot_status(robotStatus.CHARGING)
            else:
                data = request_fmeasure(mission_area)
                measured_f = float(data)
                self.recorded_fdata[self.curr_loc_idx].append(measured_f)
                loss = loss_fcn(self.max_fmeasure, measured_f)
                self.recorded_losses[self.curr_loc_idx].append(loss)
                self.counts_visited[self.curr_loc_idx - 1] += 1  # zero-indexed array

                self.update_robot_status(robotStatus.RESTORING_F)

    def update_mean_loss(self, area_idx):
        """
        Updates the mean loss of the area (or arm) using the recorded_loss
        :param area:
        :param recorded_loss:
        :return:
        """
        mean_loss = self.mean_losses[area_idx]
        new_mean_loss = np.mean(self.recorded_losses[area_idx])

        # self.debug('Recorded losses: {}. Current mean: {}. New mean: {}'.format(self.recorded_losses[area], mean_loss,
        #                                                                         new_mean_loss))
        self.mean_losses[area_idx] = new_mean_loss

    def mean_duration_decay(self, duration_matrix, area_idx):
        """
        Measures the average duration that an area decays (when robot commits to another decision than restoring area)
        Note that each column of the duration matrix represents that the corresponding area is the decision that the robot is committing to.
        We thus delete the column of the area, and then take the average to measure the average duration.
        :param duration_matrix:
        :param area:
        :return:
        """
        submatrix = np.delete(duration_matrix, area_idx, axis=1)
        return np.mean(submatrix)

    def estimate_battery_params(self, decision, curr_battery, curr_loc, fmeasures, noise):
        """
        Measures battery consumption and feasible battery
        :param decision:
        :return:
        """
        # Battery consumed travel and preserve area (if not charging station)
        # self.debug("Estimate battery params fmeasure, decision: {}, {}".format(fmeasures, decision))
        battery_consumption = self.consume_battery(start_area_idx=curr_loc, next_area_idx=decision,
                                                   curr_measure=fmeasures[decision],
                                                   noise=noise)
        # Battery consumed travel back to charging station
        battery_consumption_backto_charging_station = self.consume_battery(start_area_idx=decision,
                                                                           next_area_idx=self.charging_station,
                                                                           curr_measure=None,
                                                                           noise=noise)
        total_battery_consumption = battery_consumption + battery_consumption_backto_charging_station
        # Feasible batt = current batt - (consumption of decision + consumption to travel back to charging station)
        feasible_battery = curr_battery - total_battery_consumption

        return total_battery_consumption, feasible_battery

    def get_most_visited_areas(self):
        """
        Returns the most visited areas
        """
        most_visited = list()
        for area_idx in self.areas:
            if self.counts_visited[area_idx - 1] >= math.floor(np.sum(self.counts_visited) / self.nareas):
                most_visited.append(area_idx)
        return most_visited

    def get_max_empirical_mean(self, most_visited):
        """
        Returns max empirical mean among the most visited areas
        """
        mean_losses = list()
        for area_idx in most_visited:
            mean_losses.append(self.mean_losses[area_idx])
        max_mean = max(mean_losses)
        return max_mean

    def empirical_pseudo_reward(self, area_idx, comp_area_idx):
        """
        Returns pseudo-reward
        """
        reward = self.mean_losses[area_idx] * self.correlation_matrix[area_idx - 1, comp_area_idx - 1]
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
                    > method to measure empirical pseudo-reward
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
        # self.debug("Counts of areas visited: {}".format(self.counts_visited))
        # self.debug("Most visited arms: {}. Max empirical mean: {}".format(most_visited_areas, max_empirical_mean))
        competitive_arms = list()

        for area_idx in self.areas:
            pseudo_rewards_list = list()
            for comp_area_idx in most_visited_areas:
                if self.correlation_matrix[area_idx - 1, comp_area_idx - 1] != 0:  # Note: zero-indexed matrix
                    z = self.empirical_pseudo_reward(area_idx, comp_area_idx)
                    pseudo_rewards_list.append(z)
                if len(pseudo_rewards_list) > 0:
                    max_z = max(pseudo_rewards_list)
                    # self.debug("Area: {}. Min pseudo-reward: {}. <= max mean: {}".format(area, max_z, max_z <= max_empirical_mean))
                    if max_z <= max_empirical_mean:
                        competitive_arms.append(area_idx)
        competitive_arms = list(set(competitive_arms))
        # self.debug("Competitive arms: {}".format(competitive_arms))

        """
        Part 2: Apply UCB among the competitive arms, picking the one with least mean loss
        """
        # Measure duration matrix
        duration_matrix = self.dist_matrix / self.robot_velocity

        # Measure the average duration an area decays
        # Estimates the time/duration it takes to areas
        mean_duration_decay_dict = dict()
        for area_idx in self.areas:
            mean_duration_decay_dict[area_idx] = self.mean_duration_decay(duration_matrix, area_idx)

        # Evaluate decision among competitive arms
        decision_array = []
        for decision_idx in competitive_arms:
            # Battery consumption
            battery_consumption, feasible_battery = self.estimate_battery_params(decision_idx, self.battery,
                                                                                 self.curr_loc_idx,
                                                                                 self.curr_fmeasures, self.noise)
            # self.debug("Batt consumption: {}. Feasible batt: {}".format(battery_consumption, feasible_battery))

            if not prune(self.battery, battery_consumption, self.battery_reserve) and decision_idx != self.curr_loc_idx:
                bound = np.sqrt(
                    2 * np.log(sum(self.counts_visited) + 1) / (self.counts_visited[decision_idx - 1] + 1e-5))
                ucb_value = self.mean_losses[decision_idx] - self.exploration * bound
                # self.debug("Feasible decision, Mean loss, Feasible battery: {}, {}, {}".format(decision, ucb_value,
                #                                                                                feasible_battery))
                decision_array.append((decision_idx, ucb_value, feasible_battery))

        best_decision_idx = self.charging_station

        if len(decision_array) > 0:
            best_decision_idx = self.get_best_decision(decision_array)

        return best_decision_idx

    def compute_duration(self, start_area_idx, next_area_idx, curr_measure, restoration, noise):
        """
        Computes (time) duration of operation, which includes travelling distance plus restoration, if any
        :param distance:
        :param restoration: restore a measure (if not None) back to full measure per second
        :param noise: expected noise in distance travelled
        :return:
        """

        # Travel distance
        distance = self.dist_matrix[int(start_area_idx), int(next_area_idx)]
        distance += noise * distance  # distance + noise
        time = (distance / self.robot_velocity)

        # If next area is not the charging station: the restoration is the f-measure; else, the restoration is the battery level
        if next_area_idx != self.charging_station:
            max_restore = self.max_fmeasure
        else:
            max_restore = self.max_battery

        # Restoration time: If there is need for restoration
        if (curr_measure is not None) and (restoration is not None):
            restore_time = (max_restore - curr_measure) / restoration
            time += restore_time

        return time

    def consume_battery(self, start_area_idx, next_area_idx, curr_measure, noise):
        """
        Estimates battery consumption for the duration of the visit next_area from start_area.
        This duration includes the distance plus F-measure restoration, if any
        """

        # Batt consumed in travel
        distance = self.dist_matrix[int(start_area_idx), int(next_area_idx)]
        distance += noise * distance
        travel_time = (distance / self.robot_velocity)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time

        # Batt consumed in area restoration
        if next_area_idx != self.charging_station:
            battery_consumed += self.batt_consumed_per_restored_f * (self.max_fmeasure - curr_measure)

        return battery_consumed

    def adjust_fmeasures(self, fmeasures, visit_area_idx, duration):
        """
        Adjusts the F-measures of all areas in robot's mind. The visit area will be restored to max, while the other areas will decay for
        t duration. Note that the charging station is not part of the areas to monitor. And so, if the visit_area is the
        charging station, then all of the areas will decay as duration passes by.
        :param fmeasures:
        :param visit_area:
        :param t:
        :return:
        """

        """
        Verify:
            1. visit_area index - area_idx
            2. fmeasure area key - area_idx
            3. decay_rates_dict key - area_idx
        """

        for area_idx in self.areas:
            if area_idx == visit_area_idx:
                fmeasures[area_idx] = self.max_fmeasure
            else:
                tlapse_decay = get_time_given_decay(self.max_fmeasure, fmeasures[area_idx],
                                                    self.decay_rates_dict[area_idx]) + duration
                fmeasures[area_idx] = decay(self.decay_rates_dict[area_idx], tlapse_decay, self.max_fmeasure)

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
        Returns the area index of the best decision in an array by sorting forecasted_loss ascendingly first then by remaining feasible battery.
        :param tree:
        :return:
        """
        # Sort the decisions: the cost is key while the value is decision
        sorted_decisions = sorted(dec_arr, key=lambda x: (x[-2], -x[-1]))
        # self.debug("Decisions sorted by cost: {}".format(sorted_decisions))
        # self.debug("Best decision (branch info): {}".format(sorted_decisions[0]))
        best_decision_idx = sorted_decisions[0][
            0]  # pick the decision with least net loss and most available feasible battery

        return best_decision_idx

    def update_tlapses_areas(self):
        """
        Lapses all time elapsed for each area
        :return:
        """
        self.sim_t += 1
        for area_idx in self.tlapses:
            self.tlapses[area_idx] += 1
        # self.debug("Sim t: {}. Time elapsed since last restored: {}".format(self.sim_t, self.tlapses))
        self.debug("Time elapsed since last restored: {}".format(self.tlapses))

    def extract_sampled_node_poses(self, assigned_areas):
        """
        Extracts the node poses of the assigned clusters into a list where index 0 is the charging station pose
        :param cluster:
        :return:
        """
        poses = list()
        poses.append(self.nodes_poses[0])
        for area_id in assigned_areas:
            poses.append(self.nodes_poses[area_id])
        return poses

    """
    NOTE: We use area index in computations containers (except sampled_nodes_poses), 
        while we use assigned area ids in area subscriptions and action sending, as well as debugs
        Area index (area_idx): 1, 2, 3, ...
        Actual area id (area_id): The actual ones from the environment, wherein the assignment is a subset of
        Note that for containers that are list in nature, we may index specific/corresponding area index as [area_idx-1]
    """

    def unsubscribe_previous_area_topics(self):
        """
        Unsubscribes to previously assigned areas topics:
            > fmeasure
            > status
        :return:
        """
        for area_id in self.subscribe_fmeasures:
            if self.subscribe_fmeasures[area_id] is not None:
                self.subscribe_fmeasures[area_id].unsubscribe()
                self.debug("Unsubscribed from previously assigned Area {} fmeasure topic".format(area_id))

        for area_id in self.subscribe_statuses:
            if self.subscribe_statuses[area_id] is not None:
                self.subscribe_statuses[area_id].unsubscribe()
                self.debug("Unsubscribed from previously assigned Area {} status topic".format(area_id))

    def build_correlation_matrix(self):
        """
        Builds (nareas x nareas) correlation matrix given info on correlation between areas and magnitude.
        Note that correlation with own self is 1.0q
        """
        # corr_matrix = np.eye(self.nareas)
        # Correlation based on correlation info
        # for corr in self.correlation_info:
        #     area_1, area_2 = corr[0]-1, corr[1]-1 #zero-indexed matrix
        #     corr_matrix[area_1, area_2] = corr[2]

        # Assume assigned clustered areas are correlated with each other since they belong to the same cluster
        corr_matrix = np.ones((self.nareas, self.nareas))

        return corr_matrix

    def instantiate_variables(self, assigned_areas, decay_rates, tlapses):
        """
        Instantiates variables for every new assigned areas/cluster
        :param self:
        :param assigned_areas:
        :return:
        """
        self.debug("Instantiating variables")

        self.assigned_areas = assigned_areas  # Has the same list indexing with self.areas
        self.nareas = len(self.assigned_areas)  # Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i + 1) for i in range(self.nareas)]  # list of int area indexes, starting at index=1

        # Send notice to areas about their robot assignment
        self.area_assignment_notice(self.assigned_areas)

        self.curr_fmeasures = dict()  # container of current F-measure of areas
        self.decay_rates_dict = dict()  # dictionary for decay rates
        self.recorded_fdata = dict()  # dictionary of recorded data collected during mission per area
        self.tlapses = dict()  # dictionary containing tlapses of areas

        for area_idx in self.areas:
            self.decay_rates_dict[area_idx] = decay_rates[area_idx - 1]  # Instantiate provided decay rates
            self.tlapses[area_idx] = tlapses[area_idx - 1]  # Instantiate provided tlapses
            self.recorded_fdata[area_idx] = list()

        # Variables for UCB
        self.mean_losses = dict()
        self.recorded_losses = dict()
        self.counts_visited = np.zeros(
            self.nareas)  # array of counts of number times area has been visited; zero-indexed
        for area_idx in self.areas:
            self.mean_losses[area_idx] = 0.0  # Initiate at 0
            self.recorded_losses[area_idx] = list()
        self.correlation_matrix = self.build_correlation_matrix()  # Initialize correlation matrix.

        self.subscribe_fmeasures = dict()
        self.subscribe_statuses = dict()

        for area_idx in self.areas:
            self.curr_fmeasures[area_idx] = self.max_fmeasure #Initialize
            area_id = self.get_assigned_area_id(area_idx)
            self.subscribe_fmeasures[area_id] = rospy.Subscriber('/area_{}/fmeasure'.format(area_id), Float32,
                                                                 self.area_fmeasure_cb,
                                                                 area_id)  # REMARK: If oracle knowledge this is toggled to receive actual F
            self.subscribe_statuses[area_id] = rospy.Subscriber('/area_{}/status'.format(area_id), Int8, self.area_status_cb, area_id)


        # Sampled node poses and distance matrix
        self.sampled_nodes_poses = self.extract_sampled_node_poses(self.assigned_areas)
        self.build_dist_matrix()

    def get_assigned_area_index(self, area_id):
        """
        Gets area index of assigned area, (note area index starts at 1)
        :param area_id:
        :return:
        """
        area_idx = self.assigned_areas.index(area_id) + 1
        return area_idx

    def get_assigned_area_id(self, area_idx):
        """
        Gets actual assigned area id given area index, (note area index starts at 1)
        :param self:
        :param area_idx:
        :return:
        """
        area_id = self.assigned_areas[area_idx - 1]
        return area_id

    def get_current_state(self):
        """
        Retrieves current state
        :return:
        """
        tlapses = dict()
        decay_rates = dict()
        for area_idx in self.areas:
            area_id = self.get_assigned_area_id(area_idx)
            tlapses[area_id] = self.tlapses[area_idx]
            decay_rates[area_id] = self.decay_rates_dict[area_idx]

        # state = (self.sim_t, self.get_assigned_area_id(self.curr_loc_idx), self.battery, tlapses, decay_rates)
        state = (self.get_assigned_area_id(self.curr_loc_idx), self.battery, tlapses, decay_rates)
        return state

    # Methods: Run operation
    def run_operation(self, filename, freq=1):
        """
        :return:
        """

        if self.robot_id < 999:  # Not a dummy robot
            rate = rospy.Rate(freq)
            rospy.sleep(5)  # Wait for nodes to register

            self.register_to_central()

            while self.dist_matrix is None:
                self.debug("Initialization: No cluster assignment yet. Waiting for assignment...")
                rospy.sleep(1)

            self.sim_t = 0
            # while not rospy.is_shutdown() and self.sim_t<self.t_operation:
            while not rospy.is_shutdown():
                curr_state = self.get_current_state()
                self.state.append(curr_state)
                self.debug("Curr state: {}".format(curr_state))
                self.robot_status_pub.publish(self.robot_status)
                self.location_pub.publish(self.get_assigned_area_id(self.curr_loc_idx))
                self.status_history.append(self.robot_status)

                if self.robot_status == robotStatus.IDLE.value:  # Here, robot is available but unassigned
                    self.debug('Robot idle')
                    self.debug("With dist_matrix: {}".format(bool(self.dist_matrix is not None)))
                    if self.dist_matrix is not None:
                        self.update_robot_status(robotStatus.READY)

                elif self.robot_status == robotStatus.READY.value:
                    self.debug('Robot ready')

                    self.request_pause(True)  # Request pause simulation

                    think_start = process_time()
                    self.think_decisions()
                    think_end = process_time()
                    think_elapsed = self.time_elapsed(think_start, think_end)

                    self.process_time_counter.append(think_elapsed)

                    self.debug(
                        'Best decision: {}. Process time: {}s'.format(self.get_assigned_area_id(self.best_decision_idx),
                                                                      think_elapsed))

                    if self.requested_pause:
                        self.request_pause(False)

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
                    self.debug(
                        "Mission area: {}. Current mean losses: {}".format(self.mission_area_idx, self.mean_losses))
                    self.update_mean_loss(self.mission_area_idx)
                    self.update_robot_status(robotStatus.IN_MISSION)  # Verified

                # Update tlapse for each area when all nodes have registered
                if len(self.decisions_made) > 0 or (self.robot_status != robotStatus.IDLE.value) and (
                        self.robot_status != robotStatus.READY.value) and (
                        self.robot_status != robotStatus.CONSIDER_REPLAN.value):
                    self.update_tlapses_areas()  # Update the tlapse per area
                    self.compute_curr_fmeasures()

                if self.save:
                    if self.inference is not None:
                        pu.dump_data((self.recorded_fdata, self.recorded_losses),
                                     '{}_robot{}_recorded_data'.format(filename, self.robot_id))
                    pu.dump_data(self.state, '{}_environment_state'.format(filename))
                    pu.dump_data(self.process_time_counter, '{}_robot{}_process_time'.format(filename, self.robot_id))
                    pu.dump_data(self.decisions_made, '{}_robot{}_decisions'.format(filename, self.robot_id))
                    pu.dump_data((self.decisions_accomplished, self.total_dist_travelled),
                                 '{}_robot{}_decisions_acc_travel'.format(filename, self.robot_id))
                    pu.dump_data(self.status_history, '{}_robot{}_status_history'.format(filename, self.robot_id))
                    self.debug("Dumped all data.".format(self.robot_id))
                rate.sleep()

    """
    NOTE: For the subscribed/published topics and data storages, as well as self.debugs! use assigned area ids! All else area index
    """

    def request_pause(self, is_pause=True):
        """
        Sends pause request to pause_simulation
        :return:
        """
        self.requested_pause = is_pause
        rospy.wait_for_service('/pause_queue_server')
        try:
            pause_request = rospy.ServiceProxy('/pause_queue_server', pauseSimulation)
            agent_id = self.robot_id
            resp = pause_request(is_pause, agent_id)
            return resp.pause_result
        except rospy.ServiceException as e:
            rospy.logerr(f"Pause service call failed: {e}")

    def think_decisions(self):
        """
        Thinks of the best decision before starting mission
        :return:
        """
        self.best_decision_idx = self.greedy_best_decision()

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
        Sends the robot to the next area in the optimal path:
        :return:
        """
        if self.best_decision_idx is not None:
            self.mission_area_idx = self.best_decision_idx
            mission_area_id = self.charging_station
            if self.mission_area_idx != self.charging_station:
                mission_area_id = self.get_assigned_area_id(self.mission_area_idx)
            self.mission_area_idx_pub.publish(mission_area_id)
            self.debug('Heading to: {}. {}'.format(mission_area_id, self.sampled_nodes_poses[self.mission_area_idx]))
            self.decisions_made.append(mission_area_id)  # store decisions made
            self.go_to_target(self.mission_area_idx)
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
        if msg.data == battStatus.FULLY_CHARGED.value:
            if self.robot_id < 999: self.debug("Fully charged!")
            self.available = True
            self.update_robot_status(robotStatus.IN_MISSION)

    def notify_assignment_accomplishment(self, area_id):
        """
        Assigns clusters to robots
        :param clusters:
        :return:
        """

        rospy.wait_for_service("/assignment_accomplishment_server")
        try:
            notify_server = rospy.ServiceProxy("/assignment_accomplishment_server", assignmentAccomplishment)
            resp = notify_server(self.robot_id, area_id)
            self.debug("Robot: {}. Accomplished assignment: {}. Notified server: {}".format(self.robot_id, area_id,
                                                                                            resp.confirmation))
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def area_status_cb(self, msg, area_id):
        """

        :param msg:
        :return:
        """
        area_idx = self.get_assigned_area_index(area_id)
        status = msg.data

        if status == areaStatus.RESTORED_F.value:
            if self.robot_id < 999: self.debug("Area {} fully restored! tlapse reset...".format(area_id))
            self.tlapses[area_idx] = 0
            self.debug("Notifying server for accomplishment of restoring Area {}".format(area_id))
            self.notify_assignment_accomplishment(area_id)  # Notifies central that recent assignment is accomplished
            self.available = True

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
        area_idx = self.get_assigned_area_index(area_id)
        if self.assigned_areas is not None and self.decay_rates_dict[area_idx] == None and msg.data is not None:
            if self.robot_id < 999: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates_dict[area_idx] = msg.data

        else:
            # If we are now on mission and oracle, we immediately update the decay rates for any evolution
            if self.inference == 'oracle':
                if self.decay_rates_dict[area_idx] != msg.data: self.debug(
                    "Oracle knowledge, change in decay in area {}: {}".format(area_id, msg.data))
                self.decay_rates_dict[
                    area_idx] = msg.data  # A subscribed topic. Oracle knows exactly the decay rate happening in area

    def area_fmeasure_cb(self, msg, area_id):
        """
        Updates fmeasure of area
        :param msg:
        :param area_id:
        :return:
        """
        if self.inference == 'oracle':
            area_idx = self.get_assigned_area_index(area_id)
            self.curr_fmeasures[area_idx] = msg.data

    def compute_curr_fmeasures(self):
        """
        Computes current fmeasures based on tlapse and decay rates
        :return:
        """
        for area in self.areas:
            self.curr_fmeasures[area] = decay(self.decay_rates_dict[area], self.tlapses[area], self.max_fmeasure)

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)


if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    Robot('correlated_ucb').run_operation(filename)