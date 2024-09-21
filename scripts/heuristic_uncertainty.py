#!/usr/bin/env python

"""
Heuristic decision making under uncertainty

"""
import rospy
import math
from time import process_time
import pickle
import numpy as np
import actionlib
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
from heuristic_fcns import *
from infer_lstm import *
import json
import copy
import pandas as pd
from noise_models import *


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
SUCCEEDED = 3 #GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html
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

        #Parameters
        self.robot_id = rospy.get_param("~robot_id")
        self.debug_mode = rospy.get_param("/debug_mode")
        self.robot_velocity = rospy.get_param("/robot_velocity") #Linear velocity of robot; we assume linear and angular are relatively equal
        self.gamma = rospy.get_param("/gamma") #discount factor
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery") #Max battery
        self.battery_reserve = rospy.get_param("/battery_reserve") #Battery reserve

        f_thresh = rospy.get_param("/f_thresh")
        self.fsafe, self.fcrit = f_thresh

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = batt_consumed_per_time

        self.dec_steps = rospy.get_param("/dec_steps")
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.nareas = rospy.get_param("/nareas") #Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i+1) for i in range(self.nareas)]  # list of int area IDs
        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data

        #Initialize variables
        charging_station_coords = rospy.get_param("~initial_pose_x"), rospy.get_param("~initial_pose_y") #rospy.get_param("/charging_station_coords")
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        self.sampled_nodes_poses = [charging_pose_stamped] #list container for sampled nodes of type PoseStamped

        #Pickle load the sampled area poses
        with open('{}.pkl'.format(rospy.get_param("/file_sampled_areas")), 'rb') as f:
            sampled_areas_coords = pickle.load(f)
        for area_coords in sampled_areas_coords['n{}_p{}'.format(self.nareas, rospy.get_param("/placement"))]:
            pose_stamped = pu.convert_coords_to_PoseStamped(area_coords)
            self.sampled_nodes_poses.append(pose_stamped)

        self.x, self.y = 0.0, 0.0 #Initialize robot pose
        self.charging_station = 0
        self.curr_loc = self.charging_station #Initial location robot is the charging station
        self.battery = self.max_battery #Initialize battery at max, then gets updated by subscribed battery topic
        self.best_decision = None
        self.dist_matrix = None
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True
        self.curr_fmeasures = dict() #container of current F-measure of areas
        self.decay_rates_dict = dict() #dictionary for decay rates
        self.recorded_f_data = dict() #dictionary of recorded data collected during mission per area
        self.recorded_decay_param = dict() #dictionary of recorded decay parameter based on data collected during mission
        self.tlapses = dict() #dictionary containing tlapse for each area since last restoration
        self.state = list() #list of states

        for area in self.areas:
            self.curr_fmeasures[area] = self.max_fmeasure #initiate
            self.decay_rates_dict[area] = None
            self.recorded_f_data[area] = list()
            self.recorded_decay_param[area] = list()
            self.tlapses[area] = 0

        self.decisions_made, self.decisions_accomplished, self.status_history = [], [], [] #record of data
        self.total_dist_travelled = 0 #total distance travelled
        self.process_time_counter = [] #container for time it took to come up with decision
        self.travel_noise_data = [] #container for actual travel noise
        self.restoration_data = [] #container for actual restoration rate
        self.training_time_counter = [] #container for time it takes to train time-series model
        self.training_loss_counter = [] #container for training losses
        self.forecast_time_counter = [] #container for process time in forecasting decay data

        #Parameters under uncertainty
        self.inference = rospy.get_param("/inference")
        self.learning_rate = rospy.get_param("/learning_rate")
        self.nvisits_thresh = rospy.get_param("/nvisits_thresh")
        self.discrepancy_thresh = rospy.get_param("/discrepancy_thresh")

        # Exploration
        self.exploration = rospy.get_param("/exploration")
        self.toggle_exploration = True if self.exploration > 0.0 else False
        self.risk = 0  # Risk of decay rate growing
        self.recorded_risks = [] #Container for all recorded risks

        #We sum this up
        self.environment_status = dict()
        for node in range(self.nareas+1):
            self.environment_status[node] = 999

        # self.history_data_filename = rospy.get_param("/history_data")
        # self.history_decisions_filename = rospy.get_param("/history_decisions")
        # decisions_data, recorded_decay_data = self.read_history_data()
        # self.survey_data = self.build_decisions_recorded_data(decisions_data, recorded_decay_data)
        #Note: Restoration rate is already initialized above. This shall be learned

        #Publishers/Subscribers
        rospy.Subscriber('/robot_{}/odom'.format(self.robot_id), Odometry, self.distance_travelled_cb, queue_size=1)

        # Service request to move_base to get plan : make_Plan
        server = '/robot_' + str(self.robot_id) + '/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        # self.debug("Getplan service: {}".format(self.get_plan_service))

        rospy.Subscriber('/robot_{}/battery_status'.format(self.robot_id), Int8, self.battery_status_cb)
        rospy.Subscriber('/robot_{}/battery'.format(self.robot_id), Float32, self.battery_level_cb)

        for area in self.areas:
            rospy.Subscriber('/area_{}/decay_rate'.format(area), Float32, self.decay_rate_cb, area)
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb, area) #REMARK: Here we assume that we have live measurements of the F-measures
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb, area)

        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)

        #Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base', MoveBaseAction)
        self.robot_goal_client.wait_for_server()

    def read_history_data(self):
        """
        Reads recorded decay data and decisions for a given placement
        """
        # self.debug("Reading history decay data: {}".format(self.history_data_filename))
        with open('history_data/' + self.history_data_filename, 'rb') as f:
            data = pickle.load(f)
            f.close()

        # self.debug("Reading history decisions data: {}".format(self.history_decisions_filename))
        with open('history_data/' + self.history_decisions_filename, 'rb') as f:
            decisions_data = pickle.load(f)
            f.close()

        recorded_decay_data = copy.deepcopy(data[1])
        decisions_data = decisions_data[0]

        return decisions_data, recorded_decay_data

    def build_decisions_recorded_data(self, decisions_data, recorded_decay_data, imputation='interpolate'):
        """
        Build a dataframe of size (decisions_made, nareas)
        """
        # Create a matrix/data frame of size (decisions_made, nareas)
        recorded_data = copy.deepcopy(recorded_decay_data)
        areas = list(recorded_data.keys())
        data_matrix = np.full((len(decisions_data), len(areas)), np.nan)

        for x in range(len(decisions_data)):
            area = decisions_data[x]
            if area != 0:
                datum = recorded_data[area].pop(0)
                data_matrix[x, area - 1] = datum

        data_matrix = pd.DataFrame(data_matrix, columns=areas)

        # Impute missing data
        if imputation == 'interpolate':
            imputed_data = data_matrix.interpolate(method='linear')
            imputed_data = imputed_data.bfill()
        return imputed_data

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

        # self.debug("Dist matrix: {}".format(self.dist_matrix))

    def build_noise_matrix(self):
        """
        Builds a matrix for travel noise
        :return:
        """
        self.travel_noise = np.full(self.dist_matrix.shape, self.noise)  # Initial travel noise

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
        self.start_travel_time = self.sim_t
        # self.debug("Sending robot to mission area: {}. Departure time: {}".format(self.mission_area, self.start_travel_time))
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
            self.prev_loc = self.curr_loc
            self.curr_loc = self.mission_area
            self.end_travel_time = self.sim_t
            # self.debug(
            #     "Arrived at mision_area: {}. Arrival time: {}. Current loc: {}. Prev loc: {}".format(self.mission_area,
            #                                                                                          self.end_travel_time,
            #                                                                                          self.curr_loc,
            #                                                                                          self.prev_loc))

            self.record_decisions_accomplished()
            self.best_decision = None
            if self.mission_area == self.charging_station:
                self.update_robot_status(robotStatus.CHARGING)
            else:
                data = request_fmeasure(self.curr_loc)
                measured_f = float(data)
                self.recorded_f_data[self.curr_loc].append(measured_f)
                converted_decay = get_decay_rate(self.max_fmeasure, measured_f, self.tlapses[self.curr_loc])
                # self.debug("Collecting data...Measured f: {}. Tlapse: {}. Converted decay: {}".format(measured_f, self.tlapses[self.curr_loc], converted_decay))

                self.record_decay(converted_decay, self.curr_loc)
                self.start_restore_time = self.sim_t
                # self.debug("Restoring F-measure. Sim time: {}".format(self.start_restore_time))
                self.update_robot_status(robotStatus.RESTORING_F)

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

    def simulate_tlapses(self, tlapses, decision_area, duration):
        """
        Computes tlapses for each area that are not the decision area
        :param decision:
        :param tlapses: dictionary containing tlapse for each area
        :return: tlapses - dict containing updated tlapse for each area
        """
        updated_tlapses = tlapses.copy()
        updated_tlapses[decision_area] = 0
        for area in updated_tlapses:
            if area != decision_area:
                updated_tlapses[area] += duration
        return updated_tlapses

    def greedy_best_decision(self):
        """
        :return:
        """
        #Measure duration matrix adjusting for noise
        duration_matrix = (self.dist_matrix/self.robot_velocity) * (1 + self.travel_noise)

        # Measure the average duration an area decays
        #Estimates the time/duration it takes to areas
        mean_duration_decay_dict = dict()
        for area in self.areas:
            mean_duration_decay_dict[area] = self.mean_duration_decay(duration_matrix, area)

        #TODO: Estimate max forecast time steps here
        #We forecast decay dict beforehand
        #We then feed this to forecast_opportunity_cost
        max_forecast_timesteps = int(self.forecast_step + (np.max(duration_matrix) * self.dec_steps))
        forecast_start = process_time()
        forecast_decay_dict = forecast_decay_lstm(self.model, self.survey_data, max_forecast_timesteps)
        forecast_end = process_time()
        forecast_process_time = self.time_elapsed(forecast_start, forecast_end)
        self.forecast_time_counter.append(forecast_process_time)  # Store process time in forecasting

        debug("Max forecast tsteps: {}. Process time: {}".format(max_forecast_timesteps, forecast_process_time))
        # debug("Forecasted decay data {}:".format(forecast_decay_dict))

        #Evaluate decision

        decision_start = process_time()

        decision_array = []
        for decision in self.areas:
            # Battery consumption
            #TODO: If not oracle, we need to compute for the self.curr_fmeasures based on the believed decay rates and the tlapse for that area
            # PO: We can potentially keep updating the believed self.curr_fmeasures as we update the tlapses of areas
            # PO: Will we record the actual recorded fmeasure for that self.curr_fmeasures? Hmm. We restore it back to full measure right, i.e. the tlapse=0?
            battery_consumption, feasible_battery = self.estimate_battery_params(decision, self.battery, self.curr_loc,
                                                                                 self.curr_fmeasures, self.travel_noise[self.curr_loc, decision])
            # self.debug("Batt consumption: {}. Feasible batt: {}".format(battery_consumption, feasible_battery))


            if not prune(self.battery, battery_consumption, self.battery_reserve) and decision != self.curr_loc:
                #Immediate loss in i=1
                duration = self.compute_duration(self.curr_loc, decision, self.curr_fmeasures[decision], self.restoration, self.travel_noise[self.curr_loc, decision])
                """
                #TODO: This should be what is in the head of the robot.
                #   We can actually do an initial exploration to update the model.
                #   It is even possible to have some exploration in the middle as well.
                #   There is some problem it seems to have greedy that exploits the current learned model.
                """

                #TODO: To insert risk/exploration we may need to have a toggle
                updated_fmeasures = self.adjust_fmeasures(self.tlapses.copy(), decision, duration)  # F-measure of areas adjusted accordingly, i.e., consequence of decision
                immediate_cost_decision = self.compute_opportunity_cost(updated_fmeasures) #immediate opportunity cost
                # self.debug("Current F-measures: {}".format(self.curr_fmeasures))
                # self.debug("Feasible decision: {}. Duration: {}. Updated F: {}. Immediate loss: {}".format(decision, duration, updated_fmeasures, immediate_cost_decision))

                #Heuristic loss for i=2...k
                # self.debug("Current record tlapse: {}".format(self.tlapses))
                updated_tlapses = self.simulate_tlapses(self.tlapses.copy(), decision, duration)
                # self.debug("Simulated tlapse: {}".format(updated_tlapses))

                #TODO: Here we are forecasting future decay rates. We should use forecast_step as we will be adding it as base for future decision steps
                forecasted_cost_decision = forecast_opportunity_cost(updated_fmeasures, updated_tlapses, forecast_decay_dict, (self.fsafe, self.fcrit),
                                                         self.gamma, self.dec_steps, mean_duration_decay_dict, self.forecast_step+duration) #forecasted opportunity cost

                # self.debug("Immediate cost: {}. Discounted future losses through {} steps: {}".format(immediate_cost_decision, self.dec_steps, forecasted_cost_decision))
                evaluated_cost_decision = immediate_cost_decision + forecasted_cost_decision
                # self.debug("Appending: {}".format((decision, evaluated_cost_decision, feasible_battery)))
                decision_array.append((decision, evaluated_cost_decision, feasible_battery))

        best_decision = self.charging_station

        if len(decision_array)>0:
            best_decision = self.get_best_decision(decision_array)

        decision_end = process_time()
        decision_process_time = self.time_elapsed(decision_start, decision_end)
        self.process_time_counter.append(decision_process_time)  # Store process time in forecasting

        return best_decision

    def compute_duration(self, start_area, next_area, curr_measure, restoration, noise):
        """
        Computes (time) duration of operation, which includes travelling distance plus restoration, if any
        :param distance:
        :param restoration: restore a measure (if not None) back to full measure per second
        :param noise: expected noise in duration of distance travelled
        :return:
        """

        # Travel distance
        distance = self.dist_matrix[int(start_area), int(next_area)]
        time = (distance / self.robot_velocity) * (1 + noise)

        #If next area is not the charging station: the restoration is the f-measure; else, the restoration is the battery level
        if next_area != self.charging_station:
            max_restore = self.max_fmeasure
        else:
            max_restore = self.max_battery

        #Restoration time: If there is need for restoration
        restore_time = None
        if (curr_measure is not None) and (restoration is not None):
            restore_time = int(math.ceil((max_restore - curr_measure)/restoration))
            time += restore_time

        # self.debug("Travel time by dist: {}. Noise: {}. Restoration: {}. Total: {}".format(distance/self.robot_velocity, (distance/self.robot_velocity)*noise, restore_time, time))

        return time

    def consume_battery(self, start_area, next_area, curr_measure, noise):
        """
        Consumes curr_battery for the duration of the operation.
        This duration includes the distance plus F-measure restoration, if any
        :param curr_battery:
        :param duration:
        :return:
        """

        #Batt consumed in travel
        distance = self.dist_matrix[int(start_area), int(next_area)]
        travel_time = (distance / self.robot_velocity) * (1 + noise)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time

        if next_area != self.charging_station:
            battery_consumed += self.batt_consumed_per_restored_f * (self.max_fmeasure - curr_measure)

        return battery_consumed

    def adjust_fmeasures(self, tlapses, visit_area, duration):
        """
        #TODO: This is where we have potential overestimation.

        Adjusts the F-measures of all areas in robot's mind. The visit area will be restored to max, while the other areas will decay for
        t duration. Note that the charging station is not part of the areas to monitor. And so, if the visit_area is the
        charging station, then all of the areas will decay as duration passes by.
        :param fmeasures:
        :param visit_area:
        :param t:
        :return:
        """
        fmeasures = dict()
        # self.debug("Computation given tlapse: {}".format(tlapses))

        #TODO: We can use a scalar for exploration. If self.exploration > 0: self.toggle_exploration = True
        #   Okay what to do?
        decay_rates = self.decay_rates_dict.copy()
        if self.toggle_exploration is True:
            #TODO: Add the ratio of tlapse and time it takes to get there

            # rates = np.array(list(self.decay_rates_dict.values())) * (1 + self.risk)
            # motivation_arr = list()

            for area in self.decay_rates_dict:
                adj_decay_rate = decay_rates[area] * (1 + self.risk) #TODO: The risk here should be average.
                time_to_crit = get_time_given_decay(self.max_fmeasure, self.fcrit, adj_decay_rate)
                motivation = (tlapses[area] / time_to_crit) * self.risk
                decay_rates[area] *= (1 + self.exploration * motivation)
                # self.debug("Area: {}, Decay rate: {}, Risk={} decay rate: {}, Time to crit: {}, Motivation: {}, Adj decay rate: {}".format(area, self.decay_rates_dict[area], self.risk, adj_decay_rate, time_to_crit, motivation, decay_rates[area]))

            # decay_rates = dict(zip(self.decay_rates_dict.keys(), rates)) #TODO: Ensure that the decay rates area multiplied

            # self.debug("Motivating exploration. Risk factor: {}. Adjusted decay rates: {}".format(self.risk, decay_rates))

        for area in self.areas:
            if area == visit_area:
                fmeasures[area] = self.max_fmeasure
                # self.debug("Visit area: {}. F: {}".format(visit_area, fmeasures[area]))
            else:
                tlapse = tlapses[area] + duration
                fmeasures[area] = decay(decay_rates[area], tlapse, self.max_fmeasure) #TODO: This one here needs to be reviewed in the new. Should be forecasted
                # self.debug("Other area: {}. Tlapse: {}. New tlapse: {}. F: {}".format(area, tlapses[area], tlapse, fmeasures[area]))

        return fmeasures

    def compute_opportunity_cost(self, fmeasures):
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
        sorted_decisions = sorted(dec_arr, key = lambda x: (x[-2], -x[-1]))
        # self.debug("Decisions sorted by cost: {}".format(sorted_decisions))
        # self.debug("Best decision (branch info): {}".format(sorted_decisions[0]))
        best_decision = sorted_decisions[0][0] #pick the decision with least net loss and most available feasible battery
        return best_decision

    def update_tlapses_areas(self, sim_t):
        """
        Lapses all time elapsed for each area
        :return:
        """
        for area in self.areas:
            self.tlapses[area] += 1
        # self.debug("Sim t: {}. Time elapsed since last restored: {}".format(sim_t, self.tlapses))

    # TODO: Measure tlapses
    def estimate_curr_fmeasures(self):
        """
        Estimates current f-measures based on believed decay rates and tlapse per area
        :return:
        """
        for area in self.areas:
            self.curr_fmeasures[area] = decay(self.decay_rates_dict[area], self.tlapses[area], self.max_fmeasure)

    def reset_temp_data(self):
        """
        Resets temp data for tranining time-series model
        :return:
        """
        self.nvisits = 0
        self.temp_recorded_decay = pd.DataFrame(columns=self.areas)
        # self.debug("Temp for recorded data has been reset")

    def update_temp_data(self, new_entry, nsamples):
        """
        Updates temp data
        :param new_entry:
        :return:
        """
        #TODO: This has to be concat since we self.temp_recorded_decay will now be a DataFrame
        pd_entry = self.pad_sample_data(new_entry, nsamples=max(40, nsamples))
        # self.debug("Padded sample data: {}".format(pd_entry))
        self.temp_recorded_decay = pd.concat([self.temp_recorded_decay, pd_entry], ignore_index=True)
        # self.debug("Temp data: {}".format(self.temp_recorded_decay))
        self.nvisits += 1 #TODO: This will be nvisits

    def record_and_impute_decay(self, mission_area, decay_rate, forecast_steps):
        """
        Stores actualy decay rate in mission area while impute decay rate for all other areas using timeseries model
        :param decay_rate:
        :param forecast_step:
        :return:
        """
        entry = dict()
        forecast = forecast_decay_lstm(self.model, self.survey_data, forecast_steps)
        for j in self.areas:
            if j == mission_area:
                entry[j] = decay_rate
            else:
                entry[j] = (max(0, forecast.iloc[-1][j])) #latest forecast, area
        return entry

    def record_decay(self, decay_rate, area):
        """
        Records decay parameter for data dump
        :return:
        """
        self.recorded_decay_param[area].append(decay_rate)

    def record_decisions_accomplished(self):
        """
        Record decisions accomplished for data dump
        :return:
        """

        self.decisions_accomplished.append(self.mission_area)  # Data for saving as output

    def update_survey_data(self):
        """
        Updates survey data by appending temp data
        :return:
        """
        # temp_data = pd.DataFrame(self.temp_recorded_decay, columns=self.areas)
        self.survey_data = pd.concat([self.survey_data, self.temp_recorded_decay], ignore_index=True)

    def update_noise_model(self, noise_model, new_entry):
        """
        Updates noise model with learning rate
        :param model:
        :param new_entry:
        :return:
        """
        assert noise_model == 'travel_noise' or noise_model == 'restoration_rate', 'Invalid noise model key!'
        if noise_model == 'travel_noise':
            curr_entry = self.travel_noise[self.prev_loc, self.curr_loc]
            learned = (1-self.learning_rate)*curr_entry + self.learning_rate*new_entry
            self.travel_noise[self.prev_loc, self.curr_loc] = learned

        elif noise_model == 'restoration_rate':
            curr_entry = self.restoration
            learned = (1-self.learning_rate)*curr_entry + self.learning_rate*new_entry
            self.restoration = learned

        # self.debug("Current {}: {}. Newly measured: {}. Learned: {}".format(noise_model, curr_entry, new_entry, learned))

    def forecast_decay_rates_dict(self, forecast_steps):
        """
        Forecasts decay rates dict
        :return:
        """
        forecast = forecast_decay_lstm(self.model, self.survey_data, forecast_steps)
        return forecast.iloc[-1].to_dict()

    def discrepancy(self, actual_data, belief_data):
        """
        Measures the discrepancy rate between actual and belief data
        :param actual_data:
        :param belief_data:
        :return:
        """
        discrepancy_rate = abs((actual_data - belief_data)/belief_data)
        return discrepancy_rate

    def pad_sample_data(self, entry, nsamples):
        """
        Pads (or replicates) one row of entry by nsamples

        Inputs: entry - dictionary, one row entry
                nsamples - number of samples to pad
        Output: padded_entry - padded entry
        """
        padded_entry = pd.DataFrame([entry] * nsamples, columns=entry.keys())
        return padded_entry

    def update_risk(self, new_risk_value):
        """
        Updates the risk value by taking the average of recorded risks thus far
        :param risk_value:
        :return:
        """
        self.recorded_risks.append(new_risk_value)
        self.risk = np.mean(np.array(self.recorded_risks))

    #Methods: Run operation
    def run_operation(self, filename, freq=1):
        """
        Finite states of robot
        :return:
        """

        if self.robot_id < 999:
            rate = rospy.Rate(freq)
            # self.debug("Waiting for nodes to register...")
            rospy.sleep(15)  # Wait for nodes to register

            wait_registry = True
            while (wait_registry is True) and (len(self.sampled_nodes_poses) != self.nareas+1):
                na_count = 0
                for area in self.decay_rates_dict:
                    if self.decay_rates_dict[area] is None:
                        na_count += 1
                if na_count > 0:
                    # self.debug("Insufficient data. Decay rates: {}/{}. Sampled nodes poses: {}/{}".format(na_count, self.nareas,
                    #                                                                                       len(self.sampled_nodes_poses), self.nareas+1))
                    rate.sleep() #Data for decay rates haven't registered yet
                else:
                    wait_registry = False
            # self.debug("Sufficent data. Decay rates: {}. Sampled nodes poses: {}".format(self.decay_rates_dict, self.sampled_nodes_poses)) #Prior knowledge of decay rates

            self.survey_data = self.pad_sample_data(self.decay_rates_dict, 40)

            # self.debug("Fitting initial model on survey data...")
            train_start = process_time()
            self.model, training_loss = train_model_lstm(self.survey_data)
            #TODO: Store trained error
            self.training_loss_counter.append(training_loss)
            self.debug("Training loss: {}. Stored".format(training_loss))

            train_end = process_time()
            train_elapsed = self.time_elapsed(train_start, train_end)
            self.training_time_counter.append(train_elapsed )
            # self.debug('Done training model. Elapsed time: {}'.format(train_elapsed))
            self.forecast_step = 1
            self.decay_rates_dict = self.forecast_decay_rates_dict(self.forecast_step)

            self.reset_temp_data() #Initialize placeholders for collected data on decay and decisions made prior to training model

            self.build_dist_matrix()
            self.build_noise_matrix()
            # self.debug("Dist matrix shape: {}. Noise matrix shape: {}".format(self.dist_matrix.shape, self.travel_noise.shape))
            assert self.dist_matrix.shape == self.travel_noise.shape, 'Incongruent self.dist_matrix and self.travel_noise!'
            self.sim_t = 0 #simulation time

            while not rospy.is_shutdown() and self.sim_t < self.t_operation:

                curr_state = (self.sim_t, self.curr_loc, self.battery, self.tlapses, self.decay_rates_dict)
                self.state.append(curr_state)
                self.debug("Curr state: {}".format(curr_state))

                self.robot_status_pub.publish(self.robot_status)
                self.status_history.append(self.robot_status)
                if self.robot_status == robotStatus.IDLE.value:
                    self.debug('Robot idle')
                    if self.dist_matrix is not None:
                        self.update_robot_status(robotStatus.READY)

                elif self.robot_status == robotStatus.READY.value:
                    self.debug('Robot ready')
                    self.think_decisions()
                    self.debug('Best decision: {}. Process time: {}s'.format(self.best_decision, self.process_time_counter[-1]))
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
                    # self.debug("Mission area: {}. Current estimated decay rates: {}".format(self.mission_area, self.decay_rates_dict))

                    actual_travel_time = self.end_travel_time - self.start_travel_time
                    self.forecast_step += actual_travel_time

                    belief_decay = self.decay_rates_dict[self.curr_loc]
                    measured_decay = self.recorded_decay_param[self.curr_loc][-1]

                    #Process the newly recorded data: Store actual data for area visited while impute for all other areas by forecasting using timeseries model
                    new_entry = self.record_and_impute_decay(self.curr_loc, measured_decay, self.forecast_step) #TODO: Verify whether this is based on time step. HERE UPNEXT. SOUNDS GOOD
                    # self.debug("Imputed data entry at forecast time step, {}: {}".format(self.forecast_step, new_entry))
                    """
                    1. Init temp_data as dataframe
                    2. Update temp_data: Pad new entry and concat with temp_data. Increase nvisits += 1
                    3. If enough nvisits, concat with self.survey_data. Train LSTM model
                    """
                    self.update_temp_data(new_entry, actual_travel_time)

                    discrepancy = self.discrepancy(measured_decay, belief_decay)
                    # self.debug("Replan stats: nvisits {}, measured decay {}, belief {}, discrepancy {}".format(self.nvisits, measured_decay, belief_decay, discrepancy))

                    #TODO: Update risk to the largest growth in decay rate
                    #TODO: Actually we measure the average Type 2 error
                    if self.toggle_exploration and discrepancy > 0:
                        self.update_risk(discrepancy)
                        self.debug("Recorded risk: {} Updated risk rate: {}".format(discrepancy, self.risk))

                    #Update the model
                    if self.nvisits >= self.nvisits_thresh and discrepancy >= self.discrepancy_thresh:
                        #Update survey data
                        self.update_survey_data() #This thing concats temps data and survey data
                        # #Actually this should coincide on the last exploration
                        # ntrain = min(len(self.survey_data), math.ceil(np.mean(self.dist_matrix/self.robot_velocity))*10) #Train data on the more recent collected data
                        # train_data = self.survey_data.iloc[-ntrain:]

                        #Fit model using survey data
                        self.debug("Model update conditions met. Updating timeseries model...")
                        train_start = process_time()
                        self.model, training_loss = train_model_lstm(self.survey_data) #train_data
                        self.debug("Training loss: {}. Stored".format(training_loss))
                        # TODO: Store trained error
                        self.training_loss_counter.append(training_loss)
                        train_end = process_time()
                        train_elapsed = self.time_elapsed(train_start, train_end)
                        self.training_time_counter.append(train_elapsed)
                        # self.debug('Done training model. Elapsed time: {}'.format(train_elapsed))

                        #Reset temp data
                        self.reset_temp_data()
                        self.forecast_step = 1

                    #Forecast decay rates after one decision was made
                    self.decay_rates_dict = self.forecast_decay_rates_dict(self.forecast_step)
                    # self.debug("Updated estimated decay rates at forecast time step, {}: {}".format(self.forecast_step, self.decay_rates_dict))

                    #Learn/update noise models for travel and restoration
                    #Measure actual noise. Update current noise model
                    est_travel_time = (self.dist_matrix[self.prev_loc, self.curr_loc] / self.robot_velocity) #Note: Should just be solely on dist_matrix / robot velocity as the estimation
                    travel_noise = measure_travel_noise(actual_travel_time, est_travel_time)
                    # self.debug("Est travel time: {}. Actual: {}. Noise: {}".format(est_travel_time, actual_travel_time, travel_noise))
                    self.update_noise_model('travel_noise', travel_noise) #This will need tuple of previous and current loctions

                    #Measure actual restoration rate. Update current restoration model
                    restored_amount = self.max_fmeasure - self.recorded_f_data[self.curr_loc][-1]
                    restoration_rate = measure_restoration_rate(restored_amount, self.end_restore_time, self.start_restore_time)
                    # self.debug("Area {} Fmeasure prior restoration: {}. Restored amount: {}. Restoration time: {}".format(self.curr_loc, self.recorded_f_data[self.curr_loc][-1], restored_amount, self.end_restore_time-self.start_restore_time))
                    self.update_noise_model('restoration_rate', restoration_rate)

                    #Store actual travel noise and restoration rate for data dump
                    self.travel_noise_data.append(travel_noise)
                    self.restoration_data.append(restoration_rate)

                    self.update_robot_status(robotStatus.IN_MISSION)

                self.sim_t += 1

                #Update tlapse for each area when all nodes have registered
                if len(self.decisions_made)>1 or (self.robot_status != robotStatus.IDLE.value) and (self.robot_status != robotStatus.READY.value) and (self.robot_status != robotStatus.CONSIDER_REPLAN.value):
                    self.update_tlapses_areas(self.sim_t)
                    if self.inference != 'oracle':
                        self.compute_curr_fmeasures()

                rate.sleep()

            #Store results
            self.update_robot_status(robotStatus.SHUTDOWN)
            self.robot_status_pub.publish(self.robot_status)
            self.status_history.append(self.robot_status)

            #Wait before all other nodes have finished dumping their data
            if self.save:
                if self.inference is not None:
                    pu.dump_data((self.recorded_f_data, self.recorded_decay_param), '{}_robot{}_recorded_data'.format(filename, self.robot_id))
                pu.dump_data(self.state, '{}_environment_state'.format(filename))
                pu.dump_data(self.process_time_counter, '{}_robot{}_process_time'.format(filename, self.robot_id))
                pu.dump_data(self.training_time_counter, '{}_robot{}_training_time'.format(filename, self.robot_id))
                self.debug("Training losses: {}. Dumping data".format(self.training_loss_counter))
                pu.dump_data(self.training_loss_counter, '{}_robot{}_training_losses'.format(filename, self.robot_id))
                pu.dump_data(self.forecast_time_counter, '{}_robot{}_forecast_time'.format(filename, self.robot_id))
                pu.dump_data(self.decisions_made, '{}_robot{}_decisions'.format(filename, self.robot_id))
                pu.dump_data((self.decisions_accomplished, self.total_dist_travelled), '{}_robot{}_decisions_acc_travel'.format(filename, self.robot_id))
                pu.dump_data({'travel_noise': self.travel_noise_data, 'restoration': self.restoration_data}, '{}_robot{}_noise_data'.format(filename, self.robot_id))
                pu.dump_data(self.status_history, '{}_robot{}_status_history'.format(filename, self.robot_id))
                self.debug("Dumped all data.".format(self.robot_id))
            self.shutdown(sleep=10)

    def think_decisions(self):
        """
        Thinks of the best decision before starting mission
        :return:
        """
        self.best_decision = self.greedy_best_decision()

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
            self.decisions_made.append(self.mission_area) #store decisions made
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
        #Updates total distance travelled
        #Sets curr robot pose
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
            if self.robot_id < 999: self.debug("Fully charged!")
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
            self.tlapses[area_id] = 0 #Reset the tlapse since last restored for the newly restored area
            self.end_restore_time = self.sim_t
            if self.robot_id < 999: self.debug("Area {} fully restored! Sim time when restored: {}. Tlapse reset...".format(area_id, self.end_restore_time))

            if (self.inference is not None) and (self.inference != 'oracle'):
                self.update_robot_status(robotStatus.CONSIDER_REPLAN)
            else:
                self.update_robot_status(robotStatus.IN_MISSION)

    def decay_rate_cb(self, msg, area_id):
        """
        Store decay rate only if we have oracle knowledge
        :param msg:
        :param area_id:
        :return:
        """

        #Store the decay rates at instance, (prior knowledge as oracle)
        if self.decay_rates_dict[area_id] == None:
            # We store the prior decay rate data as first input to the recorded decay rates data
            # if self.robot_id < 999: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates_dict[area_id] = msg.data
            if len(self.recorded_decay_param[area_id]) == 0:
                self.recorded_decay_param[area_id].append(self.decay_rates_dict[area_id])

        else:
            #If we are now on mission and oracle, we immediately update the decay rates for any evolution
            if self.inference == 'oracle':
                # if self.decay_rates_dict[area_id] != msg.data: self.debug("Oracle knowledge, change in decay in area {}: {}".format(area_id, msg.data))
                self.decay_rates_dict[area_id] = msg.data #A subscribed topic. Oracle knows exactly the decay rate happening in area

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
        # self.debug("Used for computation. Tlapses: {}. Decay rates: {}".format(self.tlapses, self.decay_rates_dict))
        # self.debug("Computed current f-measures: {}".format(self.curr_fmeasures))

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)

if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    Robot('heuristic_uncertainty').run_operation(filename)
