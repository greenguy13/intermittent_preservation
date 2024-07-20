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
from infer_decay_parameters import *
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

        self.dec_steps = rospy.get_param("/dec_steps") #STAR
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
        self.inferred_decay_rates = dict() #dict for inferred decay rates
        self.state = list() #list of states

        for area in self.areas:
            self.decay_rates_dict[area] = None
            self.recorded_f_data[area] = list()
            self.recorded_decay_param[area] = list()
            self.inferred_decay_rates[area] = list()
            self.tlapses[area] = 0

        self.decay_rates_counter = 0 #counter for stored decay rates; should be equal to number of areas
        self.decisions_made, self.decisions_accomplished, self.status_history = [], [], [] #record of data
        self.total_dist_travelled = 0 #total distance travelled
        self.process_time_counter = [] #container for time it took to come up with decision
        self.travel_noise_data = [] #container for actual travel noise
        self.restoration_data = [] #container for actual restoration rate

        #Parameters under uncertainty
        self.inference = rospy.get_param("/inference")
        self.learning_rate = rospy.get_param("/learning_rate")
        self.ndata_thresh = rospy.get_param("/ndata_thresh")
        self.discrepancy_thresh = rospy.get_param("/discrepancy_thresh")

        #We sum this up
        self.environment_status = dict()
        for node in range(self.nareas+1):
            self.environment_status[node] = 999

        # TODO: Data of prior survey
        survey_filename = None #TODO: Supply filename of history data as a parameter
        decisions_data, recorded_decay_data = self.read_history_data(placement=survey_filename[0], dec_steps=survey_filename[1])
        self.survey_data = self.build_decisions_recorded_data(decisions_data, recorded_decay_data)
        self.travel_noise = np.full((self.nareas, self.nareas), self.noise) #Initial travel noise
        # Restoration rate is already defined above. This shall be learned
        self.scale, self.order, self.trend = 1e4, (1, 1), 'ct'



        #Publishers/Subscribers
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
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb, area) #REMARK: Here we assume that we have live measurements of the F-measures
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb, area)

        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)

        #Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base', MoveBaseAction)
        self.robot_goal_client.wait_for_server()

    def read_history_data(self, placement, dec_steps):
        """
        Reads recorded decay data and decisions for a given placement
        """
        filename = 'heuristic_uncertainty_expected_office_n8_p{}_non_uniform_k{}_1_robot0_recorded_data.pkl'.format(placement, dec_steps)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            f.close()

        filename = 'heuristic_uncertainty_expected_office_n8_p{}_non_uniform_k{}_1_robot0_decisions_acc_travel.pkl'.format(placement, dec_steps)
        with open(filename, 'rb') as f:
            decisions_data = pickle.load(f)
            f.close()

        recorded_decay_data = copy.deepcopy(data[1])
        decisions_data = decisions_data[0]

        return decisions_data, recorded_decay_data

    def build_decisions_recorded_data(self, decisions_data, recorded_decay_data, imputation='interpolate'):
        """
        Build a dataframe of size (decisions_made, nareas)
        """
        # Create a matrix/data frame of size (nareas, decisions_made)
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

        self.debug("Dist matrix: {}".format(self.dist_matrix))

    # METHODS: Send robot to area
    # TODO: This can be an independent script of its own
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
        self.start_travel_time = self.sim_t #TODO: Note time stamp prior to travel
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
            self.debug("Arrived at mision_area: {}. Current loc: {}. Prev loc: {}".format(self.mission_area, self.curr_loc, self.prev_loc))
            #TODO: Measure actual travel time. Shall we store this data? Potentially as a tuple to decision accomplished. DONE
            self.end_travel_time = self.sim_t
            self.record_decisions_accomplished(self.mission_area) #TODO: To-refine. DONE
            self.best_decision = None
            if self.mission_area == self.charging_station:
                self.update_robot_status(robotStatus.CHARGING)
            else:
                data = request_fmeasure(self.curr_loc)
                measured_f = float(data)
                self.recorded_f_data[self.curr_loc].append(measured_f) #TODO: Use this to measure restored amount to measure restoration rate
                converted_decay = get_decay_rate(self.max_fmeasure, measured_f, self.tlapses[self.curr_loc])
                self.debug("Inverting decay...Measured f: {}. Tlapse: {}. Converted decay: {}".format(measured_f, self.tlapses[self.curr_loc], converted_decay))

                self.record_decay(converted_decay, self.curr_loc) #TODO: To-refine. DONE
                self.start_restore_time = self.sim_t
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

    #TODO: Upnext. Identify the methods that need be adjusted for noise in travel and restoration
    #TODO: Forecasting F-measure by forecasting decay rates
    # Immediate opportunity cost
    # Forecasted opportunity cost

    def estimate_battery_params(self, decision, curr_battery, curr_loc, fmeasures, noise):
        """
        #TODO: Edit here. DONE
        #Resolution: Keep as is. Just provide self.noise_matrix[curr_loc, decision] as the noise parameter where this is being called

        Measures battery consumption and feasible battery
        :param decision:
        :return:
        """
        # Battery consumed travel and preserve area (if not charging station)
        # DONE. TODO: The noise here can be a matrix with start_area and next_area
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

    #TODO: Identify where to insert forecasting for each area for given decision steps to look ahead
    #TODO: Identify where to use the learned noise for each location pairs
    def greedy_best_decision(self):
        """
        #TODO: Edit this
        :return:
        """
        #Measure duration matrix adjusting for noise
        duration_matrix = (self.dist_matrix/self.robot_velocity) + self.noise_matrix

        # Measure the average duration an area decays
        #Estimates the time/duration it takes to areas
        mean_duration_decay_dict = dict()
        for area in self.areas:
            mean_duration_decay_dict[area] = self.mean_duration_decay(duration_matrix, area)

        #Evaluate decision
        decision_array = []
        for decision in self.areas:
            # Battery consumption
            battery_consumption, feasible_battery = self.estimate_battery_params(decision, self.battery, self.curr_loc,
                                                                                 self.curr_fmeasures, self.noise_matrix[self.curr_loc, decision]) #TODO: DONE. The noise here should be between self.curr_loc and decision
            self.debug("Batt consumption: {}. Feasible batt: {}".format(battery_consumption, feasible_battery))
            if not prune(self.battery, battery_consumption, self.battery_reserve) and decision != self.curr_loc:
                #Immediate loss in i=1
                duration = self.compute_duration(self.curr_loc, decision, self.curr_fmeasures[decision], self.restoration, self.noise[self.curr_loc, decision]) #TODO: DONE! Respective noise

                #TODO: The tlapse should use the state data on tlapse
                updated_fmeasures = self.adjust_fmeasures(self.curr_fmeasures.copy(), decision, duration)  # F-measure of areas adjusted accordingly, i.e., consequence of decision
                immediate_loss_decision = self.compute_net_loss(updated_fmeasures)
                self.debug("Current F-measures: {}".format(self.curr_fmeasures))
                self.debug("Feasible decision: {}. Duration: {}. Updated F: {}. Immediate loss: {}".format(decision, duration, updated_fmeasures, immediate_loss_decision))

                #Heuristic loss for i=2...k
                #TODO: Insert Time series forecast for future steps in heuristic_loss_decision
                #TODO: Compute the tlapse for each of the areas. For the computed duration, add that up to all other areas except the decided area. We then feed this as input into next method
                updated_tlapses = self.simulate_tlapses(self.tlapses, decision, duration)
                forecasted_loss_decision = forecast_opportunity_cost(updated_fmeasures, updated_tlapses, self.survey_data, self.model, self.scale, (self.fsafe, self.fcrit),
                                                         self.gamma, self.dec_steps, mean_duration_decay_dict) #Main

                self.debug("Discounted future losses through {} steps: {}".format(self.dec_steps, forecasted_loss_decision))
                evaluated_loss_decision = immediate_loss_decision + forecasted_loss_decision
                self.debug("Appending: {}".format((decision, evaluated_loss_decision, feasible_battery)))
                decision_array.append((decision, evaluated_loss_decision, feasible_battery))

        best_decision = self.charging_station

        if len(decision_array)>0:
            best_decision = self.get_best_decision(decision_array)

        return best_decision

    def compute_duration(self, start_area, next_area, curr_measure, restoration, noise):
        """
        #TODO: Edit this. No
        #Resoln: Just make sure the noise is from self.matrix and the correct restoration rate when this method is called, both are being learned

        Computes (time) duration of operation, which includes travelling distance plus restoration, if any
        :param distance:
        :param restoration: restore a measure (if not None) back to full measure per second
        :param noise: expected noise in distance travelled
        :return:
        """

        # Travel distance
        distance = self.dist_matrix[int(start_area), int(next_area)]
        distance += noise * distance #distance + noise #DONE. TODO: Adjust the noise here
        time = (distance / self.robot_velocity)

        #If next area is not the charging station: the restoration is the f-measure; else, the restoration is the battery level
        if next_area != self.charging_station:
            max_restore = self.max_fmeasure
        else:
            max_restore = self.max_battery

        #Restoration time: If there is need for restoration
        if (curr_measure is not None) and (restoration is not None):
            restore_time = (max_restore - curr_measure)/restoration
            time += restore_time

        return time

    def consume_battery(self, start_area, next_area, curr_measure, noise):
        """
        #DONE. TODO: Edit this? No need. But the noise needs to be re-adjusted/scaled. Here it is a percentage.
        #Resoln: Okay, we can settle with percentage. Growth rate formula

        Consumes curr_battery for the duration of the operation.
        This duration includes the distance plus F-measure restoration, if any
        :param curr_battery:
        :param duration:
        :return:
        """

        #Batt consumed in travel
        distance = self.dist_matrix[int(start_area), int(next_area)]
        distance += noise * distance
        travel_time = (distance / self.robot_velocity)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time

        if next_area != self.charging_station:
            battery_consumed += self.batt_consumed_per_restored_f * (self.max_fmeasure - curr_measure)

        return battery_consumed

    def adjust_fmeasures(self, tlapses, visit_area, duration):
        """
        #DONE. TODO: Edit this. Yes. Based on tlapse
        #Make sure that duration accounts for the travel noise and restoration, if need be

        Adjusts the F-measures of all areas in robot's mind. The visit area will be restored to max, while the other areas will decay for
        t duration. Note that the charging station is not part of the areas to monitor. And so, if the visit_area is the
        charging station, then all of the areas will decay as duration passes by.
        :param fmeasures:
        :param visit_area:
        :param t:
        :return:
        """
        fmeasures = dict()
        for area in self.areas:
            if area == visit_area:
                fmeasures[area] = self.max_fmeasure
            else:
                tlapse = tlapses[area] + duration
                fmeasures[area] = decay(self.decay_rates_dict[area], tlapse, self.max_fmeasure)

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
        sorted_decisions = sorted(dec_arr, key = lambda x: (x[-2], -x[-1]))
        self.debug("Decisions sorted by cost: {}".format(sorted_decisions))
        self.debug("Best decision (branch info): {}".format(sorted_decisions[0]))
        best_decision = sorted_decisions[0][0] #pick the decision with least net loss and most available feasible battery
        return best_decision

    def update_tlapses_areas(self, sim_t):
        """
        Lapses all time elapsed for each area
        :return:
        """
        for area in self.areas:
            self.tlapses[area] += 1
        self.debug("Sim t: {}. Time elapsed since last restored: {}".format(sim_t, self.tlapses))

    def reset_temp_data(self):
        """
        Resets temp data for tranining time-series model
        :return:
        """
        self.ndata = 0
        self.temp_recorded_decay = list()

    def update_temp_data(self, new_entry):
        """
        Updates temp data
        :param new_entry:
        :return:
        """
        self.temp_recorded_decay.append(new_entry)
        self.ndata += 1

    def record_and_impute_decay(self, mission_area, decay_rate, forecast_steps):
        """
        Stores actualy decay rate in mission area while impute decay rate for all other areas using timeseries model
        :param decay_rate:
        :param forecast_step:
        :return:
        """
        entry = list()
        forecast = forecast_decay(self.model, self.survey_data.iloc[-1], forecast_steps, self.scale)
        for i in self.areas:
            if i == mission_area:
                entry.append(decay_rate)
            else:
                entry.append(forecast[-1][i]) #latest forecast, area
        return entry

    def record_decay(self, decay_rate, area):
        """
        Records decay parameter
        :return:
        """
        self.recorded_decay_param[area].append(decay_rate)  # Estimate decay param based on measured data and time lapsed since last restoration

    def record_decisions_accomplished(self):
        """
        Record decisions accomplished
        :return:
        """

        self.decisions_accomplished.append(self.mission_area)  # Data for saving as output

    def update_survey_data(self):
        """
        Updates survey data by appending temp data
        :return:
        """
        temp_data = pd.DataFrame(self.temp_recorded_decay, columns=self.areas)
        self.survey_data = pd.concat([self.survey_data, temp_data], ignore_index=True)

    def update_noise_model(self, noise_model, new_entry):
        """
        Updates noise model with learning rate
        :param model:
        :param new_entry:
        :return:
        """
        assert noise_model == 'travel_noise' or noise_model == 'restoration_rate', 'Invalid noise model key!'
        if noise_model == 'travel_noise':
            self.travel_noise[self.prev_loc, self.curr_loc] = (1-self.learning_rate)*self.travel_noise[self.prev_loc, self.curr_loc] + self.learning_rate*new_entry
        elif noise_model == 'restoration_rate':
            self.restoration = (1-self.learning_rate)*self.restoration + self.learning_rate*new_entry

    def forecast_decay_rates_dict(self, forecast_steps):
        """
        Forecasts decay rates dict
        :return:
        """
        forecast = forecast_decay(self.model, self.survey_data.iloc[-1], forecast_steps, self.scale)
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

    #Methods: Run operation
    def run_operation(self, filename, freq=1):
        """
        #TODO: Edit this
        :return:
        """

        if self.robot_id == 0:
            rate = rospy.Rate(freq)
            if self.inference == 'oracle':
                self.decay_rates_dict = dict()  # dictionary for decay rates
                for area in self.areas:
                    self.decay_rates_dict[area] = None
                while self.decay_rates_counter != self.nareas and len(self.sampled_nodes_poses) != self.nareas+1:
                    self.debug("Insufficient data. Decay rates: {}/{}. Sampled nodes poses: {}/{}".format(len(self.decay_rates_counter), self.nareas,
                                                                                                          len(self.sampled_nodes_poses), self.nareas+1))
                    rate.sleep() #Data for decay rates haven't registered yet
                self.debug("Sufficent data. Decay rates: {}. Sampled nodes poses: {}".format(self.decay_rates_dict, self.sampled_nodes_poses)) #Prior knowledge of decay rates
            else:
                #TODO: Train forecast model here. DONE
                self.debug("Fitting initial model on survey data...")
                self.model = fit_model(self.survey_data, order=self.order, trend=self.trend, scale=self.scale)
                self.forecast_step = 1
                #TODO: Ensure this is a decay rate. Can we have a method instead that says forecast decay rates. DONE
                self.decay_rates_dict = self.forecast_decay_rates_dict(self.forecast_step) #TODO: Survey_data and recorded_decay_params should be handled together for appending. DONE
                #Initialize placeholders for collected data on decay and decisions made prior to training model
                self.reset_temp_data()

            self.build_dist_matrix()
            assert self.dist_matrix.shape == self.trave_noise.shape, 'Incongruent self.dist_matrix and self.travel_noise!'
            self.sim_t = 0 #simulation time

            while not rospy.is_shutdown() and self.sim_t < self.t_operation:
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

                #TODO: Ensure the cycle of states is correct. Note: oracle should have no replanning state
                elif self.robot_status == robotStatus.CONSIDER_REPLAN.value:
                    self.debug('Consider re-plan...')
                    self.debug("Mission area: {}. Estimated decay rates: {}".format(self.mission_area, self.decay_rates_dict))

                    belief_decay = self.decay_rates_dict[self.curr_loc]
                    measured_decay = self.recorded_decay_param[self.curr_loc][-1] #TODO: Make sure this is updated timely for each recorded F

                    #Process the newly recorded data: Store actual data for area visited while impute for all other areas by forecasting using timeseries model
                    new_entry = self.record_and_impute_decay(self.curr_loc, measured_decay, self.forecast_step) #
                    self.update_temp_data(new_entry)

                    #Update the model
                    if self.ndata >= self.ndata_thresh and self.discrepancy(belief_decay, measured_decay) >= self.discrepancy_thresh:
                        #TODO: Append recorded data to survey data
                        ## Need to append recorded data to survey data, a data frame

                        #TODO: Redo building recorded data
                        #S1: We can build a list of lists for the recorded data
                        #S2: We then have a method to update survey data. Here we transform recorded data into data frame where the columns are the areas
                        #S3: We then append recorded data to survey_data

                        self.update_survey_data()

                        #TODO: Fit model using survey data
                        self.model = fit_model(self.survey_data, order=self.order, trend=self.trend, scale=self.scale)

                        #TODO: Reset temp data
                        self.reset_temp_data()
                        self.forecast_step = 0

                    #Forecast decay rates after one decision was made
                    self.forecast_step += 1
                    self.decay_rates_dict = self.forecast_decay_rates_dict(self.forecast_step)


                    #TODO: Learn/update noise models for travel and restoration
                    #PO: We have a timer that is a function of sim_t we constantly reset
                    actual_travel_time = self.end_travel_time - self.start_travel_time
                    est_travel_time = (self.dist_matrix[self.prev_loc, self.curr_loc] / self.robot_velocity) #Note: Should just be solely on dist_matrix / robot velocity as the estimation
                    travel_noise = measure_travel_noise(actual_travel_time, est_travel_time)
                    self.update_noise_model('travel_noise', travel_noise) #This will need tuple of previous and current loctions

                    #We can then measure the actual travel time and restoration times
                    restored_amount = self.recorded_f_data[self.curr_loc][-1]
                    restoration_rate = measure_restoration_rate(restored_amount, self.end_restore_time, self.start_restore_time)
                    self.update_noise_model('restoration_rate', restoration_rate)

                    #Store actual travel noise and restoration rate for data dump
                    self.travel_noise_data.append(travel_noise)
                    self.restoration_data.append(restoration_rate)

                    self.update_robot_status(robotStatus.IN_MISSION)  # Verified

                self.sim_t += 1
                self.state.append((self.sim_t, self.curr_loc, self.battery, self.tlapses, self.decay_rates_dict)) #TODO: We can insert the state here

                if (self.robot_status != robotStatus.IDLE.value) and (self.robot_status != robotStatus.READY.value) and (self.robot_status != robotStatus.CONSIDER_REPLAN.value): #TODO: PO Insert the condition that the areas are simulating
                    self.update_tlapses_areas(self.sim_t) #Update the tlapse per area
                rate.sleep()

            #Store results
            self.update_robot_status(robotStatus.SHUTDOWN)
            self.robot_status_pub.publish(self.robot_status)
            self.status_history.append(self.robot_status)

            #Wait before all other nodes have finished dumping their data
            if self.save:
                if self.inference is not None:
                    pu.dump_data((self.recorded_f_data, self.recorded_decay_param, self.inferred_decay_rates), '{}_robot{}_recorded_data'.format(filename, self.robot_id))
                pu.dump_data(self.state, '{}_environment_state'.format(filename))
                pu.dump_data(self.process_time_counter, '{}_robot{}_process_time'.format(filename, self.robot_id))
                pu.dump_data(self.decisions_made, '{}_robot{}_decisions'.format(filename, self.robot_id))
                pu.dump_data((self.decisions_accomplished, self.total_dist_travelled), '{}_robot{}_decisions_acc_travel'.format(filename, self.robot_id))
                #TODO: Potentially, store actual travel time and restoration rates
                pu.dump_data({'travel_noise': self.travel_noise_data, 'restoration': self.restoration_data}, '{}_robot{}_noise_data'.format(filename, self.robot_id))
                pu.dump_data(self.status_history, '{}_robot{}_status_history'.format(filename, self.robot_id))
                self.debug("Dumped all data.".format(self.robot_id))
            self.shutdown(sleep=10)


    def think_decisions(self):
        """
        Thinks of the best decision before starting mission
        :return:
        """
        self.best_decision = self.greedy_best_decision() #So inside here we can

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
            self.tlapses[area_id] = 0 #Reset the tlapse since last restored for the newly restored area
            self.end_restore_time = self.sim_t
            if self.robot_id == 0: self.debug("Area {} fully restored! tlapse reset...".format(area_id))

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
        if self.inference == 'oracle':
            if self.decay_rates_dict[area_id] == None:
                # We store the prior decay rate data as first input to the recorded decay rates data
                if self.robot_id == 0: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
                self.decay_rates_dict[area_id] = msg.data
                if len(self.recorded_decay_param[area_id]) == 0:
                    self.recorded_decay_param[area_id].append(self.decay_rates_dict[area_id])
                self.decay_rates_counter += 1
            else:
                #If we are now on mission and oracle, we immediately update the decay rates for any evolution
                if self.decay_rates_dict[area_id] != msg.data: self.debug("Oracle knowledge, change in decay in area {}: {}".format(area_id, msg.data))
                self.decay_rates_dict[area_id] = msg.data #A subscribed topic. Oracle knows exactly the decay rate happening in area

    def area_fmeasure_cb(self, msg, area_id):
        """
        Updates fmeasure of area
        :param msg:
        :param area_id:
        :return:
        """
        self.curr_fmeasures[area_id] = msg.data

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)

if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    Robot('heuristic_uncertainty').run_operation(filename)