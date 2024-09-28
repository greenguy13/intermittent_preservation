#!/usr/bin/env python

"""
Heuristic decision making

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
        self.fsafe, self.fcrit = f_thresh  # (safe, crit)

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = batt_consumed_per_time  # (travel, restoration)

        self.dec_steps = rospy.get_param("/dec_steps") #STAR
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")

        #TODO: This prolly needs to be instantiated within run_operation
        self.nareas = rospy.get_param("/nareas") #Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i+1) for i in range(self.nareas)]  # list of int area IDs


        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data
        self.task_scheduler = rospy.get_param("/task_scheduler") #task scheduler

        #Initialize variables
        charging_station_coords = rospy.get_param("~initial_pose_x"), rospy.get_param("~initial_pose_y") #rospy.get_param("/charging_station_coords")
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        self.sampled_nodes_poses = [charging_pose_stamped] #list container for sampled nodes of type PoseStamped, where 0 is the charging station for that robot

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
        self.dist_matrix = None #TODO: For instantiation
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True

        # TODO: For instantiation
        self.curr_fmeasures = dict() #container of current F-measure of areas
        self.decay_rates_dict = dict() #dictionary for decay rates
        self.tlapses = dict() #dictionary containing tlapses of areas

        self.state = list() #list of states
        self.strict_bounds_list = list() #list of strict lower and upper bounds

        #TODO: We are setting the (initial) prior information of the decay rates. Can we instantiate this in run_operation instead?
        for area in self.areas:
            self.decay_rates_dict[area] = None
            self.tlapses[area] = 0

        self.decay_rates_counter = 0 #counter for stored decay rates; should be equal to number of areas #TODO: For instantiation

        self.decisions_made, self.decisions_accomplished, self.status_history = [], [], [] #record of data
        self.total_dist_travelled = 0 #total distance travelled
        self.process_time_counter = [] #container for time it took to come up with decision
        self.assigned_areas = None #TODO: This is a novel variable; can be used as reference for the areas

        #We sum this up
        self.environment_status = dict()
        for node in range(self.nareas+1):
            self.environment_status[node] = 999

        #Publishers/Subscribers
        rospy.Subscriber('/robot_{}/odom'.format(self.robot_id), Odometry, self.distance_travelled_cb, queue_size=1)

        # Service request to move_base to get plan : make_Plan
        server = '/robot_' + str(self.robot_id) + '/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        rospy.Subscriber('/robot_{}/battery_status'.format(self.robot_id), Int8, self.battery_status_cb)
        rospy.Subscriber('/robot_{}/battery'.format(self.robot_id), Float32, self.battery_level_cb)

        #TODO: We can instantiate in run_operation
        for area in self.areas:
            rospy.Subscriber('/area_{}/decay_rate'.format(area), Float32, self.decay_rate_cb, area)
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb, area) #REMARK: Here we assume that we have live measurements of the F-measures
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb, area)

        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)

        #Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base', MoveBaseAction)
        self.robot_goal_client.wait_for_server()

        #Server for assigned cluster to monitor/preserve
        self.cluster_assignment_server = rospy.Service("/cluster_assignment_server_" + str(self.robot_id), clusterAssignment, self.cluster_assignment_cb)

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

    def cluster_assignment_cb(self, msg):
        """
        Sets the cluster assignment as the areas for restoration/preservation
        :param msg:
        :return:
        """
        self.assigned_areas = msg.cluster
        self.assigned_areas = list(self.assigned_areas)
        self.debug("Cluster assignment: {}, {}".format(type(self.assigned_areas), self.assigned_areas))
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

    # def build_dist_matrix(self):
    #     """
    #     Builds the distance matrix among areas
    #     :return:
    #     """
    #     n = len(self.sampled_nodes_poses) #TODO: We may have to change this to whatever the assigned areas are. PO: Feed as a parameter then build dist_matrix from there
    #     self.dist_matrix = np.zeros((n, n))
    #
    #     for i in range(n):
    #         for j in range(n):
    #             area_i, area_j = self.sampled_nodes_poses[i], self.sampled_nodes_poses[j]
    #             if area_i != area_j:
    #                 dist = self.compute_dist_bet_areas(area_i, area_j, self.tolerance)
    #                 self.dist_matrix[i, j] = dist
    #
    #     self.debug("Dist matrix: {}".format(self.dist_matrix))

    def build_dist_matrix(self, sampled_nodes_poses):
        """
        Builds the distance matrix among areas
        :return:
        """
        #TODO: PO use proxy area id
        n = len(sampled_nodes_poses) #TODO: We may have to change this to whatever the assigned areas are. PO: Feed as a parameter then build dist_matrix from there
        self.dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                area_i, area_j = sampled_nodes_poses[i], sampled_nodes_poses[j]
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
        #TODO: PO use proxy area id

        goal = self.sampled_nodes_poses[goal_idx]
        self.send_robot_goal(goal)

    def send_robot_goal(self, goal):
        """
        Sends robot to goal via action client
        :param robot:
        :param goal: PoseStamped object
        :return:
        """
        #TODO: PO use proxy area id

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
        #TODO: PO use proxy area id

        if state == SUCCEEDED:
            self.curr_loc = self.mission_area
            self.update_robot_status(robotStatus.RESTORING_F)

            if self.mission_area == self.charging_station:
                self.update_robot_status(robotStatus.CHARGING)
            self.decisions_accomplished.append(self.mission_area)
            self.best_decision = None

    def mean_duration_decay(self, duration_matrix, area):
        """
        TODO: Potentially, remove first those areas not among the assigned

        Measures the average duration that an area decays (when robot commits to another decision than restoring area)
        Note that each column of the duration matrix represents that the corresponding area is the decision that the robot is committing to.
        We thus delete the column of the area, and then take the average to measure the average duration.
        :param duration_matrix:
        :param area:
        :return:
        """
        # TODO: This is the tricky part here. How to make it slick?

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

    def greedy_best_decision(self):
        """

        :return:
        """
        #Measure duration matrix
        duration_matrix = self.dist_matrix/self.robot_velocity

        #Measure the average duration an area decays
        mean_duration_decay_dict = dict()
        for area in self.areas:
            mean_duration_decay_dict[area] = self.mean_duration_decay(duration_matrix, area)

        #Evaluate decision
        decision_array = []
        for decision in self.assigned_areas:
            # Battery consumption
            battery_consumption, feasible_battery = self.estimate_battery_params(decision, self.battery, self.curr_loc,
                                                                                 self.curr_fmeasures, self.noise)
            # self.debug("Batt consumption: {}. Feasible batt: {}".format(battery_consumption, feasible_battery))
            if not prune(self.battery, battery_consumption, self.battery_reserve) and decision != self.curr_loc:
                #Immediate loss in i=1
                duration = self.compute_duration(self.curr_loc, decision, self.curr_fmeasures[decision], self.restoration, self.noise)
                updated_fmeasures = self.adjust_fmeasures(self.curr_fmeasures.copy(), decision, duration)  # F-measure of areas adjusted accordingly, i.e., consequence of decision
                immediate_cost_decision = self.compute_opportunity_cost(updated_fmeasures) #immediate opportunity cost
                # self.debug("Current F-measures: {}".format(self.curr_fmeasures))
                # self.debug("Feasible decision: {}. Duration: {}. Updated F: {}. Immediate loss: {}".format(decision, duration, updated_fmeasures, immediate_cost_decision))

                #Heuristic loss for i=2...k
                forecasted_cost_decision = heuristic_cost_decision(updated_fmeasures, self.decay_rates_dict, (self.fsafe, self.fcrit),
                                                         self.gamma, self.dec_steps, mean_duration_decay_dict) #forecasted opportunity cost

                # self.debug("Discounted future losses through {} steps: {}".format(self.dec_steps, forecasted_cost_decision))
                evaluated_cost_decision = immediate_cost_decision + forecasted_cost_decision #total forecasted opportunity cost
                # self.debug("Appending: {}".format((decision, evaluated_cost_decision, feasible_battery)))
                decision_array.append((decision, evaluated_cost_decision, feasible_battery))

        best_decision = self.charging_station

        if len(decision_array)>0:
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
        distance += noise * distance #distance + noise
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
        Estimates battery consumption for the duration of the visit next_area from start_area.
        This duration includes the distance plus F-measure restoration, if any
        """
        #TODO: PO use proxy area id


        #Batt consumed in travel
        distance = self.dist_matrix[int(start_area), int(next_area)]
        distance += noise * distance
        travel_time = (distance / self.robot_velocity)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time

        #Batt consumed in area restoration
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
        #TODO: PO use proxy area id
        # Q: Should this be the entire areas or just the subset of areas?

        for area in self.areas:
            if area == visit_area:
                fmeasures[area] = self.max_fmeasure
            else:
                tlapse_decay = get_time_given_decay(self.max_fmeasure, fmeasures[area], self.decay_rates_dict[area]) + duration
                fmeasures[area] = decay(self.decay_rates_dict[area], tlapse_decay, self.max_fmeasure)

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
        #TODO: Q, now do you think this should be the entire f-measures or only the subset/assigned?
        netloss = compute_cost_fmeasures(fmeasures, self.fsafe, self.fcrit)
        return netloss

    def get_best_decision(self, dec_arr):
        """
        Returns the best decision in an array by sorting forecasted_loss ascendingly first then by remaining feasible battery.
        :param tree:
        :return:
        """
        #TODO: PO use proxy area id

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
        #TODO: PO use proxy area id
        # All areas or just assigned?
        for area in self.areas:
            self.tlapses[area] += 1
        self.debug("Sim t: {}. Time elapsed since last restored: {}".format(sim_t, self.tlapses))

    """
    TODO: Estimate strict and lower upper bounds
    PO: We can do this separately or within the heuristic decision/function method
    What are the ingredients?
        > self.tlapses
        > self.gamma
        > self.decay_rates_dict
        > duration_matrix = self.dist_matrix/self.robot_velocity
        > self.dec_steps
        > self.nareas
    
    Strict lower bound
        duration_matrix = self.dist_matrix/self.robot_velocity
        min_decay_rate = min(self.decay_rates_dict.values())
        min_tlapse = min(self.tlapses.values())
        min_duration = min(duration_matrix)
        
        max_decay_rate = max(self.decay_rates_dict.values())
        max_tlapse = max(self.tlapses.values())
        max_duration = max(duration_matrix)
        
        strict_lower_bound = 0
        tlapse = min_tlapse
        for i in range(1, self.dec_steps+1):
            tlapse += min_duration
            fmeasure = decay(min_decay_rate, tlapse, self.max_fmeasure)
            strict_lower_bound += self.gamma^{i-1}*loss_fcn(fmeasure, self.max_fmeasure)
        strict_lower_bound *= (self.nareas - 1)
    
    Strict upper bound
        strict_upper_bound = 0
        tlapse = max_tlapse
        for i in range(1, self.dec_steps+1):
            tlapse += i * max_duration
            fmeasure = decay(max_decay_rate, tlapse, self.max_fmeasure)
            strict_upper_bound += self.gamma^{i-1}*loss_fcn(fmeasure, self.max_fmeasure)
        strict_upper_bound *= (self.nareas - 1)
    """

    def strict_bounds(self):
        """
        Estimates the strict and upper bounds of the opportunity cost of the schedule
        :return: strict upper and lower bounds
        """
        #TODO: PO use proxy area id
        # Q: All areas or just subset?

        # Strict lower bound
        duration_matrix = self.dist_matrix / self.robot_velocity
        min_decay_rate = min(self.decay_rates_dict.values())
        min_tlapse = min(self.tlapses.values())
        min_duration = np.min(duration_matrix)

        max_decay_rate = max(self.decay_rates_dict.values())
        max_tlapse = max(self.tlapses.values())
        max_duration = np.max(duration_matrix)

        strict_lower_bound = 0
        tlapse = min_tlapse
        for i in range(1, self.dec_steps + 1):
            tlapse += min_duration
            fmeasure = decay(min_decay_rate, tlapse, self.max_fmeasure)
            discounted_loss = (self.gamma**(i - 1)) * loss_fcn(self.max_fmeasure, fmeasure)
            strict_lower_bound += discounted_loss
        strict_lower_bound *= (self.nareas - 1)

        # Strict upper bound
        strict_upper_bound = 0
        tlapse = max_tlapse
        for i in range(1, self.dec_steps + 1):
            tlapse += i * max_duration
            fmeasure = decay(max_decay_rate, tlapse, self.max_fmeasure)
            discounted_loss = (self.gamma**(i - 1)) * loss_fcn(self.max_fmeasure, fmeasure)
            strict_upper_bound += discounted_loss
        strict_upper_bound *= (self.nareas - 1)

        #Measure average time steps for plotting
        mean_duration = np.mean(duration_matrix)
        mean_tsteps = math.ceil(self.dec_steps * mean_duration)

        return strict_lower_bound, strict_upper_bound, mean_tsteps

    def extract_sampled_node_poses(self, cluster):
        """
        Extracts the node poses of the assigned clusters into a list where index 0 is the charging station pose
        :param cluster:
        :return:
        """
        poses = list()
        poses.append(self.sampled_nodes_poses[0])
        for area in cluster:
            poses.append(self.sampled_nodes_poses[area])
        return poses

    #Methods: Run operation
    def run_operation(self, filename, freq=1):
        """
        :return:
        """

        if self.robot_id < 999: #Not a dummy robot
            rate = rospy.Rate(freq)
            rospy.sleep(15)  # Wait for nodes to register

            # TODO: This may be re-organized as part of the initialization based on what areas have been assigned to a robot
            wait_registry = True
            while (wait_registry is True) and (len(self.sampled_nodes_poses) != self.nareas + 1):
                na_count = 0
                for area in self.decay_rates_dict:
                    if self.decay_rates_dict[area] is None:
                        na_count += 1
                if na_count > 0:
                    # self.debug("Insufficient data. Decay rates: {}/{}. Sampled nodes poses: {}/{}".format(na_count, self.nareas,
                    #                                                                                       len(self.sampled_nodes_poses), self.nareas+1))
                    rate.sleep()  # Data for decay rates haven't registered yet
                else:
                    wait_registry = False
            self.debug("Sufficent data. Decay rates: {}. Sampled nodes poses: {}".format(self.decay_rates_dict, self.sampled_nodes_poses))


            sampled_nodes_poses = self.sampled_nodes_poses
            #TODO: Here we change up
            if self.task_scheduler is not None:
                while self.assigned_areas is None:
                    rospy.sleep(1)
                #TODO: We extract the sampled node poses from the clustered areas, starting with the charging station
                # PO: The sampled_node_poses can have the indices (0, nsample), and then assigned_areas as reference to these areas
                # sampled_nodes_poses = self.extract_sampled_node_poses(self.assigned_areas)

            #TODO: Here we do the initialization of all containers related to areas

            self.build_dist_matrix(sampled_nodes_poses)
            self.sim_t = 0
            while not rospy.is_shutdown() and self.sim_t<self.t_operation:
                #TODO: The self.decay_rates_dict, this has to be indexed using the actual id? Yea that makes better sense
                # But if so, there will be complication when computing for the decisions/losses/consumption
                # POSSIBLY, we can just insert as an announcement of the area equivalence/dictionary
                # And then collapse the area ids
                curr_state = (self.sim_t, self.curr_loc, self.battery, self.tlapses, self.decay_rates_dict)
                self.state.append(curr_state)
                self.debug("Curr state: {}".format(curr_state))
                self.robot_status_pub.publish(self.robot_status)
                self.status_history.append(self.robot_status)

                #I am learning something new! Each day
                # Freedom comes with courage. And with courage comes freedom.
                # New wave. I create a new culture. A new trend people will appreciate and follow

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

                    bounds = self.strict_bounds()
                    self.debug('Strict bounds:{}'.format(bounds))
                    self.strict_bounds_list.append(bounds)

                    self.process_time_counter.append(think_elapsed)

                    # TODO: PO use proxy area id for the self.best_decision

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

                self.sim_t += 1

                # Update tlapse for each area when all nodes have registered
                if len(self.decisions_made) > 1 or (self.robot_status != robotStatus.IDLE.value) and (self.robot_status != robotStatus.READY.value):
                    self.update_tlapses_areas(self.sim_t)
                rate.sleep()

            #Store results
            self.update_robot_status(robotStatus.SHUTDOWN)
            self.robot_status_pub.publish(self.robot_status)
            self.status_history.append(self.robot_status)

            #Wait before all other nodes have finished dumping their data
            if self.save:
                pu.dump_data(self.strict_bounds_list, '{}_strict_bounds'.format(filename))
                pu.dump_data(self.state, '{}_environment_state'.format(filename))
                pu.dump_data(self.process_time_counter, '{}_robot{}_process_time'.format(filename, self.robot_id))
                pu.dump_data(self.decisions_made, '{}_robot{}_decisions'.format(filename, self.robot_id))
                pu.dump_data((self.decisions_accomplished, self.total_dist_travelled), '{}_robot{}_decisions_acc_travel'.format(filename, self.robot_id))
                pu.dump_data(self.status_history, '{}_robot{}_status_history'.format(filename, self.robot_id))
                self.debug("Dumped all data.".format(self.robot_id))
            self.shutdown(sleep=10)

    def think_decisions(self):
        """
        Thinks of the best decision before starting mission
        :return:
        """
        #TODO: PO use proxy area id

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
        Sends the robot to the next area in the optimal path:
        :return:
        """
        #TODO: PO use proxy area id

        if self.best_decision is not None:
            self.mission_area = self.best_decision
            self.mission_area_pub.publish(self.mission_area)
            # TODO: PO use proxy area id
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
        # TODO: Sanity check whether this is indeed changing correctly in mission

        self.environment_status[self.charging_station] = msg.data
        if msg.data == battStatus.FULLY_CHARGED.value:
            if self.robot_id < 999: self.debug("Fully charged!")
            self.available = True
            self.update_robot_status(robotStatus.IN_MISSION)

    #TODO: We initialize all the relevant subscribed topics by the assigned areas

    def area_status_cb(self, msg, area_id):
        """

        :param msg:
        :return:
        """
        #TODO: PO use proxy area id
        # Q: All areas or just assigned?

        self.environment_status[area_id] = msg.data
        if msg.data == areaStatus.RESTORED_F.value:
            if self.robot_id < 999: self.debug("Area fully restored!")
            self.tlapses[area_id] = 0  # Reset the tlapse since last restored for the newly restored area
            self.available = True
            self.update_robot_status(robotStatus.IN_MISSION)

    def decay_rate_cb(self, msg, area_id):
        """
        Store decay rate
        :param msg:
        :param area_id:
        :return:
        """
        #TODO: PO use proxy area id
        # Q: All areas or just assigned?

        if self.decay_rates_dict[area_id] == None:
            if self.robot_id < 999: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates_dict[area_id] = msg.data
            self.decay_rates_counter += 1

    def area_fmeasure_cb(self, msg, area_id):
        """
        Updates fmeasure of area
        :param msg:
        :param area_id:
        :return:
        """
        #TODO: PO use proxy area id
        # Q: All areas or just assigned?

        self.curr_fmeasures[area_id] = msg.data

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)

if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation/results')
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    Robot('heuristic_decision').run_operation(filename)