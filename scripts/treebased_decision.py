#!/usr/bin/env python3

"""
Tree-based decision making

    Given all feasible areas and not in safe zone for the next k decision steps:
    Process:
        1. Stack all the combination of length k
        2. Compute cost
        3. Pick the least cost
"""
import rospy
from time import process_time
import pickle
import numpy as np
import actionlib
from loss_fcns import *
from pruning import *
import project_utils as pu
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from std_msgs.msg import Int8, Float32
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from status import areaStatus, battStatus, robotStatus
from reset_simulation import *

"""
Tasks
    T1: Fine tune move_base params to effectively maneuver obstacles
    T2: Modularize some aspects of the scripts, particularly the sampling of nodes from decision methods
    T3: Save data by rosbag: Identify info/data for summary
    T4: Replicate the experiments (multiple experiments with diff seeds)
    T5: Visualize real-time changes in F-measure in areas
"""

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
        self.fsafe, self.fcrit = rospy.get_param("/f_thresh") #(safe, crit)
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = rospy.get_param("/batt_consumed_per_time") #(travel, restoration)
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
        self.optimal_path = []
        self.dist_matrix = None
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True
        self.curr_fmeasures = dict() #container of current F-measure of areas
        self.decay_rates_dict = dict() #dictionary for decay rates
        for area in self.areas:
            self.decay_rates_dict[str(area)] = None
        self.decay_rates_counter = 0 #counter for stored decay rates; should be equal to number of areas
        self.decisions_made, self.decisions_accomplished, self.status_history = [], [], [] #record of data
        self.total_dist_travelled = 0 #total distance travelled
        self.process_time_counter = [] #container for time it took to come up with decision

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

        for area in self.areas:
            rospy.Subscriber('/area_{}/decay_rate'.format(area), Float32, self.decay_rate_cb, area)
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb, area) #REMARK: Here we assume that we have live measurements of the F-measures
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb, area)

        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)

        #Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base', MoveBaseAction)
        self.robot_goal_client.wait_for_server()


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
            self.curr_loc = self.mission_area
            self.update_robot_status(robotStatus.RESTORING_F)
            if self.mission_area == self.charging_station:
                self.update_robot_status(robotStatus.CHARGING)
            self.decisions_accomplished.append(self.mission_area)

    #DECISION-MAKING methods
    def grow_tree(self, dec_steps, restoration, noise):
        """
        We grow a decision tree of depth dec_steps starting from where the robot is.
        Note: Each branch is a tuple containing the path, battery, fmeasures, cost, feasible_battery info
        :param curr_location: current location of robot
        :param dec_steps: number of decision steps, (i.e., tree depth)
        :param restoration: duration it takes to restore per unit F-measure
        :param noise: anticipated noise in actual travel to an area
        :return:
        """
        """
        Comments:
        1. How do we get updated F-measure of areas?
            > Scenario 1: We are able to monitor F in real-time
                Subscribe to topic 
            > Scenario 2: We only get to know the F-measure once we are there.
                We have our belief/computation/model of what F would be. We start with 100 and decay based on that model.
                As we go through our mission and visit areas, we constantly measure F. We then update our model (or parameters) of F for that area.
        """

        branches = list() #container for final branches up to depth k
        to_grow = list()  #container for branches still being grown/expanded
        nodes = [self.charging_station]
        nodes.extend(self.areas)
        self.debug("Nodes: {}".format(nodes))

        #Start at the current location as the root node.
        #Scenario 1
        fmeasures = self.curr_fmeasures.copy()
        k = 0
        path = [self.curr_loc]
        battery = self.battery
        cost = 0 #Initialize cost of path
        if self.robot_id==0: self.debug("Areas: {}. Fmeasures: {}. Battery: {}".format(self.areas, fmeasures, battery))

        #Initial feasible battery level
        feasible_battery_consumption = self.consume_battery(start_area=self.curr_loc, next_area=self.charging_station, curr_measure=None, noise=noise)
        feasible_battery = battery - feasible_battery_consumption

        branch = (path, battery, fmeasures, cost, feasible_battery)
        to_grow.append(branch)

        #Succeeding decision steps:
        while k < dec_steps:
            self.debug("\nDec step: {}".format(k))
            consider_branches = to_grow.copy()
            to_grow = list() #At the end of the iterations, to-grow will be empty while branches must be complete
            for branch in consider_branches:
                self.debug("Branch to grow: {}".format(branch))
                considered_growing = 0 #Indicators whether the branch has been considered for growing
                for i in range(len(nodes)):
                    # Hypothetical: What if we travel to this node, what will the consumed battery be and the updated F-fmeasures?
                    # Shall we prune this in the next decision step?
                    path = branch[0].copy()  # path at depth k
                    battery = branch[1]  # battery level at depth k
                    fmeasures = branch[2].copy()  # current fmeasures at depth k
                    cost = branch[3] # cost of path at depth k
                    start_area = path[-1]
                    next_area = nodes[i]  # We are considering travelling to all nodes

                    if next_area != self.charging_station:
                        curr_measure = fmeasures[next_area]
                        tlapse_decay = get_time_given_decay(self.max_fmeasure, curr_measure, self.decay_rates_dict[str(next_area)])
                        duration = self.compute_duration(start_area=start_area, next_area=next_area, curr_measure=curr_measure, restoration=restoration, noise=noise) #Duration if we preserve area: travel plus restoration
                        decayed_fmeasure = decay(self.decay_rates_dict[str(next_area)], tlapse_decay+duration, self.max_fmeasure)  # Decayed measure of area if we travel there
                    else:
                        curr_measure, tlapse_decay, decayed_fmeasure = None, None, None
                        duration = self.compute_duration(start_area=start_area, next_area=self.charging_station, curr_measure=battery, restoration=restoration, noise=noise) #Duration if we charge up

                    #Battery consumption
                    battery_consumption = self.consume_battery(start_area=start_area, next_area=next_area, curr_measure=curr_measure, noise=noise) #Battery consumed travel and preserve area (if not charging station)
                    battery_consumption_backto_charging_station = self.consume_battery(start_area=next_area, next_area=self.charging_station, curr_measure=None, noise=noise) #Battery consumed travel back to charging station
                    feasible_battery_consumption = battery_consumption + battery_consumption_backto_charging_station

                    self.debug("Next area: {}, Batt level: {}, TLapsed decay: {}, Duration: {}, Decayed fmeasure: {}, Batt consumption: {}".format(next_area, battery, tlapse_decay, duration, decayed_fmeasure, battery_consumption))

                    # If branch is not to be pruned and length still less than dec_steps, then we continue to grow that branch
                    cond1 = prune(battery, feasible_battery_consumption, self.battery_reserve)
                    self.debug("Prune: {}".format(cond1))
                    if cond1 is False and start_area != next_area:
                        #PO condition: If next node is charging station + one of the areas is decaying if we travel to that area + battery is safe or 100 (given when in charging station?)
                        path.append(next_area) #append next area as part of the path at depth k+1. #This is where the additional or overwriting happens. We need to make dummy list/container
                        if next_area != self.charging_station:
                            battery -= battery_consumption #actual battery depleted at depth k+1
                        else:
                            battery = self.max_battery #actual battery restored to max value
                        feasible_battery = battery - feasible_battery_consumption  # battery available after taking into account battery to go back to charging station from current location. Note: if location is charging station, feasible_battery = max_battery
                        updated_fmeasures = self.adjust_fmeasures(fmeasures, next_area, duration) #F-measure of areas adjusted accordingly, i.e., consequence of decision
                        cost += (self.gamma**k)*self.compute_cost(updated_fmeasures) #Discounted cost of this decision
                        self.debug("Resultant F-measures: {}".format(updated_fmeasures))
                        self.debug("Branch to grow appended (path, batt, upd_fmeasures, cost, feas_batt): {}, {}, {}, {}, {}".format(path, battery, updated_fmeasures, cost, feasible_battery))
                        to_grow.append((path, battery, updated_fmeasures, cost, feasible_battery)) #Branch: (path, battery, updated_fmeasures, cost, feasible battery)
                        considered_growing += 1

                    #Else, we add that branch to branches (for return), which includes pruned branches. Conditions:
                    # 1.) Robot is not dead at the end of the operation, i.e., we check whether remaining feasible battery >= 0. If not, then this path ends dead, thus we don't append it.
                    # 2.) Furthermore: If even after iterating through all possible nodes, (thats why i == len(nodes)-1), branch not considered for growing.
                    # 3.) And branch not yet in branches.
                    else:
                        if (is_feasible(battery, feasible_battery_consumption, self.battery_reserve) is True) and (i == len(nodes)-1 and considered_growing == 0) and (branch not in branches):
                            self.debug("Branch appended to tree: {}".format(branch))
                            branches.append(branch)
            k += 1 #We are done with k depth, so move on to the next depth

        #We append to branches the branches of length k, (i.e., the final decision step)
        for branch in to_grow:
            if branch not in branches:
                branches.append(branch)
        self.debug("Arrived at last dec. step. Number of branches: {}".format(len(branches)))
        return branches

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
                tlapse_decay = get_time_given_decay(self.max_fmeasure, fmeasures[area], self.decay_rates_dict[str(area)]) + duration
                fmeasures[area] = decay(self.decay_rates_dict[str(area)], tlapse_decay, self.max_fmeasure)

        return fmeasures

    def compute_cost(self, fmeasures):
        """
        Computes the cost, (i.e., the sum of losses) of the fmeasures, which is a consequence of a decision
        Steps:
            1. Computes the loss for each of the F-measure of the areas
            2. Sums up the losses to get the cost of the decision
        :param fmeasures:
        :return:
        """
        cost = compute_cost_fmeasures(fmeasures, self.fsafe, self.fcrit)

        return cost

    def get_optimal_branch(self, tree):
        """
        Returns the optimal branch of the tree. This shall be the optimal decision path for the robot
        Steps:
            1. Sorts the branches of the tree by the accumulated cost, breaking ties by available feasible battery
            2. Returns the optimal path

        :param tree:
        :return:
        """
        # Sort the branches: the cost is key while the value is branch
        sorted_branches = sorted(tree, key = lambda x: (x[-2], -x[-1])) #sorted by cost, x[-2] increasing; then sorted by feasible battery x[-1] decreasing
        #Debug
        self.debug("Branches sorted by cost:")
        for branch in sorted_branches:
            self.debug("Branch: {}".format(branch))
        self.debug("Optimal branch (branch info + cost): {}".format(sorted_branches[0]))
        optimal_path = sorted_branches[0][0] #pick the branch with least cost and most available feasible battery
        optimal_path.pop(0) #we pop out the first element of the path, which is the current location, which is not needed in current mission
        return optimal_path

    #Methods: Run operation
    def run_operation(self, filename, freq=1):
        """
        :return:
        """

        if self.robot_id == 0:
            rate = rospy.Rate(freq)
            while self.decay_rates_counter != self.nareas and len(self.sampled_nodes_poses) != self.nareas+1:
                self.debug("Insufficient data. Decay rates: {}/{}. Sampled nodes poses: {}/{}".format(len(self.decay_rates_counter), self.nareas,
                                                                                                      len(self.sampled_nodes_poses), self.nareas+1))
                rate.sleep() #Data for decay rates haven't registered yet

            self.debug("Sufficent data. Decay rates: {}. Sampled nodes poses: {}".format(self.decay_rates_dict, self.sampled_nodes_poses))
            self.build_dist_matrix()
            t = 0
            while not rospy.is_shutdown() and t<self.t_operation:
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
                    self.debug('Path: {}. Process time: {}s'.format(self.optimal_path, think_elapsed))
                    self.update_robot_status(robotStatus.IN_MISSION)

                elif self.robot_status == robotStatus.IN_MISSION.value:
                    self.debug('Robot in mission. Total distance travelled: {}'.format(self.total_dist_travelled))
                    if self.available:
                        self.commence_mission()

                elif self.robot_status == robotStatus.CHARGING.value:
                    self.debug('Waiting for battery to charge up')

                elif self.robot_status == robotStatus.RESTORING_F.value:
                    self.debug('Restoring F-measure')

                t += 1
                rate.sleep()

            #Store results
            self.update_robot_status(robotStatus.SHUTDOWN)
            self.robot_status_pub.publish(self.robot_status)
            self.status_history.append(self.robot_status)

            #Wait before all other nodes have finished dumping their data
            if self.save:
                pu.dump_data(self.process_time_counter, '{}_robot{}_process_time'.format(filename, self.robot_id))
                pu.dump_data(self.decisions_made, '{}_robot{}_decisions'.format(filename, self.robot_id))
                pu.dump_data((self.decisions_accomplished, self.total_dist_travelled), '{}_robot{}_decisions_acc_travel'.format(filename, self.robot_id))
                pu.dump_data(self.status_history, '{}_robot{}_status_history'.format(filename, self.robot_id))
                self.debug("Dumped all data.".format(self.robot_id))
            self.shutdown(sleep=10)

    def think_decisions(self):
        """
        Thinks of the optimal path before starting mission
        :return:
        """
        self.optimal_path = [self.charging_station]  # Default decision is to go back to the charging station
        tree = self.grow_tree(self.dec_steps, self.restoration, self.noise)
        if tree:
            self.optimal_path = self.get_optimal_branch(tree)  # Indices of areas/nodes

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
        if len(self.optimal_path):
            self.mission_area = self.optimal_path.pop(0)
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
            if self.robot_id == 0: self.debug("Area fully restored!")
            self.available = True
            self.update_robot_status(robotStatus.IN_MISSION)

    def decay_rate_cb(self, msg, area_id):
        """
        Store decay rate
        :param msg:
        :param area_id:
        :return:
        """
        if self.decay_rates_dict[str(area_id)] == None:
            if self.robot_id == 0: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates_dict[str(area_id)] = msg.data
            self.decay_rates_counter += 1

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
    os.chdir('/home/ameldocena/.ros/int_preservation/results')
    filename = rospy.get_param('/file_data_dump')
    Robot('treebased_decision').run_operation(filename)