#!/usr/bin/env python

"""
Orienteering problem with time-varying rewards via RMA* search

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
from graph_node import *
import heapq
from loss_fcns import loss_fcn, decay


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
        self.fsafe, self.fcrit = f_thresh #(safe, crit)

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = batt_consumed_per_time #(travel, restoration)

        self.inference = rospy.get_param("/inference")
        self.dec_steps = rospy.get_param("/dec_steps") #STAR
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.nareas = rospy.get_param("/nareas") #Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i+1) for i in range(self.nareas)]  # list of int area IDs
        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data
        self.frontier_length = rospy.get_param("/frontier_length") #frontier length, either full length or truncated
        if self.frontier_length == 'None':
            self.frontier_length = math.inf
        self.debug("Frontier length: {}, {}".format(type(self.frontier_length), self.frontier_length))

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
        self.optimal_path = [] #container for the decided optimal path
        self.dist_matrix = None
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True
        self.curr_fmeasures = dict() #container of current F-measure of areas
        self.decay_rates_dict = dict() #dictionary for decay rates
        self.tlapses = dict()

        for area in self.areas:
            self.tlapses[area] = 0
            self.decay_rates_dict[area] = None
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
            # self.best_decision = None

    def consume_battery(self, start_area, next_area, curr_measure, noise):
        """
        Estimates battery consumption for the duration of the visit next_area from start_area.
        This duration includes the distance plus F-measure restoration, if any
        """

        #Batt consumed in travel
        distance = self.dist_matrix[int(start_area), int(next_area)]
        distance += noise * distance
        travel_time = (distance / self.robot_velocity)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time

        #Batt consumed in area restoration
        if next_area != self.charging_station:
            battery_consumed += self.batt_consumed_per_restored_f * (self.max_fmeasure - curr_measure)

        return battery_consumed

    def computeDurationMatrix(self, distance_matrix):
        return distance_matrix / self.robot_velocity

    def getHeuristicValue(self, label, decay_rates, tlapse_schedule):
        """
        Computes admissible loss. Should not underestimate the loss of remaining tlapse (upper bound)
        Inputs:
          label - Label object which we are currently computing heuristic value
          decay_rates - container of decay rates of areas
          tlapse_schedule - tlapse of a schedule, (time budget)
        """
        admissible_loss = []
        tlapse_to_go = tlapse_schedule - label.tlapse
        # self.debug("Path in heuristic evaluation {}".format(label.path))
        for area in decay_rates:
            if area not in label.path:
                loss_to_go = self.computeLoss(decay_rates[area], tlapse_to_go)
                admissible_loss.append(loss_to_go)
        if len(admissible_loss) > 0:
            return max(admissible_loss)
        else:
            return 0

    def computeLoss(self, decay_rate, tlapse, max_fmeasure=100):
        fmeasure = decay(decay_rate, tlapse, max_fmeasure)
        loss = loss_fcn(max_fmeasure, fmeasure)
        return -loss

    def isPruned(self, label, label_soln, frontier, dec_steps):
        """
        Note that the loss is naturally >0. Our goal is to find the schedule that has minimal loss. By applying -loss and applying a maximization scheme, we get an
          equivalent minimization solution.

        Returns true if label is pruned by:
          cond1 - f(l) is less than g(l_d) of incumbent solution
          cond2 - whether adding l as another decision is more than the number of decisions to make
          cond3 - whether l is dominated by any labels that have reached v(l)
        """
        # Cond 1
        # TODO: Current loss < 0 to deal with instantiated values
        if (label_soln is not None) and (len(label_soln.path) > 0 and label.path <= label_soln.path) and (label.valuation <= label_soln.current_loss):
            # self.debug("Did not pass pruning cond1. Label valuation: {} <= Label solution current loss: {}".format(label.valuation, label_soln.current_loss))
            return True

        # Cond 2
        if (label_soln is not None) and len(label_soln.path) + 1 > dec_steps:
            # self.debug("Did not pass pruning cond2. Exceeded dec_steps={}: {}".format(dec_steps, len(label_soln.path) + 1))
            return True

        # Cond 3
        # TODO: Any schedule length > those in frontier incurs larger loss and thus will be pruned.
        for label_prime in frontier:
            if label_prime.path <= label.path and (len(label_prime.path) > 0 and label.path <= label_prime.path) and label_prime.current_loss >= label.current_loss:
                # self.debug("Did not pass pruning cond3. Dominated by some label in frontier. Label_prime {} current loss: {} >= Label current loss: {}".format(
                #         label_prime.vertex, label_prime.current_loss, label.current_loss))
                return True
        return False

    def backtrack(self, tail_node, root=None):
        """
        #TODO: To adapt in this context
        Back tracks the path from tail node to parent node

        """
        nodes_list = [tail_node.vertex]
        parent = tail_node.parent
        if parent is not None:
            while True:
                if parent.parent is root:
                    break
                tail_node = parent
                nodes_list.append(tail_node.vertex)
                parent = tail_node.parent

            nodes_list.reverse()
        return nodes_list

    def estimate_tlapse_schedule(self, duration_matrix, dec_steps_togo):
        """
        Estimates the tlapse of schedule given the average duration of a decision and decision steps
        """
        average_duration = np.mean(duration_matrix)
        tlapse_schedule = average_duration * dec_steps_togo
        return tlapse_schedule

    def compute_decsteps_togo(self, l, dec_steps):
        """
        Computes the number of decision steps remaining to go
        """
        return dec_steps - len(l.path)

    def decision_making(self):
        """
        Given current location, come up with optimal schedule by RMA* search
        :return:
        """
        distance_matrix = self.dist_matrix.copy() * (1 + self.noise) #distance_matrix = np.array([[0, 1, 1, 1] * 3, [1, 0, 1, 1] * 3, [1, 1, 0, 1] * 3, [1, 1, 1, 0] * 3] * 3)
        duration_matrix = self.computeDurationMatrix(distance_matrix)
        # dec_steps = 4
        # frontier_length = math.inf  # tunable parameter for length of frontier. trade-off between optimality and efficiency

        #TODO: We probably will have problem here with charging station
        if self.curr_loc == self.charging_station:
            init_tlapse = 0
            init_loss = 0 #self.computeLoss(decay_rates[curr_loc], init_tlapse)  # TODO: Should this be present loss? Or shall we set as -math.inf? In ROS integration, which works right? The latter I think
        else:
            init_tlapse = self.tlapses[self.curr_loc]
            init_loss = self.computeLoss(self.decay_rates_dict[self.curr_loc], init_tlapse)
        init_path = set()

        l_0 = Label(vertex=self.curr_loc, current_loss=init_loss, tlapse=init_tlapse, path=init_path, parent=None)  # TODO: Be cautious of how we instantiate parameters of label
        # There is no goal vertex, just a number of decision steps to make. We estimate its equivalent tlapse
        dec_steps_togo = self.compute_decsteps_togo(l_0, self.dec_steps)
        tlapse_schedule = self.estimate_tlapse_schedule(duration_matrix, dec_steps_togo)
        l_0.heuristic = self.getHeuristicValue(l_0, self.decay_rates_dict, tlapse_schedule)  # heuristic
        l_0.computeValuation()  # current_cost + heuristic_cost

        open = DualCriteriaPriorityQueue()
        open.push(l_0)
        frontier = Frontier(self.frontier_length)
        frontier.add(l_0)
        l_d = None  # incumbent solution

        """
        For consideration: If we care only about the number of decision steps, then we should just have a counter of the soln path, whether it is
        """

        while not open.is_empty():
            # self.debug("Current open: {}. Highest priority: {}".format(open._queue, open.peek().vertex))
            l = open.pop()
            # self.debug("Popped label vertex: {}".format(l.vertex))
            if self.isPruned(l, l_d, frontier.frontier, self.dec_steps):
                # self.debug("Label is pruned in first pruning")
                continue
            frontier.filterAddFront(l)
            # self.debug("Label is added to frontier")

            # cond for instantiation or label has more length or the longer path and within dec_steps
            if (((l_d is None or (l_d.current_loss == 0.0 and len(l_d.path) < self.dec_steps)) or (len(l_d.path) < len(l.path)) or
                ((l.current_loss > l_d.current_loss) and (l.path >= l_d.path))) and (len(l.path) <= self.dec_steps)):  # We dont have goal vertices. Our goal is to reach desired number of decisions
                l_d = l
                # self.debug("Label now set as new incumbent solution: {}, {}, {} \n".format(l_d.current_loss, l_d.path, self.backtrack(l_d)))

            successors = l.getSuccessors(set(self.areas))  # Returns a list of successor/vertices
            # self.debug("Label successors: {}".format(successors))

            for s in successors:
                # self.debug("Evaluating successor: {}".format(s))
                # Move to vertex or stay in place
                tlapse = l.tlapse + self.tlapses[s] + duration_matrix[l.vertex - 1, s - 1]  # l.tlapse + duration_matrix[l.vertex-1, s-1]
                current_loss = l.current_loss + self.computeLoss(self.decay_rates_dict[s], tlapse)  # TODO: Compute for the loss. And then here the decay rate of the next vertex to move to? While this one is the loss of moving and visiting the next vertex
                path = l.path.copy()
                path.add(s)
                # self.debug("Moved to vertex {}. Tlapse: {}. Current loss: {}. Updated path: {}".format(s, tlapse, current_loss, path))
                l_s = Label(vertex=s, current_loss=current_loss, tlapse=tlapse, path=path, parent=None)
                dec_steps_togo = self.compute_decsteps_togo(l_s, self.dec_steps)  # Update remaining dec_steps to go
                tlapse_schedule = self.estimate_tlapse_schedule(duration_matrix, dec_steps_togo)  # Update remaining tlapse to go
                # self.debug("Decsteps to go: {}. Tlapse to go: {}".format(dec_steps_togo, tlapse_schedule))
                l_s.heuristic = self.getHeuristicValue(l_s, self.decay_rates_dict, tlapse_schedule)
                l_s.computeValuation()
                # self.debug("New label created: {}. Heuristic loss: {}. Valuation: {}".format(l_s.vertex, l_s.heuristic, l_s.valuation))

                if self.isPruned(l_s, l_d, frontier.frontier, self.dec_steps):
                    # self.debug("New label is pruned in second pruning")
                    continue
                else:
                    l_s.parent = l
                    open.push(l_s)
                    # self.debug("New label is pushed to Open and parent assigned {}".format(l_s.parent.vertex))

        min_path = self.backtrack(l_d, None)
        # self.debug("Optimal soln. Tlapse: {}. Total loss: {}. Opt schedule: {}".format(l_d.tlapse, l_d.current_loss, min_path))

        return min_path

    def update_tlapses_areas(self):
        """
        Lapses all time elapsed for each area
        :return:
        """
        for area in self.areas:
            self.tlapses[area] += 1
        # self.debug("Time elapsed since last restored: {}".format(self.tlapses))


    #Methods: Run operation
    def run_operation(self, filename, freq=1):
        """
        :return:
        """

        if self.robot_id < 999:
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
                    self.debug('Best path: {}. Process time: {}s'.format(self.optimal_path, think_elapsed))
                    self.update_robot_status(robotStatus.IN_MISSION)

                elif self.robot_status == robotStatus.IN_MISSION.value:
                    self.debug('Robot in mission. Total distance travelled: {}'.format(self.total_dist_travelled))
                    if self.available:
                        self.commence_mission()

                elif self.robot_status == robotStatus.CHARGING.value:
                    self.debug('Waiting for battery to charge up')

                elif self.robot_status == robotStatus.RESTORING_F.value:
                    self.debug('Restoring F-measure')

                if len(self.decisions_made)>0 or (self.robot_status != robotStatus.IDLE.value) and (
                        self.robot_status != robotStatus.READY.value) and (
                        self.robot_status != robotStatus.CONSIDER_REPLAN.value):
                    self.update_tlapses_areas()  # Update the tlapse per area
                    self.compute_curr_fmeasures()

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
        Thinks of the best decision before starting mission
        :return:
        """
        self.optimal_path = self.decision_making()

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
            if self.mission_area is not self.charging_station:
                battery_consumed = self.consume_battery(self.curr_loc, self.mission_area, self.curr_fmeasures[self.mission_area], self.noise)
                if not is_feasible(self.battery, battery_consumed, self.battery_reserve):
                    self.debug('Not enough battery to visit {}. Heading back to charging station and resetting schedule...'.format(self.mission_area))
                    self.optimal_path = []
                    self.mission_area = self.charging_station

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
        # Store the decay rates at instance, (prior knowledge)
        if self.decay_rates_dict[area_id] == None and msg.data is not None:
            if self.robot_id < 999: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
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
        # self.debug("Used for computation. Tlapses: {}. Decay rates: {}".format(self.tlapses, self.decay_rates_dict))
        # self.debug("Computed current f-measures: {}".format(self.curr_fmeasures))

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)

if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation/results')
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    Robot('rma_search').run_operation(filename)

