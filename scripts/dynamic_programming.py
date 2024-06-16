#!/usr/bin/env python3

"""
Heuristic decision making

"""
import json
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
        if type(f_thresh) is str:
            self.fsafe, self.fcrit = json.loads(f_thresh)
        elif type(f_thresh) is list:
            self.fsafe, self.fcrit = f_thresh

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        if type(batt_consumed_per_time) is str:
            self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = json.loads(
                batt_consumed_per_time)  # (travel, restoration)
        elif type(batt_consumed_per_time) is list:
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
        self.optimal_path = [] #container for the decided optimal path
        self.dist_matrix = None
        self.graph_areas = None #TODO: Graph of connected among vertices
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True
        self.curr_fmeasures = dict() #container of current F-measure of areas
        self.decay_rates_dict = dict() #dictionary for decay rates

        for area in self.areas:
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

    # def mean_duration_decay(self, duration_matrix, area):
    #     """
    #     Measures the average duration that an area decays (when robot commits to another decision than restoring area)
    #     Note that each column of the duration matrix represents that the corresponding area is the decision that the robot is committing to.
    #     We thus delete the column of the area, and then take the average to measure the average duration.
    #     :param duration_matrix:
    #     :param area:
    #     :return:
    #     """
    #     submatrix = np.delete(duration_matrix, area, axis=1)
    #     return np.mean(submatrix)

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

    #TODO: Insert the decision-making here
    # Step 0: Create a graph given distance matrix
    # This can be an instance variable
    def create_graph(self, dist_matrix):
        """
        Creates graph among areas (excluding charging station) given distance matrix
        """
        graph = nx.Graph()
        graph.add_nodes_from(list(range(len(dist_matrix))))
        self.debug("Nodes: {}".format(graph.nodes))
        edges = list()
        for i in graph.nodes:
            for j in graph.nodes:
                if dist_matrix[i, j] is not None:
                    edge = (i, j)
                    edges.append(edge)
        graph.add_edges_from(edges)
        return graph

    # TODO: This one is called every time we area planning from current location

        #TODO: The tlapse for the root node will be the current tlapse, as well as for the other areas
        # How? We get the current F-measures of all the areas and then invert.
        # We can then set as the initial time for each vertex at first creation
        # Note that we are excluding going to the charging station, since we suppose that the robot will charge up when no more feasible visit
        # The vertices therefore are just based on the areas.
        # Then in creating a graph, we should set this.

    def create_spatio_temporal_DAG(self, current_loc, G, duration_matrix, decay_rates, tlapses_init, k):
        """
        Creates spatio temporal DAG, G', from current_location based on G for schedule length k (time window).
        The nodes of G' will have representation (G x k)
        Inputs:
                current_loc - current location of robot
                G - graph
                decay_rates - list of decay rates
                tlapses_init - dict of tlapse for each area
                duration_matrix - travel duration from one node to another
                k - schedule length
        """

        nareas = len(G.nodes)
        assert k <= nareas, "Constraint error: schedule length <= number of areas, since an area is to be visited at most once"

        dag = nx.DiGraph()
        stemp_nodes = dict()

        # Create root node at i=0
        i = 0
        name = '{}_{}'.format(current_loc, i)
        decay_rate, tlapse_init = None, None
        if current_loc != self.charging_station:
            decay_rate = decay_rates[current_loc]
            tlapse_init = tlapses_init[current_loc]

        root_node = Node(name, id=current_loc, decay_rate=decay_rate, tlapse_init=tlapse_init)  # Root node
        dag.add_node(root_node)

        # Form weighted edges for the k visits
        stemp_nodes[0] = {name: root_node}
        prev_node = root_node
        for i in range(1, k + 1):
            stemp_nodes[i] = dict()
            stemp_edges = list()
            for prev_node in list(stemp_nodes[i - 1].values()):
                for n in list(G.neighbors(prev_node.id)):
                    if n != prev_node.id and n != self.charging_station:
                        name = '{}_{}'.format(n, i)
                        new_node = Node(name, id=n, decay_rate=decay_rates[n], tlapse_init=tlapses_init[n],
                                        tlapse_post_init=prev_node.tlapse_post_init,
                                        tlapse_visit=duration_matrix[prev_node.id, n])
                        self.debug("Tlapse: {}. Decay: {}. Loss: {}".format(new_node.tlapse, new_node.decay_rate,
                                                                       new_node.loss))

                        stemp_nodes[i][name] = new_node

                        edge = (prev_node, new_node, new_node.weight)  # weighted edge
                        stemp_edges.append(edge)
            dag.add_nodes_from(list(stemp_nodes[i].values()))
            dag.add_weighted_edges_from(stemp_edges)

            self.debug("i={}: {}, {}".format(i, [node.name for node in list(stemp_nodes[i].values())],
                                        [(edge[0].name, edge[1].name) for edge in stemp_edges]))
        return dag

    #TODO: This happens next after DAG
    def topological_sort_dag(self, dag):
        """
        Topologically sorts a DAG object
        """
        ordered = list(nx.topological_sort(dag))
        return ordered

    def backtrack(self, tail_node, root_name):
        """
        Back tracks the path from tail node to parent node

        """
        nodes_list = [tail_node]
        parent = tail_node.parent
        while parent.name != root_name:
            tail_node = parent
            nodes_list.append(tail_node.id)
            parent = tail_node.parent

        nodes_list.reverse()
        return nodes_list

    #TODO: Minimal loss path
    def min_loss_path(self, dag, sorted_nodes, current_loc):
        """
        Returns the path that yields the minimal loss in a DAG of length k starting from the root node, which is the current location
        """
        root_name = '{}_{}'.format(current_loc, 0)
        for node in sorted_nodes:
            if node.name == root_name:  # root node
                node.sum = 0
                self.debug("Root node: {}. sum: {}\n".format(node.name, node.sum))
            for succ in list(dag.successors(node)):
                self.debug("\n{} -> {}".format(node.name, succ.name))
                self.debug("Node {} path: {}. Succesor node: {}. Succesor in path?: {}".format(node.name, node.path, succ.id,
                                                                                          succ.id in node.path))
                if succ.id not in node.path:
                    self.debug("Node sum: {} - Succ loss: {} <= Succ sum: {} => {}".format(node.sum, succ.loss, succ.sum,
                                                                                      node.sum + succ.loss <= succ.sum))
                    if node.sum - succ.loss <= succ.sum:
                        succ.sum = node.sum - succ.loss
                        succ.parent = node
                        succ.path = node.path.copy()
                        succ.path.append(succ.id)
                        self.debug("Updated {} sum: {}. parent: {}. path: {}".format(succ.name, succ.sum, succ.parent.name,
                                                                                succ.path))

        # Search for the minimal sum among nodes in dag
        min_node = list(dag.nodes)[0]
        for node in list(dag.nodes):
            self.debug("Min node: {}, sum: {}. Next node: {}, sum: {}".format(min_node.name, min_node.sum, node.name,
                                                                         node.sum))
            if (node != min_node) and (not math.isinf(node.sum)) and (node.sum <= min_node.sum):
                min_node = node
                self.debug("replaced")
        self.debug("Min node: {}. sum: {}. parent: {}. path: {}".format(min_node.name, min_node.sum, min_node.parent.name,
                                                                   min_node.path))

        # Retrieve a path
        # path = self.backtrack(min_node, root_name)
        path = min_node.path
        self.debug("Decided path: {}".format(path))
        # print("\nNode id path:", [node.id for node in path])
        return path

    #I have 4hours to create this and make this happen
    def decision_making(self):
        """
        Given current location, come up with optimal schedule by dynamic programming
        :return:
        """


        """
        Step 1: Create a DAG from current location for length k
            Inputs:
                > current_loc
                > graph_areas
                > duration_matrix
                > decay_rates
                > tlapses_init, dict where areas are the keys, tlapse = get_time_given_decay(max_fmeasure, decayed_fmeasure, rate)
                > k
            Output: dag            
        Step 2: Topologically sort
            Input: dag
            Output: sorted nodes
        Step 3: Get the minimal feasible path
            Inputs:
                > dag
                > sorted nodes
                > current_loc
            Output:
                > path of length k to be taken from current loc (which means this is not included in the path)
        
        Ensuring feasibility:        
        PO1: We get the minimal path. Then from current battery we can estimate the consumption of the path.
        We then truncate those visits that are beyond estimated capacity.
        
        PO2: At each iteration where we pop the visits from the schedule, we check whether it is feasible. If not, we choose the charging station and reset the path.
        >> PO2 is easier to implement but also practical
        """

        #Step 1
        duration_matrix = self.dist_matrix/self.robot_velocity
        tlapses_init = dict()
        for area in self.areas:
            tlapses_init[area] = get_time_given_decay(max_fmeasure=self.max_fmeasure, decayed_fmeasure=self.curr_fmeasures[area], rate=self.decay_rates_dict[area])
        dag = self.create_spatio_temporal_DAG(self.curr_loc, self.graph_areas, duration_matrix, self.decay_rates_dict, tlapses_init, self.dec_steps)

        #Step 2
        ordered = self.topological_sort_dag(dag)

        #Step 3
        min_path = self.min_loss_path(dag, ordered, self.curr_loc)
        return min_path


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
            #TODO: Ensure decay rates dict has non-empty rates
            self.debug("Sufficent data. Decay rates: {}. Sampled nodes poses: {}".format(self.decay_rates_dict, self.sampled_nodes_poses))
            self.build_dist_matrix()
            t = 0
            while not rospy.is_shutdown() and t<self.t_operation:
                self.robot_status_pub.publish(self.robot_status)
                self.status_history.append(self.robot_status)
                if self.robot_status == robotStatus.IDLE.value:
                    self.debug('Robot idle')
                    if self.dist_matrix is not None:
                        #TODO (DONE): Initialize the graph if none yet
                        if self.graph_areas is None:
                            self.graph_areas = self.create_graph(self.dist_matrix)
                        self.update_robot_status(robotStatus.READY)

                elif self.robot_status == robotStatus.READY.value:
                    self.debug('Robot ready')
                    think_start = process_time()
                    self.think_decisions() #TODO: Update the method here
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
        #TODO: Update the method here
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
        #TODO: Evaluate whether next visit decided is still feasible, otherwise send back to charging station and reset the path
        # Use the structure from treebased_decision

        """
        if len(self.optimal_path):
            self.mission_area = self.optimal_path.pop(0)
            PO: Check for feasibility if visit is to restore area
            If feasible send to popped mission_area,
                if not, send to charging station, then reset optimal_path = []
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
        if self.decay_rates_dict[area_id] == None:
            if self.robot_id == 0: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates_dict[area_id] = msg.data
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
    Robot('dynamic_programming').run_operation(filename)

