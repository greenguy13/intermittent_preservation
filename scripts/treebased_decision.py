#!/usr/bin/env python

"""
Tree-based decision making

    Given all feasible areas and not in safe zone for the next k decision steps:
    Process:
        1. Stack all the combination of length k
        2. Compute cost
        3. Pick the least cost
"""
from enum import Enum
import random as rd
import numpy as np
import rospy
import actionlib
from loss_fcns import *
from cost_fcns import *
from pruning import *
import tf
import project_utils as pu
from scipy.spatial import distance, Voronoi
import igraph
from grid import Grid
from int_preservation.srv import flevel, flevelRequest
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import Int8, Float32
from visualization_msgs.msg import Marker
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

"""
Phases of progress toward completion:
    P1: Compute distance matrix (Done)
    P2: Send the robot to each area (Done)
    P3: Restore F and battery (For review)
    P4: Incorporate decision making via decision tree given forecast steps, k (For review)
    P5: Gather and analyze baseline results
    
Currently in P4. Sub-phases:
    S1: Ensure the run_operation cycles of robot, battery, and area work. [Works!]
    S2: Ensure the initialized location is the charging station
        > For now, we go to the charging station 
        > Moreover, the first element of the path, which is the current location should not be visited anymore when it is already taken cared of in the previous operation [DONE!]
    S3: Verify the thinking decisions: grow_tree, pruning, appending of branches, optimal branch
    S4: Verify the robot parameters are the same with battery and area for the noise, restoration, and batt consumed per unit time
"""

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3 #GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html


"""
TODO: A better way for the class enums is to have them outside the constructors and then just import them if we wish to use them.
"""

class batteryStatus(Enum):
    IDLE = 0
    DEPLETING = 1
    CHARGING = 10
    FULLY_CHARGED = 11

class areaStatus(Enum):
    IDLE = 0
    DECAYING = 1
    RESTORING_F = 10
    RESTORED_F = 11

class Robot:
    IDLE = 0
    READY = 11
    IN_MISSION = 20
    CHARGING = 30
    RESTORING_F = 40

    def __init__(self, node_name):
        """

        :param node_name:
        :param areas:
        :param est_distance_matrix:
        :param est_batt_consumption_matrix:
        """

        rospy.init_node(node_name, anonymous=True)
        self.listener = tf.TransformListener()

        #Parameters
        self.robot_id = rospy.get_param("~robot_id")
        self.debug_mode = rospy.get_param("/debug_mode")  # 0 or 1
        self.robot_velocity = rospy.get_param("/robot_velocity") #Linear velocity of robot; we assume linear and angular are relatively equal
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery") #Max battery
        self.fsafe, self.fcrit = rospy.get_param("/f_thresh") #(safe, crit)
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = rospy.get_param("/batt_consumed_per_time") #(travel, restoration)
        self.decay_rates_dict = rospy.get_param("/decay_rates_dict") #Decay rate of areas
        self.areas = list(self.decay_rates_dict.keys())
        self.areas = [int(i) for i in self.areas] #list of int IDs
        self.dec_steps = rospy.get_param("/dec_steps")
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.robot_radius = rospy.get_param("/robot_radius") #for grid cell computation
        self.nsample_nodes = rospy.get_param("/area_count") + 1 #Sample nodes from voronoi equal to area count plus 1 (for charging station)
        self.seed = 100 + 10*rospy.get_param("/run")
        self.degree_criterion_node_selection = rospy.get_param("/degree_criterion_node_selection")
        self.charging_station_radius = rospy.get_param("/charging_station_radius")  # if we are within the radius of the charging station
        self.tolerance = rospy.get_param("/move_base_tolerance")

        #Initialize variables
        self.charging_station = 0  # NOTE: In my code, this should indexed as 0. charging station area, pose. Later, randomly pick from among the nodes in Voronoi
        self.curr_loc = self.charging_station #Initial location robot is the charging station #TODO: Update the sampled_nodes_poses such that the first pose of the charging station in the map. How though? Transformation
        self.battery = self.max_battery
        self.optimal_path = []
        self.graph, self.dist_matrix = None, None
        self.sampled_nodes_poses = list()  # list of sampled nodes of type PoseStamped, which are the charging station and areas to preserve
        self.mission_area = None
        self.robot_status = self.IDLE
        self.available = True
        self.curr_fmeasures = dict() #container of current F-measure of areas

        #Publishers/Subscribers
        # Service request to move_base to get plan : make_Plan
        server = '/robot_' + str(self.robot_id) + '/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        rospy.Subscriber('/map', OccupancyGrid, self.static_map_callback)
        rospy.Subscriber('/robot_{}/battery_status'.format(self.robot_id), Int8, self.battery_status_cb)

        for area in self.areas:
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb, area) #REMARK: Here we assume that we have live measurements of the F-measures
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb)

        self.marker_pub = rospy.Publisher('voronoi', Marker, queue_size=0)
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
        """

        """
        On area restoration:
            Robot's mission area is a specific area. If reaches the area, it changes its status to RESTORING_F.
            Now, the current mission area, which subscribes to both robot_status and robot_mission_area topics, will restore F; while,
                those other areas not the mission area will have their F continually decay 
        """

    # MAP/NAVIGATION METHODS
    def static_map_callback(self, data):
        """
        Callback for grid
        :param data:
        :return:
        """
        self.latest_map = Grid(data)

        self.min_range_radius = 8 / self.latest_map.resolution
        self.min_edge_length = self.robot_radius / self.latest_map.resolution
        self.compute_gvg()
        if len(self.sampled_nodes_poses) == 0:
            self.sample_nodes_from_voronoi()
            self.build_dist_matrix()

    def compute_gvg(self):
        """Compute GVG for exploration."""

        start_time_clock = rospy.Time.now().to_sec()
        # Get only wall cells for the Voronoi.
        obstacles = self.latest_map.wall_cells()
        end_time_clock = rospy.Time.now().to_sec()
        pu.log_msg('robot', self.robot_id,"generate obstacles2 {}".format(end_time_clock - start_time_clock),self.debug_mode)

        start_time_clock = rospy.Time.now().to_sec()

        # Get Voronoi diagram.
        vor = Voronoi(obstacles)
        end_time_clock = rospy.Time.now().to_sec()
        pu.log_msg('robot', self.robot_id,"voronoi {}".format(end_time_clock - start_time_clock),self.debug_mode)
        start_time_clock = rospy.Time.now().to_sec()

        # Initializing the graph.
        self.graph = igraph.Graph()
        # Correspondance between graph vertex and Voronoi vertex IDs.
        voronoi_graph_correspondance = {}

        # Simplifying access of Voronoi data structures.
        vertices = vor.vertices
        ridge_vertices = vor.ridge_vertices
        ridge_points = vor.ridge_points

        edges = []
        weights = []
        # Create a graph based on ridges.
        for i in range(len(ridge_vertices)):
            ridge_vertex = ridge_vertices[i]
            # If any of the ridge vertices go to infinity, then don't add.
            if ridge_vertex[0] == -1 or ridge_vertex[1] == -1:
                continue
            p1 = vertices[ridge_vertex[0]]
            p2 = vertices[ridge_vertex[1]]

            # Obstacle points determining the ridge.
            ridge_point = ridge_points[i]
            q1 = obstacles[ridge_point[0]]
            q2 = obstacles[ridge_point[1]]

            # If the vertices on the ridge are in the free space
            # and distance between obstacle points is large enough for the robot
            if self.latest_map.is_free(p1[INDEX_FOR_X], p1[INDEX_FOR_Y]) and \
                self.latest_map.is_free(p2[INDEX_FOR_X], p2[INDEX_FOR_Y]) and \
                    distance.euclidean(q1, q2) > self.min_edge_length:

                # Add vertex and edge.
                graph_vertex_ids = [-1, -1] # temporary for finding verted IDs.

                # Determining graph vertex ID if existing or not.
                for point_id in range(len(graph_vertex_ids)):
                    if ridge_vertex[point_id] not in voronoi_graph_correspondance:
                        # if not existing, add new vertex.
                        graph_vertex_ids[point_id] = self.graph.vcount()
                        self.graph.add_vertex(coord=vertices[ridge_vertex[point_id]])
                        voronoi_graph_correspondance[ridge_vertex[point_id]] = graph_vertex_ids[point_id]
                    else:
                        # Otherwise, already added before.
                        graph_vertex_ids[point_id] = voronoi_graph_correspondance[ridge_vertex[point_id]]

                # Add edge.
                self.graph.add_edge(graph_vertex_ids[0], graph_vertex_ids[1], weight=distance.euclidean(p1, p2))
            else:
                # Otherwise, edge not added.
                continue

        # Take only the largest component.
        cl = self.graph.clusters()
        self.graph = cl.giant()
        self.prune_leaves()
        end_time_clock = rospy.Time.now().to_sec()
        pu.log_msg('robot', self.robot_id,"ridge {}".format(end_time_clock - start_time_clock),self.debug_mode)

        # Publish GVG.
        self.publish_edges()

    def prune_leaves(self):
        current_vertex_id = 0 # traversing graph from vertex 0.
        self.leaves = {}
        self.intersections = {}
        while current_vertex_id < self.graph.vcount():
            if self.graph.degree(current_vertex_id) == 1:
                neighbor_id = self.graph.neighbors(current_vertex_id)[0]
                neighbor = self.graph.vs["coord"][neighbor_id]
                current_vertex = self.graph.vs["coord"][current_vertex_id]
                if not self.latest_map.is_frontier(neighbor, current_vertex, self.min_range_radius):
                    self.graph.delete_vertices(current_vertex_id)
                else:
                    self.leaves[current_vertex_id] = self.latest_map.unknown_area_approximate(current_vertex)
                    current_vertex_id += 1
            else:
                if self.graph.degree(current_vertex_id) > 2:
                    self.intersections[current_vertex_id] = self.graph.vs["coord"][current_vertex_id]
                current_vertex_id += 1

    def publish_edges(self):
        """For debug, publishing of GVG."""

        # Marker that will contain line sequences.
        m = Marker()
        m.id = 0
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.LINE_LIST
        m.color.a = 1.0
        m.color.r = 1.0
        m.scale.x = 0.1

        # Plot each edge.
        for edge in self.graph.get_edgelist():
            for vertex_id in edge:
                # Grid coordinate for the vertex.
                p = self.graph.vs["coord"][vertex_id]
                p_t = self.latest_map.grid_to_pose(p)
                p_ros = Point(x=p_t[0], y=p_t[1])

                m.points.append(p_ros)

        # Publish marker.
        self.marker_pub.publish(m)

    def sample_nodes_from_voronoi(self):
        """
        Samples nodes from potential nodes generated by voronoi
        Potential nodes are filtered from the entire tree by self.degree_criterion_node_selection
        Called in static_map_callback
        :return:
        """
        rd.seed(self.seed+1)
        potential_nodes = self.graph.vs.select(_degree=self.degree_criterion_node_selection)
        sampled_nodes = rd.sample(potential_nodes.indices, self.nsample_nodes)
        print("Sampled nodes:", sampled_nodes)

        #Establish the coordinate dictionary here
        self.sampled_nodes_poses = list()
        for node in sampled_nodes:
            p = self.graph.vs[node]["coord"]
            p_t = self.latest_map.grid_to_pose(p)
            p_ros = (p_t[0], p_t[1])
            pose_stamped = self.convert_coords_to_PoseStamped(p_ros)
            self.sampled_nodes_poses.append(pose_stamped)

    def convert_coords_to_PoseStamped(self, coords, frame='map'):
        """
        Converts x,y coords to PoseStampled wrt frame
        :param coord:
        :return:
        """
        print("Coords:", coords, type(coords), type(coords[0]), type(coords[1]))
        pose = PoseStamped()
        pose.header.seq = 0
        pose.header.frame_id = frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = coords[0]
        pose.pose.position.y = coords[1]
        pose.pose.orientation.w = 1.0

        return pose

    def run_operation(self):
        """
        :return:
        """
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            if self.robot_id == 0:
                self.robot_status_pub.publish(self.robot_status)
                if self.robot_status == self.IDLE:
                    pu.log_msg('robot', self.robot_id, 'Robot idle', self.debug_mode)
                    if self.dist_matrix is not None:  # Here, the distance matrix we have to supply the correct distance computations
                        pu.log_msg('robot', self.robot_id, "Nodes to preserve: " + str(self.sampled_nodes_poses), self.debug_mode)
                        self.update_robot_status(self.READY)

                elif self.robot_status == self.READY:
                    pu.log_msg('robot', self.robot_id, 'Robot ready', self.debug_mode)
                    self.think_decisions()
                    pu.log_msg('robot', self.robot_id, 'Path: ' + str(self.optimal_path), self.debug_mode)
                    self.update_robot_status(self.IN_MISSION)

                elif self.robot_status == self.IN_MISSION:
                    pu.log_msg('robot', self.robot_id, 'Robot in mission', self.debug_mode)
                    if self.available:
                        self.commence_mission()

                elif self.robot_status == self.CHARGING:
                    pu.log_msg('robot', self.robot_id, 'Waiting for battery to charge up', self.debug_mode)

                elif self.robot_status == self.RESTORING_F:
                    pu.log_msg('robot', self.robot_id, 'Restoring F-measure', self.debug_mode)

            rate.sleep()

    def think_decisions(self):
        """
        Thinks of the optimal path before starting mission
        :return:
        """
        tree = self.grow_tree(self.dec_steps, self.restoration, self.noise)
        self.optimal_path = self.get_optimal_branch(tree)  # Indices of areas/nodes

    def commence_mission(self):
        """
        Commences mission
        :return:
        """
        if self.send2_next_area() == 0:
            self.update_robot_status(self.IDLE)

    def send2_next_area(self):
        """
        Sends the robot to the next area in the optimal path:
        :return:
        """
        if len(self.optimal_path):
            self.mission_area = self.optimal_path.pop(0)
            self.mission_area_pub.publish(self.mission_area)
            pu.log_msg('robot', self.robot_id, 'Heading to: {}. {}'.format(self.mission_area, self.sampled_nodes_poses[self.mission_area]), self.debug_mode)
            self.go_to_target(self.mission_area)
            return 1
        return 0

    #TODO: Update previous versions in updating the status of the robot
    def update_robot_status(self, status):
        """
        Updates robot status
        :param status:
        :return:
        """
        self.robot_status = status


    # METHODS: Build distance matrix
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
        #self.debug("start: {} {}. goal: {} {}. tol: {} {}.".format(type(req.start), req.start, type(req.goal), req.goal, type(req.tolerance), req.tolerance))
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
        self.dist_matrix = np.zeros((self.nsample_nodes, self.nsample_nodes))

        for i in range(self.nsample_nodes):
            for j in range(self.nsample_nodes):
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
        print("Movebase goal:", movebase_goal)
        self.available = False
        action_goal_cb = (lambda state, result: self.action_send_done_cb(state, result, self.robot_id))
        self.robot_goal_client.send_goal(movebase_goal, done_cb=action_goal_cb, active_cb=self.action_send_active_cb)

    def action_send_active_cb(self):
        """
        Sets robot as unavailable when pursuing goal
        :return:
        """
        self.available = False
        self.robot_status = self.IN_MISSION

    def action_send_done_cb(self, state, result, robot_id):
        """

        :param msg:
        :return:
        """
        # Note: Currently there is a tiny bug on the lambda
        print("Robot id: {}. Succeeded: {}".format(robot_id, state == SUCCEEDED))
        if state == SUCCEEDED:
            self.curr_loc = self.mission_area
            self.robot_status = self.RESTORING_F
            if self.mission_area == self.charging_station:
                self.robot_status = self.CHARGING


    #DECISION-MAKING methods
    #TODO: This can be a module of its own since we can have different decision making methods
    def grow_tree(self, dec_steps, restoration, noise):
        """
        We grow a decision tree of depth dec_steps starting from where the robot is.
        :param curr_location: current location of robot
        :param dec_steps: number of decision steps, (i.e., tree depth)
        :param restoration: duration it takes to restore per unit F-measure
        :param noise: anticipated noise in actual travel to an area
        :return:
        """
        #NOTE: WE NEED TO UPDATE IN THE PRUNING SUCH THAT THE ROBOT WILL NOT DIE IN THE MISSION/DURING THE OPERATION

        # st = rospy.time()
        # time.sleep(1)


        """
        Comments:
        1. How do we get updated F-measure of areas?
            > Scenario 1: We are able to monitor F in real-time
                Subscribe to topic 
            > Scenario 2: We only get to know the F-measure once we are there.
                We have our belief/computation/model of what F would be. We start with 100 and decay based on that model.
                As we go through our mission and visit areas, we constantly measure F. We then update our model (or parameters) of F for that area.
                
        2. When we are thinking, we designed it that we consider where we currently are
            > If this is the case, then the starting location should be the charging station. Note that this should be in map frame.
            > How will we do the conversion?
                We subscribe to odom
                We convert odom wrt map
        """

        branches = list() #container for final branches up to depth k
        to_grow = list()  #container for branches still being grown/expanded
        nodes = self.areas.copy()
        nodes.append(self.charging_station) #append the charging station
        pu.log_msg('robot', self.robot_id, "Nodes: {}".format(nodes), self.debug_mode)

        #Start at the current location as the root node.
        #Scenario 1
        fmeasures = self.curr_fmeasures.copy()
        pu.log_msg('robot', self.robot_id, "Areas: {}".format(self.areas), self.debug_mode)
        pu.log_msg('robot', self.robot_id, "Fmeasures: {}".format(fmeasures), self.debug_mode)

        k = 0  #
        #We need current location index
        path = [self.curr_loc]
        battery = self.battery
        cost = 0 #Initialize cost of path

        #Initial feasible battery level
        feasible_battery_consumption = self.consume_battery(start_area=self.curr_loc, next_area=self.charging_station, curr_measure=None, noise=noise)
        feasible_battery = battery - feasible_battery_consumption

        branch = (path, battery, fmeasures, cost, feasible_battery)
        to_grow.append(branch)



        #Succeeding decision steps:
        while k < dec_steps:
            pu.log_msg('robot', self.robot_id, "\nDec step: {}".format(k), self.debug_mode)
            consider_branches = to_grow.copy()
            to_grow = list() #At the end of the iterations, to-grow will be empty while branches must be complete
            for branch in consider_branches:
                pu.log_msg('robot', self.robot_id, "Branch to grow: {}".format(branch), self.debug_mode)
                considered_growing= 0 #Indicators whether the branch has been considered for growing
                for i in range(len(nodes)):
                    # Hypothetical: What if we travel to this node, what will the consumed battery be and the updated F-fmeasures?
                    # Shall we prune this in the next decision step?
                    start_area = path[-1]
                    next_area = nodes[i] #We are considering travelling to all nodes
                    path = branch[0].copy()  # path at depth k
                    battery = branch[1]  # battery level at depth k
                    fmeasures = branch[2].copy()  # current fmeasures at depth k
                    cost = branch[3] # cost of path at depth k

                    if next_area != self.charging_station:
                        curr_measure = fmeasures[next_area]
                        tlapse_decay = get_time_given_decay(self.max_fmeasure, curr_measure, self.decay_rates_dict[str(next_area)])
                        duration = self.compute_duration(start_area=start_area, next_area=next_area, curr_measure=curr_measure, restoration=restoration, noise=noise) #Duration if we preserve area
                        decayed_fmeasure = decay(self.decay_rates_dict[str(next_area)], tlapse_decay+duration, self.max_fmeasure)  # Decayed measure of area if we travel there
                    else:
                        curr_measure, tlapse_decay, decayed_fmeasure = None, None, None
                        duration = self.compute_duration(start_area=start_area, next_area=self.charging_station, curr_measure=battery, restoration=restoration, noise=noise) #Duration if we charge up

                    #Battery consumption
                    battery_consumption = self.consume_battery(start_area=start_area, next_area=next_area, curr_measure=curr_measure, noise=noise)
                    battery_consumption_backto_charging_station = self.consume_battery(start_area=next_area, next_area=self.charging_station, curr_measure=None, noise=noise)
                    feasible_battery_consumption = battery_consumption + battery_consumption_backto_charging_station

                    pu.log_msg('robot', self.robot_id, "Next area: {}, Batt level: {}, TLapsed decay: {}, Duration: {}, Decayed fmeasure: {}, Batt consumption: {}".format(next_area, battery, duration, battery_consumption, decayed_fmeasure, tlapse_decay), self.debug_mode)

                    # If branch is not to be pruned and length still less than dec_steps, then we continue to grow that branch
                    cond1 = prune(battery, feasible_battery_consumption, decayed_fmeasure, self.fsafe)
                    pu.log_msg('robot', self.robot_id, "Condition: {}".format(cond1), self.debug_mode)
                    if (cond1 is False and (k < dec_steps)) or next_area == self.charging_station:
                        path.append(next_area) #append next area as part of the path at depth k+1. #This is where the additional or overwriting happens. We need to make dummy list/container
                        if next_area != self.charging_station:
                            battery -= battery_consumption #actual battery depleted at depth k+1
                        else:
                            battery = self.max_battery #actual battery restored to max value
                        feasible_battery = battery - feasible_battery_consumption  # battery available after taking into account battery to go back to charging station from current location. Note: if location is charging station, feasible_battery = max_battery
                        updated_fmeasures = self.adjust_fmeasures(fmeasures, next_area, duration) #F-measure of areas adjusted accordingly, i.e., consequence of decision
                        cost += self.compute_cost(updated_fmeasures) #Cost of this decision
                        pu.log_msg('robot', self.robot_id, "Branch to grow appended: {}".format(path, battery, updated_fmeasures, cost, feasible_battery), self.debug_mode)
                        to_grow.append((path, battery, updated_fmeasures, cost, feasible_battery)) #Branch: (path, battery, updated_fmeasures, cost, feasible battery)
                        considered_growing += 1

                    #Else, we add that branch to branches (for return), which includes pruned branches. Conditions:
                    # 1.) Robot is not dead at the end of the operation, i.e., we check whether remaining feasible battery >= 0. If not, then this path ends dead, thus we don't append it.
                    # 2.) Furthermore: If even after iterating through possible areas to preserve, (thats why i equals number of nodes minus 1), branch not considered for growing.
                    # 3.) And branch not yet in branches.
                    else:
                        if (is_feasible(battery, feasible_battery_consumption) is True) and (i == len(nodes)-1 and considered_growing == 0) and (branch not in branches):
                            pu.log_msg('robot', self.robot_id, "Branch appended to tree: {}".format(branch), self.debug_mode)
                            branches.append(branch)
            k += 1 #We are done with k depth, so move on to the next depth

        #We append to branches the branches of length k, (i.e., the final decision step)
        print("We arrived at last decision step!")
        for branch in to_grow:
            if branch not in branches:
                branches.append(branch)
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
            restore_time = restoration * (max_restore - curr_measure)
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
        distance += noise * distance  # distance + noise #TODO: Is there really a need to incorporate noise in the thinking/planning process?
        travel_time = (distance / self.robot_velocity)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time #TODO: Is this parameter batt_consumed_per_travel_time synchronized with the one in battery.py?

        if next_area != self.charging_station:
            battery_consumed += self.batt_consumed_per_restored_f * (self.max_fmeasure - curr_measure) #TODO: Do we need this? Is this reflective in battery.py?

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
        pu.log_msg('robot', self.robot_id, "Branches sorted by cost:", self.debug_mode)
        for branch in sorted_branches:
            pu.log_msg('robot', self.robot_id, "Branch: {}".format(branch), self.debug_mode)
        pu.log_msg('robot', self.robot_id, "Optimal branch (branch info + cost): {}".format(sorted_branches[0]), self.debug_mode)
        optimal_path = sorted_branches[0][0] #pick the branch with least cost and most available feasible battery
        optimal_path.pop(0) #we pop out the first element of the path, which is the current location, which is not needed in current mission
        return optimal_path

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def battery_status_cb(self, msg):
        """

        :param msg:
        :return:
        """
        if msg.data == batteryStatus.FULLY_CHARGED.value:
            if self.robot_id == 0: pu.log_msg('robot', self.robot_id, "Fully charged!")
            self.available = True
            self.update_robot_status(self.IN_MISSION)

    def area_status_cb(self, msg):
        """

        :param msg:
        :return:
        """
        if msg.data == areaStatus.RESTORED_F.value:
            if self.robot_id == 0: pu.log_msg('robot', self.robot_id, "Area fully restored!")
            self.available = True
            self.update_robot_status(self.IN_MISSION)

    def area_fmeasure_cb(self, msg, area_id):
        """
        Updates fmeasure of area
        :param msg:
        :param area_id:
        :return:
        """
        self.curr_fmeasures[area_id] = msg.data


    # def run_operation(self):
    #     area = 0
    #     rospy.sleep(5)
    #     while not rospy.is_shutdown():
    #         if self.robot_id == 0:
    #             self.robot_status_pub.publish(self.robot_status)
    #             #Build the distance matrix
    #             if self.robot_status is self.IDLE:
    #                 if self.dist_matrix is not None:
    #                     self.robot_status = self.READY
    #
    #             elif self.robot_status is self.READY:
    #                 self.mission_area = None
    #                 self.robot_status = self.IN_MISSION
    #
    #             elif self.robot_status is self.IN_MISSION:
    #                 if self.available:
    #                     self.mission_area = area
    #                     self.mission_area_pub.publish(area)
    #                     goal = self.sampled_nodes_poses[area]  # Select (think about) the mission area
    #                     self.debug("Goal: {} {}".format(self.mission_area, goal))
    #                     self.send_robot_goal(goal)
    #                     area += 1
    #                     if area>=len(self.sampled_nodes_poses):
    #                         area = 0
    #         rospy.sleep(1)

    """
    UPNEXT: Phase 4. Seems okay.
    UPNEXT: Phase 5. Verified Phase 4 with results/analysis.
    """


if __name__ == '__main__':
    rd.seed(1234)
    Robot('treebased_decision').run_operation()

    #TODO: Just make sure we can run the world and then communicate with the robot to move to one goal to other
