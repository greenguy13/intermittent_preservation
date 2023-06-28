#!/usr/bin/env python

"""
Tree-based decision making

    Given all feasible areas and not in safe zone for the next k decision steps:
    Process:
        1. Stack all the combination of length k
        2. Compute cost
        3. Pick the least cost
"""
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

from int_preservation.msg import visitAction, visitGoal
from int_preservation.srv import flevel, flevelRequest
from int_preservation.srv import location, locationRequest
from int_preservation.srv import battery_level, battery_levelRequest
from nav_msgs.srv import GetPlan #GetPlanRequest
from nav_msgs.GetPlan.srv import make_plan make_planRequest
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import Int8
from visualization_msgs.msg import Marker
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


INDEX_FOR_X = 0
INDEX_FOR_Y = 1

class Robot:
    IDLE = 10
    READY = 11
    IN_MISSION = 12
    REQUEST2CHARGE = 30
    CHARGING = 31
    REQUEST2RESTORE_F = 40
    RESTORING_F = 41

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
        self.robot_velocity = rospy.get_param("/robot_velocity") #Linear velocity of robot; we assume linear and angular are relatively equal
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery") #Max battery
        self.fsafe, self.fcrit = rospy.get_param("/f_thresh") #(safe, crit)
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = rospy.get_param("/batt_consumed_per_time") #(travel, restoration)
        self.decay_rates_dict = rospy.get_param("/decay_rates_dict") #Decay rate of areas
        self.areas = list(self.decay_rates_dict.keys())
        self.dec_steps = rospy.get_param("/dec_steps")
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.debug_mode = rospy.get_param("/debug_mode") #0 or 1
        self.charging_station_radius =rospy.get_param("/charging_station_radius")
        self.robot_radius = rospy.get_param("/robot_radius")
        self.nsample_nodes = rospy.get_param("/nsample_nodes")
        self.seed = 100 + 10*rospy.get_param("/run")
        self.degree_criterion_node_selection = rospy.get_param("/degree_criterion_node_selection")

        #Publishers/Subscribers
        self.move_base_client_pub = rospy.Publisher('/move_base/goal', PoseStamped, queue_size=1) #TODO: move_base action client
        #self.point_pub = rospy.Publisher('/robot_{}/robot_nav/goal'.format(self.robot_id), Point, queue_size=1)
        #rospy.Subscriber('/robot_{}/robot_nav/feedback'.format(self.robot_id), Pose, self.robot_nav_callback)
        rospy.Subscriber('/map', OccupancyGrid, self.static_map_callback)
        self.marker_pub = rospy.Publisher('voronoi', Marker, queue_size=0)
        self.robot_status = self.IDLE
        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)

        #Service client to move_base: make_Plan
        rospy.wait_for_service('/move_base/make_plan') #TODO: Is this make_plan indeed?
        self.make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        global cast
        cast = type(self.areas[0])
        self.charging_station = cast(0)  # NOTE: In my code, this should indexed as 0. charging station area, pose. Later, randomly pick from among the nodes in Voronoi
        self.target_pose, self.target_id = None, None
        self.curr_pose, self.curr_loc = None, None #Current location pose and index
        self.battery = self.max_battery
        self.optimal_path = []
        self.graph, self.sampled_nodes, self.dist_matrix = None, None, None

    # RUN OPERATION methods
    def run_operation(self):
        """
        :return:
        """
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            if self.robot_status == self.IDLE:
                pu.log_msg('robot', self.robot_id, 'Robot idle', self.debug_mode)
                if self.dist_matrix is not None: #Here, the distance matrix we have to supply the correct distance computations
                    pu.log_msg('robot', self.robot_id, "Nodes to preserve: " + str(self.sampled_nodes_poses), self.debug_mode)
                    self.update_robot_status(self.READY)

            elif self.robot_status == self.READY:
                pu.log_msg('robot', self.robot_id, 'Robot ready', self.debug_mode)
                self.think_decisions()
                pu.log_msg('robot', self.robot_id, 'Path: ' + str(self.optimal_path), self.debug_mode)
                self.update_robot_status(self.IN_MISSION)

            elif self.robot_status == self.IN_MISSION:
                pu.log_msg('robot', self.robot_id, 'Robot in mission', self.debug_mode)
                if len(self.optimal_path) == 0:
                    self.update_robot_status(self.IDLE)
                self.commence_mission()

            elif self.robot_status == self.REQUEST2CHARGE:
                pu.log_msg('robot', self.robot_id, 'Request battery charge', self.debug_mode)
                #Request from charging station server
                self.request_charge()

            elif self.robot_status == self.CHARGING:
                pu.log_msg('robot', self.robot_id, 'Waiting for battery to charge up', self.debug_mode)

            elif self.robot_status == self.RESTORING_F:
                pu.log_msg('robot', self.robot_id, 'Restoring F-measure', self.debug_mode)
                self.restore_f_request()
            rate.sleep()

    def think_decisions(self):
        """
        Thinks of the optimal path before starting mission
        :return:
        """
        # tree = self.grow_tree(self.dec_steps, self.restoration, self.noise)
        # self.optimal_path = self.get_optimal_branch(tree) #Indices of areas/nodes

        #TO-UNCOMMENT: Here, we will navigate through the sampled node poses
        self.optimal_path = self.sampled_nodes_poses

    def commence_mission(self):
        """
        Commences mission
        :return:
        """
        self.update_robot_status(self.IN_MISSION)
        if self.send2_next_area() == 0:
            self.update_robot_status(self.IDLE)

    def send2_next_area(self):
        """
        Sends the robot to the next area in the optimal path:
        :return:
        """
        if len(self.optimal_path) > 0:
            # self.target_id = self.optimal_path.pop(0)
            # next_area = self.sampled_nodes_poses[self.target_id]

            #TODO: For now, we just go through the sampled node poses
            next_area = self.optimal_path.pop(0)
            pu.log_msg('robot', self.robot_id, 'Heading to: ' + str(next_area), self.debug_mode)
            self.go_to_target(next_area)
            return 1
        return 0

    def update_robot_status(self, status):
        """
        Updates robot status
        :param status:
        :return:
        """
        self.robot_status = status

    def go_to_target(self, goal):
        """
        Action client to move_base to move to target goal
        Goal is PoseStamped msg
        TODO: Let's sanity check the go_to_target
        :return:
        """
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()
        client.send_goal(goal)
        wait = client.wait_for_result()
        if not wait:
            rospy.logerr("move_base action server not available!")
            rospy.signal_shutdown("move_base action server not available!")
        else:
            if client.get_result():
                rospy.loginfo("Goal action achieved!")
                return client.get_result()


    """
    Applying the move_base in the program:
    1. Build the distance matrix prior to execution: via service request from move_base/GetPlan (start, goal) PoseStamped for each of the areas
    2. Build decision tree for a given forecast k
    3. Send the robot to each area after decision making, using move_base actionlib client
    4. Capture/store the data of F-measures
    """



    # def go_to_target(self, target):
    #     """
    #     Goes to cartesian (x, y) coordinate
    #     :param target:
    #     :return:
    #     """
    #     self.target_pose = target
    #
    #     #Create PoseStamped msg
    #     goal = PoseStamped()
    #     goal.header.stamp = rospy.Time.now()
    #     goal.header.frame_id = 'map' #TODO: PO. Is this map or something else
    #     goal.pose.position.x = target[0]
    #     goal.pose.position.y = target[1]
    #     self.move_base_client_pub.publish(goal)
    #     # self.point_pub.publish(Point(x=self.target_pose[0], y=self.target_pose[1]))
    #
    #     self.update_robot_status(self.IN_MISSION)

    # MAP/NAVIGATION METHODS
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
                if not self.latest_map.is_frontier(
                    neighbor, current_vertex, self.min_range_radius): # TODO parameter.
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
        # TODO constant values set at the top.
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

    def select_preservation_areas(self):
        """
        Selects areas to preserve from potential nodes
        Called in static_map_callback
        :return:
        """
        rd.seed(self.seed+1)
        potential_areas = self.graph.vs.select(_degree=self.degree_criterion_node_selection)
        self.sampled_nodes = rd.sample(potential_areas.indices, self.nsample_nodes)
        print("Sampled nodes:", self.sampled_nodes)

        #Establish the coordinate dictionary here
        self.sampled_nodes_poses = list()
        for node in self.sampled_nodes:
            p = self.graph.vs[node]["coord"]
            p_t = self.latest_map.grid_to_pose(p)
            p_ros = (p_t[0], p_t[1])
            self.sampled_nodes_poses.append(p_ros)

        #TO-DELETE: Try navigating through the sampled nodes/areas
        # self.optimal_path = self.sampled_nodes_poses[:]

    def convert_coords_to_PoseStamped(self, coords, frame='map'):
        """
        Converts x,y coords to PoseStampled wrt frame
        :param coord:
        :return:
        """
        msg = PoseStamped()
        msg.header.frame_id = frame
        msg.header.stamp = rospy.Time(0)
        msg.pose.position.x = coords[0]  # 2.6
        msg.pose.position.y = coords[1]  # 1.3

        return msg

    def build_dist_matrix(self):
        """
        Builds the distance matrix among selected areas for preservation

        TODO: Build the dist_matrix by requesting from make_plan service to get path plan from one area to another
        1. For each of the area, we get the path from one area to another
        2. Measure the distance of the path
        3. Filling out the entries of the dist_matrix

        :return:
        """
        n = len(self.sampled_nodes)
        self.dist_matrix = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                from_area, next_area = self.convert_coords_to_PostStamped(self.sampled_nodes_poses[i]), self.convert_coords_to_PostStamped(self.sampled_nodes_poses[j])
                #TODO: Ask from GetPlan the path from from_area to next_area
                #Compute the total distance for each path, which is one cell in the dist_matrix
                req = GetPlan()
                req.start = from_area
                req.goal = next_area
                req.tolerance = .5
                resp = self.make_plan(req.start, req.goal, req.tolerance)
                print(resp)
                dist = None

                #dist = distance.cityblock(from_area, next_area)
                # pu.log_msg('robot', self.robot_id, "From: {}. To: {}. distance: {}".format(from_area, next_area, dist), self.debug_mode)
                self.dist_matrix[i, j] = dist
        #pu.log_msg('robot', self.robot_id, "Distance matrix: {}".format(self.dist_matrix), self.debug_mode)

        # return dist_matrix
        # for i in range(n):
        #     for j in range(n):
        #         from_area, next_area = self.sampled_nodes[i], self.sampled_nodes[j]
        #         sp = self.graph.get_shortest_paths(from_area, next_area)[0]
        #         print("From: {}. To: {}. SP: {}".format(from_area, next_area, sp))
        #         pu.log_msg('robot', self.robot_id, "From: {}. To: {}. SP: {}".format(from_area, next_area, sp), self.debug_mode)
        #         distance = 0
        #         for h in range(len(sp) - 1):
        #             sp_from_area, sp_next_area = sp[h], sp[h + 1]
        #             weight = self.graph.es.select(_from=sp_from_area, _target=sp_next_area)['weight'] #[0]
        #             print("Edge: {}. Weight: {}".format((sp_from_area, sp_next_area), weight))
        #             pu.log_msg('robot', self.robot_id, "Distance: {}. Edge: {}. Weight: {}".format(distance, (sp_from_area, sp_next_area), weight), self.debug_mode)
        #             distance += weight
        #         print("Total distance:", distance)
        #         self.dist_matrix[i, j] = distance
        # print("Distance matrix:", self.dist_matrix)


    #REQUESTS and CALLBACKS
    ## Navigation
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
        if self.sampled_nodes is None:
            self.select_preservation_areas()
            #self.build_dist_matrix() #TO-FIX: base the distance info using nav2d computation

    ## THIS PART HERE NEEDS FIXING UP/CLARIFICATION
    def is_charging_station(self, target_pose):
        """
        PO1: pose, radius
        PO2: optimal path itself.
        :return:
        """
        if distance.euclidean(target_pose, self.charging_station) < self.charging_station_radius: #Or within the radius of the pose of charging station,
            return True
        return False

    def robot_nav_callback(self, pose):
        """
        callback from navigator node informing the control of arrival to set goal
        if area charging station: It requests to charge up
        if area is not battery: It requests to restore F-measure
        """

        #This rpose is the pose in the global tf frame
        rpose = pu.get_robot_pose(self.listener, self.robot_id)
        # self.total_traveled_distance += sp_distance.euclidean(self.prev_pose, rpose)
        self.prev_pose = rpose


        """
        Q: Where do we use the subscribed topic pose?
        Potential use: Get pose.point.x and pose.point.y, then check whether they are within the radius of target pose.
        If yes: If target pose is the charging station, request to charge; else, restore F-measure
        
        Above is a good solution to this one.
        """

        if self.robot_status == self.IN_MISSION:
            if self.is_charging_station(self.target_pose):
                self.update_robot_status(self.REQUEST2CHARGE)
                return
            #This is wrong. Any where in the map beyond the charging station radius becomes an area to restore F
            self.update_robot_status(self.RESTORING_F)

    ## Charge battery
    def request_charge(self):
        """
        UPNEXT (Mar 30): Needs updating

        Action request (to battery) to charge up battery
        :param max_charge:
        :return:
        """
        self.charge_battery_client.wait_for_server()
        goal = charge_batteryGoal() #TODO: Charge up battery
        goal.curr_batt_level = self.battery
        self.charge_battery_client.send_goal(goal, feedback_cb=self.charge_battery_feedback_cb)
        self.update_robot_status(self.CHARGING)
        self.charge_battery_client.wait_for_result()
        result = bool(self.charge_battery_client.get_result())
        if result is True:
            self.battery = self.max_battery #Battery charged to max level
            pu.log_msg('robot', self.robot_id, 'Fully-charged battery', self.debug_mode)
            self.update_robot_status(self.READY)

    def charge_battery_feedback_cb(self, msg):
        """
        Feedback for action request to charge up battery
        :param msg:
        :return:
        """
        if msg: pu.log_msg('robot', self.robot_id, 'Charging battery', self.debug_mode)

    ## Restore F-measure
    def restore_f_request_feedback_cb(self, msg):
        if msg: pu.log_msg('robot', self.robot_id, 'Restoring F...', self.debug_mode)


    def restore_f_request(self):
        """
        If result is True:
            self.update_robot_status(self.READY)
        :return:
        """

        pass

    def request_fmeasure(self, area, msg=True):
        """
        Service request for F-measure
        :param msg:
        :return:
        """
        rospy.wait_for_service("flevel_server_" + str(area))
        flevel_service = rospy.ServiceProxy("flevel_server_" + str(area), flevel)
        request = flevelRequest(msg)
        result = flevel_service(request)
        # print("Received from server current Fmeasure:", result.current_fmeasure)
        return result.current_fmeasure

    #DECISION-MAKING methods
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

        branches = list() #container for final branches up to depth k
        to_grow = list()  #container for branches still being grown/expanded
        nodes = self.areas[:]
        pu.log_msg('robot', self.robot_id, "Nodes prior cast: {}".format(nodes), self.debug_mode)
        #cast = type(scripts[0])
        nodes.append(cast(self.charging_station)) #append the charging station
        pu.log_msg('robot', self.robot_id, "Nodes: {}".format(nodes), self.debug_mode)

        #Start at the current location as the root node.
        fmeasures = dict() #F-measures of areas
        pu.log_msg('robot', self.robot_id, "Areas: {}".format(self.areas), self.debug_mode)
        for area in self.areas:
            fmeasures[area] = float(self.request_fmeasure(area))
        pu.log_msg('robot', self.robot_id, "Fmeasures: {}".format(fmeasures), self.debug_mode)
        k = 0

        #We need current location index

        path = [cast(self.curr_loc)]
        battery = self.battery
        cost = 0 #Initialize cost of path

        #Initial feasible battery level
        feasible_battery_consumption = self.consume_battery(start_area=cast(self.curr_loc), next_area=self.charging_station,
                                                            curr_measure=None, noise=noise)
        feasible_battery = battery - feasible_battery_consumption

        branch = (path, battery, fmeasures, cost, feasible_battery)
        to_grow.append(branch)

        #Succeeding decision steps:
        while k < dec_steps:
            pu.log_msg('robot', self.robot_id, "\nDec step: {}".format(k), self.debug_mode)
            consider_branches = to_grow[:]
            to_grow = list() #At the end of the iterations, to-grow will be empty while branches must be complete
            for branch in consider_branches:
                pu.log_msg('robot', self.robot_id, "Branch to grow: {}".format(branch), self.debug_mode)
                considered_growing= 0 #Indicators whether the branch has been considered for growing
                for i in range(len(nodes)):
                    # Hypothetical: What if we travel to this area, what will the consumed battery be and the updated F-fmeasures?
                    # Shall we prune this in the next decision step?
                    start_area = path[-1]
                    next_area = nodes[i]
                    path = branch[0][:]  # path at depth k
                    battery = branch[1]  # battery level at depth k
                    fmeasures = branch[2].copy()  # current fmeasures at depth k
                    cost = branch[3] # cost of path at depth k

                    #Need to compute duration to go back to charging station from next_area
                    #We would then store it as feasible battery which will be needed for breaking ties in sorting later
                    if next_area != cast(self.charging_station):
                        curr_measure = fmeasures[next_area]
                        tlapse_decay = get_time_given_decay(self.max_fmeasure, curr_measure, self.decay_rates_dict[next_area])
                        decayed_fmeasure = decay(self.decay_rates_dict[next_area], tlapse_decay, self.max_fmeasure) #Decayed measure of area if we travel there
                        duration = self.compute_duration(start_area=start_area, next_area=next_area, curr_measure=curr_measure, restoration=restoration, noise=noise) #Duration if we preserve area
                    else:
                        curr_measure, tlapse_decay, decayed_fmeasure = None, None, None
                        duration = self.compute_duration(start_area=start_area, next_area=self.charging_station, curr_measure=battery, restoration=restoration, noise=noise) #Duration if we charge up

                    #Battery consumption
                    battery_consumption = self.consume_battery(start_area=start_area, next_area=next_area, curr_measure=curr_measure, noise=noise)
                    battery_consumption_backto_charging_station = self.consume_battery(start_area=next_area, next_area=self.charging_station, curr_measure=None, noise=noise)
                    feasible_battery_consumption = battery_consumption + battery_consumption_backto_charging_station

                    pu.log_msg('robot', self.robot_id, "Next area: {}, Batt level: {}, Duration: {}, Batt consumption: {}, Decayed fmeasure: {}, TLapse decay: {}".format(next_area, battery, duration, battery_consumption, decayed_fmeasure, tlapse_decay), self.debug_mode)
                    # If branch is not to be pruned and length still less than dec_steps,
                    # then we continue to grow that branch

                    if (prune(battery, feasible_battery_consumption, decayed_fmeasure, self.fsafe) is False and (k < dec_steps)) or next_area == cast(self.charging_station): #TO-DO. Ok
                        # IDEA: If the next area is the charging station, we assume that the robot can always go back
                        path.append(next_area) #append next area as part of the path at depth k+1. #This is where the additional or overwriting happens. We need to make dummy list/container
                        if next_area != cast(self.charging_station):
                            battery -= battery_consumption #actual battery depleted at depth k+1
                        else:
                            battery = self.max_battery #actual battery restored to max value
                        feasible_battery = battery - feasible_battery_consumption  # battery available after taking into account battery to go back to charging station from current location
                        updated_fmeasures = self.adjust_fmeasures(fmeasures, next_area, duration) #F-measure of areas adjusted accordingly
                        cost += self.compute_cost_path(updated_fmeasures)
                        pu.log_msg('robot', self.robot_id,
                                   "Branch to grow appended: {}".format(path, battery, updated_fmeasures, cost, feasible_battery), self.debug_mode)
                        to_grow.append((path, battery, updated_fmeasures, cost, feasible_battery)) #Branch: (path, battery, updated_fmeasures, cost, feasible battery)
                        considered_growing += 1

                    #Else, we add that branch to branches (for return)
                    else:
                        # We need to make sure that the robot is not dead at the end of the operations, i.e.,
                        #   we check whether remaining feasible battery >= 0. If not, then this path ends dead, thus we don't append it
                        # Furthermore: If after iterating through possible scripts branch not considered for growing, and not yet in branches
                        if (is_feasible(battery, feasible_battery_consumption) is True) and (i == len(nodes)-1 and considered_growing == 0) and (branch not in branches):
                            pu.log_msg('robot', self.robot_id,
                                       "Branch appended to tree: {}".format(branch), self.debug_mode)
                            branches.append(branch)

                        #We need to check whether the branches are empty or not. We need to send the robot home.
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
        :param restoration: restore a measure (if not None) back to full measure
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
        distance += noise * distance  # distance + noise
        travel_time = (distance / self.robot_velocity)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time

        if next_area != self.charging_station:
            battery_consumed += self.batt_consumed_per_restored_f * (self.max_fmeasure - curr_measure)

        return battery_consumed

    def adjust_fmeasures(self, fmeasures, visit_area, duration):
        """
        Adjusts the F-measures of all areas. The visit area will be restored to max, while the other areas will decay for
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
                tlapse_decay = get_time_given_decay(self.max_fmeasure, fmeasures[area], self.decay_rates_dict[area]) + duration
                fmeasures[area] = decay(self.decay_rates_dict[area], tlapse_decay, self.max_fmeasure)

        return fmeasures

    def compute_cost_path(self, fmeasures):
        """
        Computes the cost, (i.e., the sum of losses) of the path
        :param fmeasures:
        :return:
        """

        """
        Steps:
        1. Computes the loss for each of the F-measure of the areas
        2. Sums up the losses to get the cost of the branch
        """
        #for area in list(values)

        cost = compute_cost_fmeasures(fmeasures, self.fsafe, self.fcrit)

        return cost

    def get_optimal_branch(self, tree):
        """
        Returns the optimal branch of the tree. This shall be the optimal decision path for the robot
        :param tree:
        :return:
        """

        """
        Steps:
        1. Sorts the branches of the tree by the accumulated cost, breaking ties by battery level, potentially feasible battery
        2. Returns the optimal path
        
        Nuances:
        What if the length is not of decision steps?
            > We resolve this in tree() where we decide whether to include in a branch
        """

        # Sort the branches of length k: the cost is key while the value is branch
        sorted_branches = sorted(tree, key = lambda x: (x[-2], -x[-1]))
        pu.log_msg('robot', self.robot_id,
                   "Branches sorted by cost:", self.debug_mode)
        for branch in sorted_branches:
            pu.log_msg('robot', self.robot_id,
                       "Branch: {}".format(branch), self.debug_mode)
        pu.log_msg('robot', self.robot_id,
                   "Optimal branch (branch info + cost): {}".format(sorted_branches[0]), self.debug_mode)
        optimal_path = sorted_branches[0][0]

        return optimal_path

if __name__ == '__main__':
    rd.seed(1234)
    #Robot('treebased_decision').run_operation()

    #TODO: Just make sure we can run the world and then communicate with the robot to move to one goal to other
