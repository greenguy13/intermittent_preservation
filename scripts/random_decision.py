#!/usr/bin/env python

"""
Randomized decision making
    1. Collect feasible areas
    2. Randomly select from those feasible areas
        > With or without replacement?
    3. Monitor that area
"""
"""
Updated Random decision making
    1. Collect feasible areas by appropriate methods
        > Subscribe to make plan
        > Come up with the distance matrix
    2. Among the feasible areas, randomly pick the next area (perhaps apply some seed)
    3. Preserve that area
    4. Question: When will the robot charge up? When no area is feasible
"""
import pickle
import numpy as np
import random as rd
import rospy
import actionlib
from loss_fcns import *
from pruning import *
import project_utils as pu
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8, Float32
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from status import areaStatus, battStatus, robotStatus
from reset_simulation import *


INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3 #GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html
SHUTDOWN_CODE = 99

class Robot():
    def __init__(self, node_name):
        """

        :param node_name:
        :param areas:
        :param est_duration_matrix:
        :param est_batt_consumption_matrix:
        """
        rospy.init_node(node_name)

        # Parameters
        self.robot_id = rospy.get_param("~robot_id")

        self.debug_mode = rospy.get_param("/debug_mode")
        self.robot_velocity = rospy.get_param("/robot_velocity")  # Linear velocity of robot; we assume linear and angular are relatively equal
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery")  # Max battery
        self.battery_reserve = rospy.get_param("/battery_reserve")  # Battery reserve
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = rospy.get_param("/batt_consumed_per_time")  # (travel, restoration)
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.nareas = rospy.get_param("/nareas")  # Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i + 1) for i in range(self.nareas)]  # list of int area IDs
        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save") #Whether to save data

        # Initialize variables
        charging_station_coords = rospy.get_param("~initial_pose_x"), rospy.get_param("~initial_pose_y")  # rospy.get_param("/charging_station_coords")
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        self.sampled_nodes_poses = [charging_pose_stamped]  # list container for sampled nodes of type PoseStamped

        # Pickle load the sampled area poses
        with open(rospy.get_param("/file_sampled_areas") + '.pkl', 'rb') as f:
            sampled_areas_coords = pickle.load(f)
        for area_coords in sampled_areas_coords['n{}_p{}'.format(self.nareas, rospy.get_param("/placement"))]:
            pose_stamped = pu.convert_coords_to_PoseStamped(area_coords)
            self.sampled_nodes_poses.append(pose_stamped)
        # rd.seed(100*self.nareas + 10*rospy.get_param("/placement"))

        self.charging_station = 0
        self.curr_loc = self.charging_station  # Initial location robot is the charging station
        self.battery = self.max_battery  # Initialize battery at max, then gets updated by subscribed battery topic
        self.chosen_area = None
        self.dist_matrix = None
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True
        self.curr_fmeasures = dict()  # container of current F-measure of areas
        self.total_dist_travelled = 0  # total distance travelled
        self.x, self.y = 0.0, 0.0  # Initialize robot pose

        # Containers for recorded data
        self.decisions_made, self.decisions_accomplished, self.status_history = [], [], []

        # We sum this up
        self.environment_status = dict()
        for node in range(self.nareas + 1):
            self.environment_status[node] = None

        # Publishers/Subscribers
        # Service request to move_base to get plan : make_Plan
        server = '/robot_' + str(self.robot_id) + '/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        rospy.Subscriber('/robot_{}/battery_status'.format(self.robot_id), Int8, self.battery_status_cb)
        rospy.Subscriber('/robot_{}/battery'.format(self.robot_id), Float32, self.battery_level_cb)

        for area in self.areas:
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb, area)
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb, area)

        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)

        rospy.Subscriber('/robot_{}/odom'.format(self.robot_id), Odometry, self.distance_travelled_cb, queue_size=1)

        # Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base', MoveBaseAction)
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
        self.robot_goal_client.send_goal(movebase_goal, done_cb=action_goal_cb,
                                         active_cb=self.action_send_active_cb)

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
            self.chosen_area = None #Reset the chosen area since we already reached there

    #DECISION-MAKING
    def choose_area_randomly(self, feasible_areas):
        """
        Chooses among feasible areas that have fallen below safe zone randomly
        :param: feasible_areas (list): list of feasible areas
        :return:
        """
        pick = 0
        #If there are feasible areas that are in caution zone in the next decision step, pick randomly
        choices = np.array(feasible_areas)
        condition = choices <= math.ceil(0.25 * self.nareas)
        if len(choices[condition]) > 0:
            #pick randomly by a coin flip
            flip = rd.random()
            if flip < 0.95:
                pick = rd.choice(choices[condition])
            else:
                pick = rd.choice(feasible_areas)
        return pick

    #NOTE: We can use compute_loss to find out whether an area will fall below the safe zone in the next decision step.

    def compute_duration(self, start_area, next_area, curr_measure, restoration, noise):
        """
        Computes (time) duration of operation, which includes travelling distance plus restoration, if any
        Furthermore, we include the battery consumed
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
        distance += noise * distance
        travel_time = (distance / self.robot_velocity)
        battery_consumed = self.batt_consumed_per_travel_time * travel_time

        if next_area != self.charging_station:
            battery_consumed += self.batt_consumed_per_restored_f * (self.max_fmeasure - curr_measure)

        return battery_consumed


    def find_feasible_areas(self, curr_robot_loc, curr_batt_level):
        """
        Find feasible areas
        :return:
        """
        feasible_areas = list()
        nodes = [self.charging_station]
        nodes.extend(self.areas[:])
        for area in nodes:
            curr_measure = None
            if area != self.charging_station:
                curr_measure = self.curr_fmeasures[area]
            #Feasible battery
            batt_consumption = self.consume_battery(start_area=curr_robot_loc, next_area=area, curr_measure=curr_measure, noise=self.noise)
            batt_consumption += self.consume_battery(start_area=area, next_area=self.charging_station, curr_measure=None, noise=self.noise)

            if is_feasible(curr_batt_level, batt_consumption, self.battery_reserve):
                feasible_areas.append(area)

        return feasible_areas

    def run_operation(self, filename):
        """
        Among the feasible areas not yet assigned, pick randomly the next area to monitor
        :return:
        """

        if self.robot_id == 0:
            rate = rospy.Rate(1)
            while len(self.sampled_nodes_poses) != self.nareas+1:
                self.debug("Insufficient data. Sampled nodes poses: {}/{}. Area status None: {}".format(len(self.sampled_nodes_poses), self.nareas + 1, sum(self.environment_status.values())))
                rate.sleep()

            self.debug("Sufficent data. Sampled nodes poses: {}".format(self.sampled_nodes_poses))
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
                    self.think_decisions()
                    self.update_robot_status(robotStatus.IN_MISSION)

                elif self.robot_status == robotStatus.IN_MISSION.value:
                    self.debug('Robot in mission')
                    if self.available:
                        self.commence_mission()

                elif self.robot_status == robotStatus.CHARGING.value:
                    self.debug('Waiting for battery to charge up')

                elif self.robot_status == robotStatus.RESTORING_F.value:
                    self.debug('Restoring F-measure')

                t += 1
                rate.sleep()

            #Store data and shut down
            self.update_robot_status(robotStatus.SHUTDOWN)
            self.robot_status_pub.publish(self.robot_status)
            self.status_history.append(self.robot_status)

            if self.save:
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
        self.chosen_area = self.charging_station #Default is the charging station
        feasible_areas = self.find_feasible_areas(self.curr_loc, self.battery)
        if len(feasible_areas)>0:
            self.chosen_area = self.choose_area_randomly(feasible_areas)
            self.debug("Feasible areas: {}. Chosen area: {}".format(feasible_areas, self.chosen_area))

    def commence_mission(self):
        """
        Commences mission
        :return:
        """
        if self.send2_next_area() == 0:
            self.update_robot_status(robotStatus.IDLE)

    def send2_next_area(self):
        """
        Sends the robot to the randomly chosen area
        :return:
        """
        if self.chosen_area is not None:
            self.mission_area = self.chosen_area
            self.mission_area_pub.publish(self.mission_area)
            self.debug('Heading to {}'.format(self.mission_area))
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
        self.environment_status[0] = msg.data
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

    def area_fmeasure_cb(self, msg, area_id):
        """
        Updates fmeasure of area
        :param msg:
        :param area_id:
        :return:
        """
        self.curr_fmeasures[area_id] = msg.data

    def distance_travelled_cb(self, msg):
        #Updates total distance travelled
        #Sets curr robot pose
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.total_dist_travelled += math.dist((self.x, self.y), (x, y))
        self.x, self.y = x, y

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)

if __name__ == '__main__':
    os.chdir('/home/ameldocena/.ros/int_preservation/results')
    filename = rospy.get_param('/file_data_dump')
    Robot('random_decision').run_operation(filename)


