#!/usr/bin/env python
import math # use of pi.
import rospy # module for ROS APIs
import actionlib
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import LaserScan # message type for scan
from nav_msgs.msg import Odometry #message type for Odometry
from tf.transformations import euler_from_quaternion
from intermittent_monitoring.msg import visitAction, visitFeedback, visitResult
from intermittent_monitoring.msg import monitorAction, monitorGoal
from intermittent_monitoring.srv import location, locationResponse
from intermittent_monitoring.msg import charge_batteryAction, charge_batteryGoal
from intermittent_monitoring.srv import battery_level, battery_levelResponse
from intermittent_monitoring.srv import flevel, flevelRequest
from intermittent_monitoring.srv import area_assignment, area_assignmentResponse
from intermittent_monitoring.srv import bid_area, bid_areaResponse
from intermittent_monitoring.srv import robot_availability, robot_availabilityResponse
from intermittent_monitoring.loss_fcns import *
from intermittent_monitoring.cost_fcns import *
from std_msgs.msg import Float32

"""
Robot moves from one area to another

Inputs:
> odom
> laser
> waypoints

Process:
Moves to a waypoint. If that area is a charging station it charges up. 
If moitoring area it monitors (or raises) the F-measure

Outputs:
## Visit area
> Arrived at goal area requested (as action) by decision_making
> Action request to fmeasure to restore F-measure

## Battery levels (WORK)
> Service to decision_making for current battery level
> Action request to charging station to charge up battery
> Publishes battery levels

"""
def request_fmeasure(area, msg=True):
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

class Robot():
    def __init__(self, robot_no, motion_config, paths_config):
        """Constructor."""
        rospy.init_node('robot_'+str(robot_no))
        self.x, self.y, self.theta = None, None, None
        self.global_batt_consumed = 0.0
        self.global_duration = 0.0

        self.battery_consumption_restoring_fmeasure = motion_config['battery_consumption_restoring_fmeasure'] # = 0.25
        self.battery_consumption_moving_forward = motion_config['battery_consumption_moving_forward'] #= 0.10

        # Setting up publishers/subscribers.
        self._cmd_pub = rospy.Publisher(motion_config['DEFAULT_CMD_VEL_TOPIC'], Twist, queue_size=1)

        # Setting up subscriber.
        self._laser_sub = rospy.Subscriber(motion_config['DEFAULT_SCAN_TOPIC'], LaserScan, self._laser_callback, queue_size=1)
        self._odom_sub = rospy.Subscriber(motion_config['DEFAULT_ODOM_TOPIC'], Odometry, self._current_coordinates)

        # Other simulation configuration variables.
        self.linear_velocity = motion_config['LINEAR_VELOCITY'] # Constant linear velocity set.
        self.angular_velocity = motion_config['ANGULAR_VELOCITY'] # Constant angular velocity set.
        self._close_obstacle = False # Flag variable that is true if there is a close obstacle.
        self.frequency = motion_config['FREQUENCY']
        self.laser_angle_front = motion_config['LASER_ANGLE_FRONT']
        self.min_thresh_distance = motion_config['MIN_THRESHOLD_DISTANCE']

        # Paths, current location and battery level
        self.paths_dict = paths_config['paths_dict']
        self.location = paths_config['location']
        self.est_duration_dict = paths_config['dict_duration']
        self.est_batt_consumption = paths_config['dict_batt_consumption']
        self.areas = paths_config['areas']
        self.decay_rates = paths_config['decay_rates']
        self.max_battery = motion_config['MAX_BATTERY']
        self.battery = self.max_battery
        self.charge_battery = False #Starting battery is max. No need to charge battery just yet.

        # Publisher: Publishes battery levels per second
        self.battery_pub = rospy.Publisher("/battery_level_" + str(robot_no), Float32, queue_size=1)

        #Action server to visit an area: Could be an area or charging station
        #So: Depending on which the area is, we either monitor (request action to restore Fmeasure) or charge (request action to raise batt levels)
        #PO1: Here we place an ActionServer. If there is a request to visit a node, the robot then visits that node
        # [Consequential scripts in the program (that would change as well): fmeasure of areas, battery
        self.visit_action_server = actionlib.SimpleActionServer("visit_action_server_" + str(robot_no), visitAction, execute_cb=self.visit_area_cb, auto_start=False)
        self.visit_action_server.start()

        #Action Client: Monitor an area, raising its Fmeasure to max level
        #NOTE: If we have multiple areas, we will need to have multiple action clients to restore their F-measure
        # self.restore_fmeasure_action_client = actionlib.SimpleActionClient("restore_fmeasure_action_server", monitorAction)
        # self.restore_fmeasure_action_client

        self.restore_fmeasure_action_clients = dict()
        for area in self.areas:
            self.restore_fmeasure_action_clients[str(area)] = actionlib.SimpleActionClient('restore_fmeasure_action_server_' + str(area), monitorAction)

        # Action client: Request charging station to charge up battery
        self.charge_battery_client = actionlib.SimpleActionClient("charge_battery_server", charge_batteryAction)

        #Service server: Provide location information
        #P01: We need as a server of our location. This then triggers whether to monitor an area.
        #PO2: We have a monitor command/request for fmeasure to raise the fmeasure gradually. We can devise a function on
        self.location_server = rospy.Service("location_server_" + str(robot_no), location, self.report_location_cb)

        #Service server: Provide battery level info (to decision_making) upon request
        self.battery_server = rospy.Service("battery_server_" + str(robot_no), battery_level, self.report_battery_cb)

        #AUCTION variables
        self.bid_availability = True #Availability of robot for bidding

        ##Service server: Bid for a given area
        self.bid_area_server = rospy.Service("bid_area_server_" + str(robot_no), bid_area, self.bid_for_area_cb)

        ##Service server: Assignment of area after bidding if awarded
        self.area_assignment_server = rospy.Service("area_assignment_server_" + str(robot_no), area_assignment, self.area_assignment_cb)
        self.assignment = None

        ##Service server: Availability
        self.available = True
        self.robot_availability_server = rospy.Service("robot_availability_server_" + str(robot_no), robot_availability, self.robot_availability_cb)

    #AUCTION BID METHODS
    def report_location_cb(self, msg):
        """
        Callback for reporting location as a Service Server
        :param msg:
        :return:
        """
        if bool(msg.location_request) is True:
            return locationResponse(self.location)

    def compute_loss(self, area):
        """
        Computes loss of a given area
        :return:
        """
        fmeasure = request_fmeasure(area)
        if fmeasure != 'None': #There is F-measure data
            curr_fmeasure = float(fmeasure)
            est_duration = self.est_duration_dict[self.location, area]

            #Cool idea: These decay rates are given. In reality, we can have the center provide for initial values, potentially prior information.
            #The robots then periodically share information on gathered data. They then share this data to the center.
            #The center then in turn updates its decay function and broadcasts to the robots.
            loss = compute_loss(curr_fmeasure, self.decay_rates[str(area)], list(self.decay_rates.values()), est_duration)
        else: #No F-measure data, no loss [For now, we will not bid].
            loss = 0.0
        return loss

    def compute_cost(self, area):
        """
        Computes cost of a given area
        :param area:
        :return:
        """
        return 0.0 #For now we set to 0 to avoid artifacts.

    def is_feasible(self, area):
        """
        Determines whether an area is feasible to visit given current battery level
        :param area:
        :return:
        """
        if self.battery >= self.est_batt_consumption[self.location, area]:
            return True
        else:
            return False

    def SS_component(self, area):
        """
        Computes SS component: is_feasible*(loss/cost)
        :param area:
        :return:
        """
        loss = self.compute_loss(area)
        cost = self.compute_cost(area)
        is_feasible = self.is_feasible(area)
        print("Area: {}. Loss: {}. Cost: {}. Is feasible: {}".format(area, loss, cost, is_feasible))
        component = is_feasible * (loss/(cost+1))
        return component

    ###AUCTION BID METHODS PART 2:
    """
    Part 1. Bidding: Service server to auctioneer for bid given an area
        > Robot bids if there is at least one non-zero component.
        > Otherwise, it doesn't return any bid and goes back to the charging station
    Part 2. Awarding of bid and Visiting the area
        > Robot becomes unavailable and area becomes assigned temporarily until accomplished
        PO1: Client to Auctioneer for whether awarded the bid
            > Possible: Awarded. Not Awarded. Not available
            > If awarded visits the area
        PO2: Server to Auctioneer for awarding of bid
            > Receives an awarded bid
            > Acknowledges as receipt of info
            > Visits the area after that
    Part 3. Requests for registry into the auction
        > PO1: (This could be part 1 actually) The robot (a client) only requests for an area to bid if it is available
        > PO2: Client to Auctioneer letting it know that service has been rendered
        > PO3: Area itself has an assignment status 
    
    """

    def visit_area(self, area):
        """
        This method asks the robot to visit bid (or awarded) area. This is inserted in the run_operation as a while loop.
        :param area:
        :return:
        """
        path = self.paths_dict["(" + str(self.location) + ", " + str(area) + ")"]
        self.available = False
        #If current location is not the target area, we traverse to that area
        if self.location != area:
            for coord in path:
                self.move_to_coords(coord)  # Traverse to the next coordinates

        request_result = True
        self.location = area  # arrived at the goal
        print("Arrived area:", self.location)

        # Insert here action request to either raise fmeasure if location is area or charge if charging station
        if self.location != 0:
            request_result = bool(self.raise_fmeasure_action_request(area, goal_fmeasure=100))
            # To-insert: Battery consumption. DONE. Inserted inside raise F-measure
        else:
            request_result = bool(self.charge_battery_request())

        print("GLOBAL DURATION:", self.global_duration)
        print("GLOBAL BATT CONSUMED:", self.global_batt_consumed)
        print("Request result:", request_result)

        if request_result is True:
            self.global_duration = 0.0  # Reset for the next action request
            self.global_batt_consumed = 0.0
            self.available = True
            print("Robot available:", self.available)

    def robot_availability_cb(self, msg):
        """
        Callback for robot availability
        :param msg:
        :return:
        """
        print("Callback request robot availability:", self.available)
        return robot_availabilityResponse(self.available)

    def area_assignment_cb(self, msg):
        """
        Callback for area assignment
        :param msg:
        :return:
        """

        self.assignment = msg.area_assigned
        print("Assigned area cb", type(msg.area_assigned), msg.area_assigned)
        self.available = False #Update availability after receiving assigned (or awarded) area
        return area_assignmentResponse(True)

    #TO-COMPOSE: BID for an area
    def bid_for_area_cb(self, msg):
        """
        PO: Compute for SS components for all of the areas.
        If all of the SS components equal 0, bid 'None' and go back to charging station
        Else: bid for that area, which includes 0.
        :msg: area to bid for
        :return:
        """
        print("Bid for area:", msg.area)
        SS_components = dict()
        for area in self.areas:
            SS_components[str(area)] = self.SS_component(area)
        if sum(list(SS_components.values())) < 0: #Note all components: [-inf, 0]
            bid = SS_components[str(msg.area)]
        else: #Robot does not have enough battery to raise a bid; Or no bid at all, no loss; or No availablee F-measure data
            self.charge_battery = True
            bid = 0.0 #No bid
        print("Bid:", bid)
        return bid

    ##PART 1: Callbacks for Server Service/Action Server/Publisher
    def _laser_callback(self, msg):
        """Processing of laser message."""
        # Access to the index of the measurement in front of the motion.
        # NOTE: assumption: the one at angle 0 corresponds to the front.
        i = int((self.laser_angle_front - msg.angle_min) / msg.angle_increment)
        if msg.ranges[i] <= self.min_thresh_distance:
            self._close_obstacle = True
        else:
            self._close_obstacle = False

    def _current_coordinates(self, msg):
        "Sets the current coordinates of the motion"
        self.x, self.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, self.theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

    def publish_battery(self):
        """
        Publish battery levels as a topic
        :return:
        """
        self.battery_pub.publish(self.battery)

    def visit_area_cb(self, goal):
        """
        Action server call back: Visits an area as requested by action client (decision_making), and then charges or restores
        F-measure depending on whether the area is to be monitored or a charging station.
        :param goal (int8): area
        :return:
        """
        success = True
        visit_feedback = visitFeedback()
        result = visitResult()
        rate = rospy.Rate(1)

        path = self.paths_dict["(" + str(self.location) + ", " + str(goal.area) + ")"]

        # print("Current coordinates:", self.x, self.y, self.theta)

        for coord in path:
            if self.visit_action_server.is_preempt_requested():
                success = False
                break
            self.move_to_coords(coord) #Traverse to the next coordinates

            #To-insert: Battery consumption. DONE. Inserted inside the components of move_to_coords

            visit_feedback.current_coords = "Current coords: " + str(self.x) + ", " + str(self.y)
            self.visit_action_server.publish_feedback(visit_feedback)
            rate.sleep()

        if success is True:
            self.location = goal.area #arrived at the goal
            print("Goal area:", self.location)

            #Insert here action request to either raise fmeasure if location is area or charge if charging station
            if self.location != 0:
                request_result = bool(self.raise_fmeasure_action_request(goal.area, goal_fmeasure=100))
                #To-insert: Battery consumption. DONE. Inserted inside raise F-measure
            else:
                request_result = bool(self.charge_battery_request())

            print("REQUEST RESULT:", request_result)
            print("GLOBAL DURATION:", self.global_duration)
            print("GLOBAL BATT CONSUMED:", self.global_batt_consumed)

            if request_result is True:
                result.arrived_goal = True
                self.global_duration = 0.0 #Reset for the next action request
                self.global_batt_consumed = 0.0
            else:
                result.arrived_goal = False
            self.visit_action_server.set_succeeded(result)

    def restore_fmeasure_action_feedback_cb(self, msg):
        """
        Feedback for action request to restore F-measure
        :param msg:
        :return:
        """
        print("Monitor feedback received:", msg)

    def raise_fmeasure_action_request(self, area, goal_fmeasure):
        """
        Action request to raise F-measure of an area.
        :param goal_fmeasure:
        :return:
        """

        #Request for current F-measure
        curr_fmeasure = request_fmeasure(area)

        self.restore_fmeasure_action_clients[str(area)].wait_for_server()
        goal = monitorGoal()
        goal.max_fmeasure = goal_fmeasure

        #Identify which area to restore F-measure
        self.restore_fmeasure_action_clients[str(area)].send_goal(goal, feedback_cb=self.restore_fmeasure_action_feedback_cb)
        self.restore_fmeasure_action_clients[str(area)].wait_for_result()
        result = bool(self.restore_fmeasure_action_clients[str(area)].get_result())
        print("Restore Area {} Fmeasure result {}:".format(area, result))
        if result is True:
            #BATTERY DEPLETION
            self.battery -= self.battery_consumption_restoring_fmeasure * (goal_fmeasure - curr_fmeasure) #function of the current F-measure and max F
            self.global_batt_consumed += self.battery_consumption_restoring_fmeasure
            # print("Battery depleted from restoring F-measure:", self.battery)

        return result

    def charge_battery_feedback_cb(self, msg):
        """
        Feedback for action request to charge up battery
        :param msg:
        :return:
        """
        print("Charging feedback received:", msg)

    def charge_battery_request(self):
        """
        Action request (to charging_station) to charge up battery
        :param max_charge:
        :return:
        """
        self.charge_battery_client.wait_for_server()
        goal = charge_batteryGoal()
        goal.curr_batt_level = self.battery
        self.charge_battery_client.send_goal(goal, feedback_cb=self.charge_battery_feedback_cb)
        self.charge_battery_client.wait_for_result()
        result = bool(self.charge_battery_client.get_result())
        print("Charging result:", result)
        if result is True:
            self.battery = self.max_battery #Battery charged to max level
            self.charge_battery = False #No need to charge battery because fully charged
            print("Fully-charged battery:", self.battery)

        return result

#TO-DO: (DONE) Battery depletes as we move around plus monitor an area
#Another: If area that we are moving toward is an area, we request to restore F-measure
    #If charging station, we request to charge up battery

    def report_location_cb(self, msg):
        """
        Callback for reporting location as a Service Server
        :param msg:
        :return:
        """
        if bool(msg.location_request) is True:
            return locationResponse(self.location)

    def report_battery_cb(self, msg):
        """
        Callback for reporting current battery level as a Service Server
        :param msg:
        :return:
        """
        if bool(msg.batt_level_request) is True:
            return battery_levelResponse(self.battery)

    ##PART 2: METHODS FOR ROBOT MOTION
    def move_forward(self, distance):
        """Function to move_forward for a given distance."""
        # Rate at which to operate the while loop.
        rate = rospy.Rate(self.frequency)

        # Setting velocities.
        twist_msg = Twist()
        twist_msg.linear.x = self.linear_velocity
        start_time = rospy.get_rostime()
        duration = rospy.Duration(distance/twist_msg.linear.x)
        # print("Duration move_forward", type(duration), duration.secs)
        self.global_duration += duration.secs
        # Loop.
        while not rospy.is_shutdown():
            # Check if traveled of given distance based on time.
            if rospy.get_rostime() - start_time >= duration:
                break

            # Publish message.
            if self._close_obstacle:
                self.stop()
            else:
                self._cmd_pub.publish(twist_msg)

            #BATTERY DEPLETION
            self.battery -= self.battery_consumption_moving_forward
            self.global_batt_consumed += self.battery_consumption_moving_forward

            # Sleep to keep the set publishing frequency.
            rate.sleep()

        # Traveled the required distance, stop.
        self.stop()

    def goal_theta(self, goal_coords):
        """
        Computes the goal theta to steer toward from current coordinates to the goal coordinates.
        We can think of a right triangle from current coordinates to the goal coordinates.
        We then compute the theta (angle) to steer toward by arc tangent of the opposite and adjacent sides of the angle.
        :param goal_coordinates: (x, y) tuple
        :return: theta_goal in radians
        """
        theta_goal = math.atan2((goal_coords[1] - self.y), (goal_coords[0] - self.x)) #theta = arc_tan[(y'-y)/(x'-x)]

        return theta_goal

    def measure_distance(self, goal_coords):
        """
        Measures the distance (or hypotenuse) between two points. This shall be the distance to move forward to after correcting the bearing (or theta)
        of the motion
        :param goal_coords:
        :return:
        """
        hypotenuse = math.sqrt((goal_coords[1] - self.y)**2 + (goal_coords[0] - self.x)**2)
        return hypotenuse

    def correct_bearing(self, goal_theta):
        """
        Rotate in place the motion of rotation_angle (rad) based on fixed velocity.
        Here the orientation of the rotation can be counter-clockwise or clockwise
        """
        twist_msg = Twist()

        diff_theta = self.theta - goal_theta
        # print("Current theta:", self.theta, "Goal theta:", goal_theta, "Adjustment:", diff_theta)
        if diff_theta >= 0:
            # print("Rotate clockwise")
            twist_msg.angular.z = -self.angular_velocity #Clockwise
        else:
            # print("Rotate counter")
            twist_msg.angular.z = self.angular_velocity #Counter-clockwise

        duration = abs(diff_theta) / self.angular_velocity
        # print("Duration correct bearing:", type(duration), duration)
        self.global_duration += duration

        start_time = rospy.get_rostime()
        rate = rospy.Rate(self.frequency)
        while not rospy.is_shutdown():
            # Check if done
            if rospy.get_rostime() - start_time >= rospy.Duration(duration):
                break

            # Publish message.
            self._cmd_pub.publish(twist_msg)

            # BATTERY DEPLETION
            self.battery -= 0.10
            self.global_batt_consumed += 0.10

            # Sleep to keep the set frequency.
            rate.sleep()

        # Rotated the required angle, stop.
        self.stop()

    def move_to_coords(self, goal_coords):
        """
        Moves to target coordinates
        :return:
        """
        ##Measure the angle to rotate to
        ###We use tangent
        goal_angle = self.goal_theta(goal_coords)
        self.correct_bearing(goal_angle)

        ##Move forward
        ###Traverse that distance
        distance = self.measure_distance(goal_coords)
        self.move_forward(distance)

    def stop(self):
        """Stop the motion."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def get_coordinates(self, vertex):
        "Gets the coordinates of the vertex"
        # print("Vertex: {}. Coordinates: ({}, {})".format(vertex, self.x, self.y))
        return self.x, self.y

    def distance_bet_vertices(self, vertex1, vertex2):
        """Measures the L2-distance between two vertices"""
        distance = 0
        for p in range(len(vertex1)):
            distance += (vertex1[p] - vertex2[p])**2
        return distance

    # def visit_area(self, area):
    #     """
    #     From current area we visit a next area
    #     :param name:
    #     :return:
    #     """
    #     path = self.paths_dict["(" + str(self.location) + ", " + str(area) + ")"]
    #
    #     for coord in path:
    #         # print("Current coordinates: ", self.x, self.y, self.theta)
    #         # print("Next waypoint: ", coord)
    #         self.move_to_coords(coord)
    #     self.location = area

    def run_operation(self):
        # 1st. initialization of node.
        #rospy.init_node(node)

        # Sleep for a few seconds to wait for the registration.
        rospy.sleep(5)

        # If interrupted, send a stop command.
        # rospy.on_shutdown(self.stop())

        # Robot traverses an edge, rotates by the next vertex, repeating this process
        # until the entire polygon has been traversed.

        has_init_odom_data = False
        while has_init_odom_data is False:
            if (self.x is not None) and (self.y is not None) and (self.theta is not None):
                print("Initial coordinates: ", self.x, self.y, self.theta)
                has_init_odom_data = True
            else:
                print("No odom data registered. Please wait...")

        rate = rospy.Rate(1)
        try:
            print("We entered try")
            self.publish_battery()  # Note: The publishing of battery is not per second, rather per operation (or iteration of operation)
            while not rospy.is_shutdown():
                """
                Here we insert if there has been a notice.
                To be exact, we check whether there is an assignment in self.assignment
                If there is we visit that area
                """
                print("\nRobot stats:")
                print("Battery:", self.battery)
                print("Charge battery?", self.charge_battery)
                print("Assignment:", self.assignment)
                if self.charge_battery is False:
                    if self.assignment is not None: #There is an assigned (or awarded) area
                        print("Visiting assigned area:", self.assignment)
                        self.visit_area(self.assignment)
                        self.assignment = None
                    else: #No assigned area. Just waiting for next bid area
                        print("Location: {}. Just waiting for next bid area. Still has battery...".format(self.location))
                        pass
                else: #Not enough battery, so charge up!
                    print("Back to charging station...")
                    self.visit_area(0)
                    # self.assignment = None
                self.publish_battery() #Post-iteration battery
                rate.sleep()

        except rospy.ROSInterruptException:
            rospy.logerr("ROS node interrupted.")

