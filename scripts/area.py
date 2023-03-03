#!/usr/bin/env python

"""
F-measure

Process:
1. F-measure that gradually drops given a decay function

Inputs:
1. Robot monitors the area (action request) which then would raise the F-level
2. Request from decision_making about current F-level

Output:
1. Notify action request from robot after F-measure is raised to max level
2. Respond to request from decision_making on current F-level

Thus:
1. Action server for the robot, which asks to raise F-level
2. Server to decision_making about current F-level
"""
import math
import rospy
import actionlib
from int_preservation.msg import monitorAction, monitorFeedback, monitorResult
from int_preservation.srv import flevel, flevelResponse
from int_preservation.srv import assignment_notice, assignment_noticeResponse
from std_msgs.msg import Float32, Int8
import project_utils as pu


# area, max_fmeasure, decay_rate, restoration, t_operation

self.sub_fmeasure_dict[area] = rospy.Subscriber('/fmeasure_' + str(area), Float32, self.store_fmeasure_cb, area)

class Area():
    IDLE = 10
    IN_MISSION = 12
    RESTORING_F = 41
    RESTORED_F = 42
    def __init__(self, node_name, robot_operating):
        rospy.init_node(node_name)
        self.area = rospy.get_param("~area_id")
        self.decay_rate = rospy.get_param("/decay_rate")
        self.t_operation = rospy.get_param("/t_operation") #total duration of the operation
        self.max_fmeasure = rospy.get_param("/max_fmeasure")
        self.fmeasure = self.max_fmeasure #initialize at max fmeasure
        self.restoration = rospy.get_param("/restoration")
        self.restore_delay = int(math.ceil(self.restoration * (self.max_fmeasure-self.fmeasure))) #delay in restoring the F-measure to max level
        self.request_restore_status = None
        self.global_decay_fmeasure = 0

        # Fmeasure publisher
        self.fmeasure_pub = rospy.Publisher("/fmeasure_" + str(self.area), Float32, queue_size=1)

        #Action server: Raise Fmeasure
        self.restore_fmeasure_action_server = actionlib.SimpleActionServer("restore_fmeasure_action_server_" + str(self.area), monitorAction, execute_cb=self.raise_fmeasure_cb, auto_start=False)
        self.restore_fmeasure_action_server.start()

        #Service server: Fmeasure
        self.fmeasure_server = rospy.Service("flevel_server_" + str(self.area), flevel, self.report_flevel_cb)

        #Subscriber: Robot status
        self.robot_status_sub = rospy.Subscriber("/robot_" + robot_operating + "/robot_status", Int8, self.robot_status_cb)

        self.status = self.IDLE

    def run_operation(self, freq_hz=1):
        t = 0
        # PO (for finite?):for duration in range(self.t_operation): #We use this if we want t_operation to be finite
        # Need to ensure that the decay is per second

        rate = rospy.Rate(freq_hz)
        while not rospy.is_shutdown():
            self.publish_fmeasure()
            print("\nArea status:")
            print("Request restore status:", self.request_restore_status)
            print("Assigned to bidder:", self.assigned_to_bidder)

            """
            Statuses:
            1. IDLE
                > If accept info that robot is in mission: move to next state, start decaying.
                > If robot is in idle: Status should also be idle
            2. Decaying
                > STarts decaying and publishing F
            3. Restoring
                > Action server to restore F back to max measure
            """

            if self.area_status == self.IDLE:
                pass

            elif self.area_status == self.IN_MISSION:
                self.decay(t)
                t += 1

            elif self.area_status == self.RESTORING:
                pass #not decaying because being restored

            elif self.request_restore_status == self.RESTORED:
                #Restore parameters
                t = 0
                self.global_decay_fmeasure = 0
            rate.sleep()

    def robot_status_cb(self, robot_status):
        """
        Call for robot status
        :return:
        """
        if robot_status == self.IDLE:
            self.area_status == self.IDLE
        # elif robot_status == self.


    def publish_fmeasure(self):
        """
        Publishes F-measure as a topic
        :return:
        """
        self.fmeasure_pub.publish(self.fmeasure)

    def raise_fmeasure_cb(self, goal):
        """
        Callback as action server for restoring F-measure upon request of action client
        :param goal:
        :return:
        """
        success = True
        monitor_feedback = monitorFeedback()
        monitor_result = monitorResult()
        rate = rospy.Rate(1)
        self.area_status = self.RESTORING_F

        for i in range(self.restore_delay):
            if self.restore_fmeasure_action_server.is_preempt_requested():
                success = False
                break
            monitor_feedback.current_fmeasure = 'Restoring F-measure...'
            self.restore_fmeasure_action_server.publish_feedback(monitor_feedback)
            rate.sleep()

        self.fmeasure = goal.max_fmeasure
        monitor_result.raised_max = True

        if success:
            self.area_status = self.RESTORED_F
            self.restore_fmeasure_action_server.set_succeeded(monitor_result)

    def report_flevel_cb(self, msg):
        """
        Callback as Service Server for F-measure
        :param msg:
        :return:
        """
        if (bool(msg.fmeasure_request) is True) and (self.assigned_to_bidder is False):
            print("Reporting FMeasure", str(self.fmeasure))
            return flevelResponse(str(self.fmeasure))
        else:
            print("No Fmeasure reported")
            return 'None'

    def decay(self, t):
        """
        Decay function
        :param t:
        :return:
        """
        decayed_f = self.max_fmeasure*(1 - self.decay_rate)**t
        self.global_decay_fmeasure += self.fmeasure - decayed_f # Global decay F-measure
        self.fmeasure = decayed_f