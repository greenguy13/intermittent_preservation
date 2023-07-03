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

Relay of information:
1. Action server for the robot, which asks to raise F-level
2. Server to decision_making about current F-level
"""

"""
UPNEXT:
1. Robot goes to area's location
2. Measures F
3. Asks to restore F
"""
from enum import Enum
import math
import rospy
import actionlib
from std_msgs.msg import Float32, Int8
from intermittent_monitoring.msg import monitorAction, monitorFeedback, monitorResult
import project_utils as pu


# area, max_fmeasure, decay_rate, restoration, t_operation

# self.sub_fmeasure_dict[area] = rospy.Subscriber('/fmeasure_' + str(area), Float32, self.store_fmeasure_cb, area)

class robotStatus(Enum):
    IDLE = 10
    READY = 11
    THINKING = 12
    IN_MISSION = 20
    CHARGING = 30
    RESTORING_F = 40

class Area():
    IDLE = 0
    DECAYING = 1
    RESTORING_F = 11
    RESTORED_F = 12

    def __init__(self):
        rospy.init_node('area', anonymous=True)
        self.area = rospy.get_param("~area_id")
        self.robot_id = rospy.get_param("~robot_id")
        self.decay_rate = float(rospy.get_param("~decay_rate"))

        #We can write these params as general param
        #Change/evolution in decay rate; po in the general parameter
        self.t_operation = rospy.get_param("~t_operation") #total duration of the operation
        self.max_fmeasure = rospy.get_param("~max_fmeasure")
        self.fmeasure = self.max_fmeasure #initialize at max fmeasure
        self.restoration = rospy.get_param("~restoration")

        pu.log_msg('area', self.area, 'Area {}. decay rate: {}. t_operation: {}. max_f: {}. restore: {}. robot id: {}'.format(self.area, self.decay_rate, self.t_operation, self.max_fmeasure, self.restoration, self.robot_id), 1)

        # Fmeasure publisher
        self.fmeasure_pub = rospy.Publisher("/area_{}/fmeasure".format(self.area), Float32, queue_size=1)

        # Server for pausing/commencing simulation
        # TODO P5: Make sure that the topic is robot_status
        rospy.Subscriber('/robot_{}/robot_status'.format(self.robot_id), Int8, self.robot_status_cb)

        # TODO P5: Continue operation subscriber: shuts down node if received message is True
        # rospy.Subscriber("/continue_operation", Bool, self.continue_operation_cb)

        # Action server: Raise Fmeasure
        self.restore_fmeasure_action_server = actionlib.SimpleActionServer("/restore_fmeasure_action_server_" + str(self.area), monitorAction, execute_cb=self.raise_fmeasure_cb, auto_start=False)
        self.restore_fmeasure_action_server.start()

        self.status = self.IDLE
        self.restore_request = False

    def restore_delay(self):
        """
        Delay in restoring F-measure back to max level
        :return:
        """
        delay = int(math.ceil(self.restoration * (self.max_fmeasure-self.fmeasure)))
        return delay

    def robot_status_cb(self, msg):
        """
        TODO P5: Callback for robot status. If robot status is thinking decisions, we pause decay simulation
        :return:
        """

        if (msg == robotStatus.IDLE) or (msg == robotStatus.READY) or (msg == robotStatus.THINKING):
            self.status = self.IDLE
        elif msg == robotStatus.IN_MISSION or (msg == robotStatus.RESTORING_F and self.restore_request is False) \
                or msg == robotStatus.CHARGING:
            self.status = self.DECAYING

    def raise_fmeasure_cb(self, goal):
        """
        Callback as action server for restoring F-measure upon request of action client, (which is the robot)
        :param goal:
        :return:
        """
        success = True
        monitor_feedback = monitorFeedback()
        monitor_result = monitorResult()
        rate = rospy.Rate(1)

        self.restore_request = True
        self.status = self.RESTORING_F
        delay = self.restore_delay()
        for i in range(delay):
            if self.restore_fmeasure_action_server.is_preempt_requested():
                success = False
                break
            monitor_feedback.current_fmeasure = 'Restoring F-measure...' + str(self.fmeasure)
            self.restore_fmeasure_action_server.publish_feedback(monitor_feedback)
            rate.sleep()

        self.fmeasure = goal.max_fmeasure #F-measure restored back to max_fmeasure
        monitor_result.raised_max = True

        if success:
            self.restore_fmeasure_action_server.set_succeeded(monitor_result)
            self.status = self.RESTORED_F
            self.restore_request = False

    def publish_fmeasure(self):
        """
        Publishes F-measure as a Float topic
        :return:
        """
        self.fmeasure_pub.publish(self.fmeasure)

    def decay(self, t):
        """
        Decay function
        :param t:
        :return:
        """
        decayed_f = self.max_fmeasure*(1 - self.decay_rate)**t
        self.fmeasure = decayed_f

    def run_operation(self, freq_hz=1):
        """
        Statuses:
        1. Idle
            > Decay of F is paused
        2. Decaying
            > Resumes decaying of F
        3. Restoring F
            > F is being restored back to max measure
        4. Restored F
            > F is fully restored
        """

        t = 0
        rate = rospy.Rate(freq_hz)
        while not rospy.is_shutdown():
            self.publish_fmeasure()
            if self.status == self.IDLE:
                pass

            elif self.status == self.DECAYING:
                self.decay(t)
                t += 1

            elif self.status == self.RESTORING_F:
                pass

            elif self.status == self.RESTORED_F:
                # Restore parameters
                t = 0
                self.global_decay_fmeasure = 0
                self.status = self.IDLE

            rate.sleep()

if __name__ == '__main__':
    Area().run_operation()