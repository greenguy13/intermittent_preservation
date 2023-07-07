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
from std_msgs.msg import Float32, Int8
import project_utils as pu


# area, max_fmeasure, decay_rate, restoration, t_operation

# self.sub_fmeasure_dict[area] = rospy.Subscriber('/fmeasure_' + str(area), Float32, self.store_fmeasure_cb, area)

class robotStatus(Enum):
    IDLE = 10
    READY = 11
    IN_MISSION = 20
    CHARGING = 30
    RESTORING_F = 40

class Area():
    IDLE = 0
    DECAYING = 1
    RESTORING_F = 10
    RESTORED_F = 11

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

        # Published topics
        self.fmeasure_pub = rospy.Publisher("/area_{}/fmeasure".format(self.area), Float32, queue_size=1)
        self.status_pub = rospy.Publisher("/area_{}/status".format(self.area), Int8, queue_size=1)

        # Subscribed topics
        rospy.Subscriber('/robot_{}/robot_status'.format(self.robot_id), Int8, self.robot_status_cb)
        rospy.Subscriber('/robot_{}/mission_area'.format(self.robot_id), Int8, self.mission_area_cb)

        self.status = self.IDLE
        self.robot_mission_area = None

    def restore_delay(self):
        """
        Delay in restoring F-measure back to max level
        :return:
        """
        delay = int(math.ceil(self.restoration * (self.max_fmeasure-self.fmeasure)))
        return delay

    def robot_status_cb(self, msg):
        """
        TODO P5: Callback for robot status. If robot is not in mission, we pause decay simulation
        :return:
        """
        robot_status = msg.data
        if (robot_status == robotStatus.IDLE.value) or (robot_status == robotStatus.READY.value):
            self.status = self.IDLE
        elif robot_status == robotStatus.IN_MISSION.value or (robot_status == robotStatus.RESTORING_F.value and self.robot_mission_area != self.area) \
                or robot_status == robotStatus.CHARGING.value:
            self.status = self.DECAYING
        elif (robot_status == robotStatus.RESTORING_F.value and self.robot_mission_area == self.area) and (self.fmeasure < self.max_fmeasure):
            self.status = self.RESTORING_F
        pu.log_msg('robot', self.robot_id, 'robot status: {}. area {} status: {}'.format(robot_status, self.area, self.status))

    def mission_area_cb(self, msg):
        """

        :param msg:
        :return:
        """
        self.robot_mission_area = msg.data

    def publish_fmeasure(self):
        """
        Publishes F-measure as a Float topic and status as Int topic
        :return:
        """
        self.fmeasure_pub.publish(self.fmeasure)  # publish F-measure
        self.status_pub.publish(self.status)  # publish area status

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
                delay = self.restore_delay()
                for i in range(delay):
                    rate.sleep()
                self.fmeasure = self.max_fmeasure  # F-measure restored back to max_fmeasure
                self.status = self.RESTORED_F

            elif self.status == self.RESTORED_F:
                # Restore parameters
                t = 0
                self.status = self.IDLE

            rate.sleep()

if __name__ == '__main__':
    Area().run_operation()