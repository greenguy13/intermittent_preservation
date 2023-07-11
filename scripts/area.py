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

from enum import Enum
import math
import rospy
from std_msgs.msg import Float32, Int8
import project_utils as pu
import pickle

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
        self.t_operation = rospy.get_param("/t_operation") #total duration of the operation
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

    def robot_status_cb(self, msg):
        """
        Callback for robot status. If robot is not on mission, we pause decay simulation
        :return:
        """
        robot_status = msg.data
        if (robot_status == robotStatus.IDLE.value) or (robot_status == robotStatus.READY.value):
            self.update_status(self.IDLE)
        elif robot_status == robotStatus.IN_MISSION.value or (robot_status == robotStatus.RESTORING_F.value and self.robot_mission_area != self.area) \
                or robot_status == robotStatus.CHARGING.value:
            self.update_status(self.DECAYING)
        elif (robot_status == robotStatus.RESTORING_F.value and self.robot_mission_area == self.area) and (self.fmeasure < self.max_fmeasure):
            self.update_status(self.RESTORING_F)
        pu.log_msg('robot', self.robot_id, 'robot status: {}. area {} status: {}'.format(robot_status, self.area, self.status))

    def mission_area_cb(self, msg):
        """

        :param msg:
        :return:
        """
        self.robot_mission_area = msg.data

    def restore_delay(self):
        """
        Delay in restoring F-measure back to max level
        :return:
        """
        delay = int(math.ceil((self.max_fmeasure-self.fmeasure) / self.restoration))
        return delay

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

    def update_status(self, status):
        """
        Updates area status
        :param status:
        :return:
        """
        self.status = status

    def dump_data(self, recorded_data):
        """
        Pickle dumps recorded F-measure data
        :return:
        """
        with open('area_{}_fmeasure.pkl'.format(self.area), 'wb') as f:
            pickle.dump(recorded_data, f)

    def shutdown(self):
        pu.log_msg('robot', self.robot_id, "Reached {} time operation. Shutting down...".format(self.t_operation), self.debug_mode)

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
        rate = rospy.Rate(freq_hz)
        f_record = list()
        t = 0
        while not rospy.is_shutdown() and len(f_record)<self.t_operation:
            if self.status == self.IDLE:
                pass

            elif self.status == self.DECAYING:
                self.decay(t)
                t += 1

            elif self.status == self.RESTORING_F:
                delay = self.restore_delay()
                for i in range(delay):
                    self.fmeasure = min(self.fmeasure+self.restoration, self.max_fmeasure)
                    f_record.append(self.fmeasure)
                    self.publish_fmeasure()
                    rate.sleep()
                self.update_status(self.RESTORED_F)

            elif self.status == self.RESTORED_F:
                # Restore parameters
                t = 0
                self.update_status(self.IDLE)

            # Save F-measure here
            f_record.append(self.fmeasure)
            self.publish_fmeasure()

            rate.sleep()

        #Pickle dump
        self.dump_data(f_record)

        rospy.on_shutdown(self.shutdown)

if __name__ == '__main__':
    Area().run_operation()