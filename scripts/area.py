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

"""
Upnext: Mar 24
1. Make sure that the area node works. All the msg and srv.
2. Make sure the relay of information works for the entire program.
    > Make the grow tree work
    > Make the decision making work
    > Check out if everything is working
    
Upnext: Mar 30
1. Draw/sketch the program and implementation based on discussion with Alberto
"""

import math
import rospy
from std_msgs.msg import Float32, Int8
import project_utils as pu


# area, max_fmeasure, decay_rate, restoration, t_operation

# self.sub_fmeasure_dict[area] = rospy.Subscriber('/fmeasure_' + str(area), Float32, self.store_fmeasure_cb, area)

class Area():
    IDLE = 10
    IN_MISSION = 12
    RESTORING_F = 41
    RESTORED_F = 42
    def __init__(self):
        rospy.init_node('area', anonymous=True)
        self.area = rospy.get_param("~area_id")
        self.robot_id = rospy.get_param("~robot_id")
        #This can be given/written as a single parameter
        self.decay_rate = float(rospy.get_param("~decay_rate"))

        #We can write these params as general param
        #Change/evolution in decay rate; po in the general parameter
        self.t_operation = rospy.get_param("~t_operation") #total duration of the operation
        self.max_fmeasure = rospy.get_param("~max_fmeasure")
        self.fmeasure = self.max_fmeasure #initialize at max fmeasure
        self.restoration = rospy.get_param("~restoration")
        self.restore_delay = int(math.ceil(self.restoration * (self.max_fmeasure-self.fmeasure))) #delay in restoring the F-measure to max level

        pu.log_msg('area', self.area, 'Area {}. decay rate: {}. t_operation: {}. max_f: {}. restore: {}. robot id: {}'.format(self.area, self.decay_rate, self.t_operation, self.max_fmeasure, self.restoration, self.robot_id), 1)

        # Fmeasure publisher
        self.fmeasure_pub = rospy.Publisher("/area_{}/fmeasure".format(self.area), Float32, queue_size=1)

        # Server for pausing/commencing simulation
        rospy.Subscriber('/robot_{}/robot_status'.format(self.robot_id), Int8, self.robot_status_cb)

        self.status = self.IDLE

    def run_operation(self, freq_hz=1):
        t = 0

        rate = rospy.Rate(freq_hz)
        while not rospy.is_shutdown():
            self.publish_fmeasure()
            print("\nArea status:")

            """
            Previous version:
            Statuses:
            1. IDLE
                > If accept info that robot is in mission: move to next state, start decaying.
                > If robot is in idle: Status should also be idle
            2. Decaying
                > STarts decaying and publishing F
            3. Restoring
                > Action server to restore F back to max measure
                
            Updated new version:
            1. See the int_monitoring version
            
            UPNEXT: Mar 25
            1. Fix up the area node
            """

            if self.status == self.IDLE:
                self.status == self.IN_MISSION

            elif self.status == self.IN_MISSION:
                self.decay(t)
                t += 1

            elif self.status == self.RESTORING:
                pass #not decaying because being restored

            elif self.request_restore_status == self.RESTORED:
                #Restore parameters
                t = 0
                self.global_decay_fmeasure = 0
            rate.sleep()

    def robot_status_cb(self):
        """
        Callback for robot status. If robot status is thinking decisions, we pause decay simulation
        :return:
        """
        pass

    def publish_fmeasure(self):
        """
        Publishes F-measure as a Float topic
        :return:
        """
        self.fmeasure_pub.publish(self.fmeasure)

    def restore_fmeasure(self):
        """
        Restore F-measure back to max level
        """
        rate = rospy.Rate(1)
        self.status = self.RESTORING_F

        for i in range(self.restore_delay):
            #PO: Give notice that we are restoring F
            rate.sleep()

        self.fmeasure = self.max_fmeasure

    def decay(self, t):
        """
        Decay function
        :param t:
        :return:
        """
        decayed_f = self.max_fmeasure*(1 - self.decay_rate)**t
        self.fmeasure = decayed_f


if __name__ == '__main__':
    Area().run_operation()