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
import os
import math
import rospy
from std_msgs.msg import Float32, Int8
import project_utils as pu
import pickle
from status import areaStatus, robotStatus
from int_preservation.srv import flevel, flevelResponse
from loss_fcns import *



class Area():
    def __init__(self):
        rospy.init_node('area', anonymous=True)
        self.debug_mode = rospy.get_param("/debug_mode")
        self.area = rospy.get_param("~area_id")
        # TODO: When this becomes multi-robot, there has to be a robot assigned to it sent by a planner/auctioneer
        #   PO: We can do an if, something like if nrobots = 1: self.robot_id = rospy.get_param("~decay_rate")
        #   PO: Else (nrobots > 1): this becomes an assignment which can be a subscribed topic/message request

        # TODO: Pause of simulation
        #   PO1: We can have a global msg that tells all areas to pause their simulation?
        #   PO2: Or by the robot assigned to them?
        #   Remark: These 2 are possible


        self.robot_id = rospy.get_param("~robot_id")
        decay_rate = rospy.get_param("~decay_rate")
        self.decay_rate = float(decay_rate)
        decay_evolution = rospy.get_param("~decay_evolution")
        self.decay_evolution_list = eval(decay_evolution)
        self.evolving_decay = True if len(self.decay_evolution_list) > 0 else False
        self.t_operation = rospy.get_param("/t_operation") #total duration of the operation
        self.max_fmeasure = rospy.get_param("~max_fmeasure")
        self.fmeasure = self.max_fmeasure #initialize at max fmeasure
        self.restoration = rospy.get_param("~restoration")
        self.save = rospy.get_param("/save")  # Whether to save data
        self.debug_mode = rospy.get_param("/debug_mode")

        self.debug('Area {}. decay rate: {}. t_operation: {}. max_f: {}. restore: {}. robot id: {}'.format(self.area, self.decay_rate, self.t_operation, self.max_fmeasure, self.restoration, self.robot_id))
        self.debug("Decay evolution list: {}".format(self.decay_evolution_list))

        # Published topics
        self.decay_rate_pub = rospy.Publisher("/area_{}/decay_rate".format(self.area), Float32, queue_size=1)
        self.fmeasure_pub = rospy.Publisher("/area_{}/fmeasure".format(self.area), Float32, queue_size=1)
        self.status_pub = rospy.Publisher("/area_{}/status".format(self.area), Int8, queue_size=1)

        # Subscribed topics
        ## Suggestion: General/multi-robots
        rospy.Subscriber('/robot_{}/robot_status'.format(self.robot_id), Int8, self.robot_status_cb)
        rospy.Subscriber('/robot_{}/mission_area'.format(self.robot_id), Int8, self.mission_area_cb)

        # Service server: Fmeasure
        self.fmeasure_server = rospy.Service("/flevel_server_" + str(self.area), flevel, self.report_flevel_cb)

        self.status = areaStatus.IDLE.value
        self.robot_mission_area = None
        self.tlapse = 0
        self.decay_evolve_tframe = round(self.t_operation / (len(self.decay_evolution_list) + 1))
        self.sim_t = 0


    def robot_status_cb(self, msg):
        """
        Callback for robot status. If robot is not on mission, we pause decay simulation
        :return:
        """
        robot_status = msg.data
        if (robot_status == robotStatus.IDLE.value) or (robot_status == robotStatus.READY.value) or (robot_status == robotStatus.CONSIDER_REPLAN.value):
            self.update_status(areaStatus.IDLE)
        elif robot_status == robotStatus.IN_MISSION.value or (robot_status == robotStatus.RESTORING_F.value and self.robot_mission_area != self.area) \
                or robot_status == robotStatus.CHARGING.value:
            self.update_status(areaStatus.DECAYING)
        elif (robot_status == robotStatus.RESTORING_F.value and self.robot_mission_area == self.area) and (self.fmeasure < self.max_fmeasure):
            self.update_status(areaStatus.RESTORING_F)
        self.debug('robot status: {}. area {} status: {}, decay: {}, fmeasure: {} tlapse: {}, sim_t: {}'.format(robot_status, self.area, self.status, self.decay_rate, self.fmeasure, self.tlapse, self.sim_t))

    def mission_area_cb(self, msg):
        """

        :param msg:
        :return:
        """
        self.robot_mission_area = msg.data

    #TODO: For now, we are setting the as-is naming for F-level request when we modified it to give/provide the decay rate
    def report_flevel_cb(self, msg):
        """
        Callback as Service Server for F-measure
        :param msg:
        :return:
        """
        if bool(msg.fmeasure_request) is True:
            record = self.fmeasure
            return flevelResponse(record)

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
        self.decay_rate_pub.publish(self.decay_rate) # publish decay rate
        self.fmeasure_pub.publish(self.fmeasure)  # publish F-measure
        self.status_pub.publish(self.status)  # publish area status

    def decay(self, t):
        """
        Decay function
        :param t:
        :return:
        """
        # decayed_f = self.max_fmeasure*(1 - self.decay_rate)**t
        derivative = 0
        if t>0:
            derivative = (self.max_fmeasure*(1 - self.decay_rate)**t) * math.log(1 - self.decay_rate)
        decayed_f2 = self.fmeasure + derivative
        # self.debug("Orig decay: {}. Derivative decay: {}, tlapse: {}".format(decayed_f, decayed_f2, t))
        self.fmeasure = max(decayed_f2, 0)

    def update_status(self, status):
        """
        Updates area status
        :param status:
        :return:
        """
        self.status = status.value

    def run_operation(self, filename, freq_hz=1):
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
        f_record, status_record, decay_rates_record = [], [], []
        if self.evolving_decay:
            time_decay_evolves, evolve_decay_idx = self.sim_t + self.decay_evolve_tframe, 0 #first time stamp where decay rate evolves

        while not rospy.is_shutdown():
            status_record.append(self.status)
            if self.evolving_decay and (self.sim_t >= time_decay_evolves) and (self.sim_t < self.t_operation):
                self.decay_rate = (1 + self.decay_evolution_list[evolve_decay_idx])*self.decay_rate
                self.debug("Decay now evolved to {} beginning time {}".format(self.decay_rate, time_decay_evolves))
                #Set the next evolution time stamp
                time_decay_evolves = self.sim_t + self.decay_evolve_tframe
                evolve_decay_idx += 1

            if self.status == areaStatus.IDLE.value:
                pass

            elif self.status == areaStatus.DECAYING.value:
                self.decay(self.tlapse)
                self.tlapse += 1

            elif self.status == areaStatus.RESTORING_F.value:
                delay = self.restore_delay()
                for i in range(delay):
                    self.fmeasure = min(self.fmeasure+self.restoration, self.max_fmeasure)
                    f_record.append(self.fmeasure)
                    self.publish_fmeasure()
                    rate.sleep()
                # Restore parameters
                self.tlapse = 0
                self.update_status(areaStatus.RESTORED_F)

            elif self.status == areaStatus.RESTORED_F.value:
                self.update_status(areaStatus.IDLE)

            if self.status != areaStatus.IDLE.value:
                f_record.append(self.fmeasure)
                decay_rates_record.append(self.decay_rate)
                self.sim_t += 1
            self.publish_fmeasure()

            if self.save:
                pu.dump_data(decay_rates_record, '{}_area{}_decay_rates'.format(filename, self.area))
                pu.dump_data(f_record, '{}_area{}_fmeasure'.format(filename, self.area))
                pu.dump_data(status_record, '{}_area{}_status'.format(filename, self.area))
            rate.sleep()

    def debug(self, msg):
        pu.log_msg('area', self.area, msg, self.debug_mode)

if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation/results')
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    Area().run_operation(filename)