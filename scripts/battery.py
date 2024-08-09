#!/usr/bin/env python

"""
### Robot battery
Battery node that depletes when the robot is in mission, charges up when in charging station, and idle when robot is idle/thinking.

    Publisher:
        battery level
    Subscriber:
        robot status

"""
import math
import rospy
from std_msgs.msg import Float32, Int8
import project_utils as pu
import pickle
import os
from status import battStatus, robotStatus

class Battery():
    def __init__(self, node_name):
        #Params
        rospy.init_node(node_name)
        self.robot_id = rospy.get_param('~robot_id')
        self.max_battery = rospy.get_param("/max_battery")
        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_depletion_travel, self.batt_depletion_restoring_f = batt_consumed_per_time #two types of batt depletion rate: while travelling and restoring F
        self.batt_restoration = rospy.get_param("/restoration")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data
        self.debug_mode = rospy.get_param("/debug_mode")

        #Publisher/subscriber
        rospy.Subscriber("/robot_{}/robot_status".format(self.robot_id), Int8, self.robot_status_cb)
        self.battery_pub = rospy.Publisher("/robot_{}/battery".format(self.robot_id), Float32, queue_size=1)
        self.battery_status_pub = rospy.Publisher("/robot_{}/battery_status".format(self.robot_id), Int8, queue_size=1)

        #Init values
        self.battery = self.max_battery
        self.status = battStatus.IDLE.value
        self.batt_depletion_rate = self.batt_depletion_travel

    def robot_status_cb(self, msg):
        """
        Callback to robot status. Here we update the battery status as either charging or depeting depending on the robot status
        :return:
        """
        robot_status = msg.data
        if robot_status == robotStatus.IDLE.value or robot_status == robotStatus.READY.value or robot_status == robotStatus.CONSIDER_REPLAN.value:
            self.update_status(battStatus.IDLE)
        elif robot_status == robotStatus.IN_MISSION.value or robot_status == robotStatus.RESTORING_F.value:
            #Differentiate the rate of battery depletion between travelling and restoring F
            self.batt_depletion_rate = self.batt_depletion_travel
            if robot_status == robotStatus.RESTORING_F.value:
                self.batt_depletion_rate = self.batt_depletion_restoring_f
            self.update_status(battStatus.DEPLETING)
        elif (robot_status == robotStatus.CHARGING.value) and (self.battery < self.max_battery):
            self.update_status(battStatus.CHARGING)
        self.debug('robot status: {}. battery status: {} level: {}'.format(robot_status, self.status, self.battery))

    def update_status(self, status):
        """
        Updates battery status
        :param status:
        :return:
        """
        self.status = status.value


    def charge_delay(self):
        """
        Delay in restoring F-measure back to max level
        :return:
        """
        delay = int(math.ceil((self.max_battery - self.battery) / self.batt_restoration))
        return delay

    def publish_battery(self):
        """
        Publishes battery level as a Float topic and status as Int topic
        :return:
        """
        self.battery_pub.publish(self.battery)
        self.battery_status_pub.publish(self.status)

    def run_operation(self, filename, freq=1):
        if self.robot_id == 0:
            rate = rospy.Rate(freq)
            battery_record, battery_status_record = [], []
            while not rospy.is_shutdown():
                battery_status_record.append(self.status)
                if self.status == battStatus.IDLE.value:
                    pass

                elif self.status == battStatus.DEPLETING.value:
                    self.battery -= self.batt_depletion_rate

                elif self.status == battStatus.CHARGING.value:
                    delay = self.charge_delay()
                    for i in range(delay):
                        self.battery = min(self.battery+self.batt_restoration, self.max_battery)
                        battery_record.append(self.battery)
                        self.publish_battery()
                        rate.sleep()
                    self.update_status(battStatus.FULLY_CHARGED)

                elif self.status == battStatus.FULLY_CHARGED.value:
                    self.update_status(battStatus.IDLE)

                #Store battery here
                if self.status != battStatus.IDLE.value:
                    battery_record.append(self.battery)
                self.publish_battery()

                if self.save:
                    pu.dump_data(battery_record, '{}_robot{}_battery'.format(filename, self.robot_id))
                    pu.dump_data(battery_status_record, '{}_robot{}_batt_status'.format(filename, self.robot_id))
                rate.sleep()

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation/results')
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    Battery('battery').run_operation(filename)
