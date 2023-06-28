#!/usr/bin/env python

# Python modules
import numpy as np
import tf
import rospy
import actionlib
import project_utils as pu
from std_msgs.msg import String
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point
from std_srvs.srv import Trigger, TriggerResponse
from nav2d_navigator.msg import MoveToPosition2DAction, MoveToPosition2DGoal

INDEX_FOR_X = 0
INDEX_FOR_Y = 1


class RobotNavigator:
    # States of the node
    IDLE = 0  # Not initiated.
    MOVE_TO = 1  # Go to the next location
    STOPPED=9

    def __init__(self):
        # Current state of the node.
        self.listener = tf.TransformListener()

        self.current_state = self.IDLE
        # Parameters.
        self.robot_id = rospy.get_param('~robot_id')
        self.ros_freq = 1 #rospy.get_param('/ros_freq')
        self.map_res= rospy.get_param('/discretization_resolution')

        self.previous_point = ()
        self.current_pose = ()
        self.prev_pose = []
        self.last_distance = -1

        rospy.Subscriber('/shutdown', String, self.save_all_data)
        rospy.Subscriber('/robot_{}/robot_nav/goal'.format(self.robot_id), Point, self.initial_action_handler)
        self.start_gvg_explore = rospy.Service('/robot_{}/robot_nav/start_stop'.format(self.robot_id), Trigger, self.start_stop)
        self.goal_feedback_pub = rospy.Publisher("/robot_{}/robot_nav/feedback".format(self.robot_id), Pose, queue_size=1)
        self.idle_feedback_pub = rospy.Publisher("/robot_{}/robot_nav/idle".format(self.robot_id), Pose, queue_size=1)
        # nav2d MoveTo action.
        self.client_motion = actionlib.SimpleActionClient("/robot_{}/MoveTo".format(self.robot_id),
                                                          MoveToPosition2DAction)
        self.client_motion.wait_for_server()
        # tf listener.

        rospy.loginfo("Robot {}: Exploration server online...".format(self.robot_id))

    def start_stop(self):
        if self.current_state != self.STOPPED:
            self.client_motion.cancel_goal()
            self.current_state=self.STOPPED
        else:
            self.current_state=self.IDLE    
        return TriggerResponse()

    def move_robot_to_goal(self, goal, theta=0):
        if len(self.previous_point) == 0:
            p=self.get_robot_pose(self.robot_id)
            self.previous_point = (p[0],p[1])
        angle = np.arctan2(goal[1]-self.previous_point[1],goal[0]-self.previous_point[0])
        if angle==0:
            angle=1.57
        # rospy.logerr("Angle: {}".format(angle))
        move = MoveToPosition2DGoal()
        frame_id = '/map'.format(self.robot_id)  # TODO check.
        move.header.frame_id = frame_id
        move.target_pose.x = goal[INDEX_FOR_X]
        move.target_pose.y = goal[INDEX_FOR_Y]
        move.target_pose.theta = theta
        move.target_distance = 0.15
        move.target_angle = angle
        self.client_motion.send_goal(move, feedback_cb=self.feedback_motion_cb)
        self.prev_pose.append(self.current_pose)
        if self.client_motion.wait_for_result():
            state = self.client_motion.get_state()
            if self.current_state == self.MOVE_TO:
                if state == GoalStatus.SUCCEEDED:
                    self.previous_point=self.current_pose
                    self.current_pose = goal
                    # publish to feedback
                    self.report_to_controller()
                    # rospy.logerr("Current state: successful: {}".format(state))
                else:
                    # rospy.logerr("Current state: {}".format(state))
                    p=self.get_robot_pose(self.robot_id)
                    self.previous_point=self.current_pose
                    self.current_pose = (p[0],p[1])
                    self.report_to_controller()

                self.current_state = self.IDLE

        else:
            self.current_state = self.IDLE
            # self.report_to_controller()
            rospy.logerr("UNABLE TO INITIATE NAVIGATION")

    def report_to_controller(self):
        pose = Pose()
        pose.position.x = self.current_pose[INDEX_FOR_X]
        pose.position.y = self.current_pose[INDEX_FOR_Y]
        self.goal_feedback_pub.publish(pose)

    def feedback_motion_cb(self, feedback):
        if self.current_state == self.MOVE_TO:
            p=self.get_robot_pose(self.robot_id)
            self.prev_pose.append((p[0],p[1]))
        if np.isclose(self.last_distance, feedback.distance) or np.isclose(feedback.distance, self.last_distance):
            self.same_location_counter += 1
            if self.same_location_counter > 5:  # TODO parameter:
                self.client_motion.cancel_goal()
        else:
            self.last_distance = feedback.distance
            self.same_location_counter = 0

    def spin(self):
        r = rospy.Rate(self.ros_freq)
        while not rospy.is_shutdown():
            if self.current_state != self.STOPPED:
                if self.current_state == self.IDLE and self.current_pose != self.previous_point:
                    self.current_state = self.MOVE_TO
                    self.move_robot_to_goal(self.current_pose, pu.angle_pq_line(np.asarray(self.current_pose), np.asarray(self.previous_point)))
            r.sleep()

    def save_all_data(self, data):
        rospy.signal_shutdown("Shutting down Sampler")

    def initial_action_handler(self, req):
        self.previous_point = self.current_pose
        self.current_pose = (req.x, req.y)
        self.current_state = self.MOVE_TO
        self.move_robot_to_goal(self.current_pose, 0.0)


    def received_prempt_handler(self, data):
        self.client_motion.cancel_goal()
        self.current_state = self.IDLE
        return TriggerResponse()

    def get_robot_pose(self, rid):
        robot_pose = None
        while not robot_pose:
            try:
                self.listener.waitForTransform("map".format(rid),
                                               "robot_{}/base_link".format(rid),
                                               rospy.Time(0),
                                               rospy.Duration(4.0))
                (robot_loc_val, rot) = self.listener.lookupTransform("map".format(rid),
                                                                     "robot_{}/base_link".format(rid),
                                                                     rospy.Time(0))
                robot_pose = robot_loc_val[0:2]
            except:
                rospy.sleep(1)
                pass
        robot_pose = np.array(robot_pose)
        return robot_pose


if __name__ == "__main__":
    rospy.init_node("robot_nav")

    robot_nav = RobotNavigator()
    robot_nav.spin()
