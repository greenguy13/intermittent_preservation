#!/usr/bin/env python

"""
Pauses simulation whenever a robot/central planner is thinking for a decision
For brevity, we refer to the robot/central planner as agent
"""

import rospy
from int_preservation.srv import pauseSimulation, pauseSimulationResponse
from std_srvs.srv import SetBool
import project_utils as pu


class PauseSimulation:
    def __init__(self, node_name):

        rospy.init_node(node_name, anonymous=True)
        self.debug_mode = rospy.get_param('/debug_mode')

        #TODO: Client will have to wait for this service and request
        self.pause_queue_server = rospy.Service("/pause_queue_server", pauseSimulation, self.pause_request_cb) #Server for pause simulation requests, i.e., when an agent is thinking
        self.pause_queue = list() #Queue of agents thinking
        self.is_simulation_paused = False #Bool whether simulation is paused

    def pause_request_cb(self, msg):
        """

        :param agent_id:
        :return:
        """
        is_pause = msg.is_pause #Receives message from agent whether thinking or done thinking
        agent_id = msg.agent_id #Agent id

        #Agent is thinking, so we pause simulation, if not yet
        is_pause = bool(is_pause)
        if is_pause is True:
            if self.is_simulation_paused is False:
                self.request_pause_simulation(is_pause) #Request Stage to pause simulation
                self.is_simulation_paused = True
            self.pause_queue.append(agent_id) #Insert into agents that are thinking

        #Agent is done thinking, and so resume simulation if all agents concurrently thinking are likewise done
        else:
            self.pause_queue.remove(agent_id) #Remove the agent that finished thinking from those that are still thinking
            if len(self.pause_queue) == 0:
                self.request_pause_simulation(is_pause)
                self.is_simulation_paused = False

        #TODO: Print diagnostics here, like the simulation time when a request is to pause. Do the same for un-pause
        time = rospy.Time.now()
        if self.is_simulation_paused:
            self.debug("Simulation paused. Time: {}".format(time))
        else:
            self.debug("Simulation un-paused. Time: {}".format(time))


        #TODO: Send out the response of this pause/unpause request
        return pauseSimulationResponse(self.is_simulation_paused) #TODO: If this is the result, then we should match this with the client input/msg

    def request_pause_simulation(self, is_pause):
        """
        Here we are a client. We send a pause request, (i.e., pause if is_pause is True, while unpause otherwise)
        :param is_pause:
        :return:
        """
        rospy.wait_for_service('/stage/pause')

        try:
            pause_sim = rospy.ServiceProxy('/stage/pause', SetBool)
            resp = pause_sim(is_pause)
            result = resp.success
            message = resp.message
            self.debug("Pause simulation result: {}. Message: {}".format(result, message))
            return bool(result)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
    def run_operation(self):
        rospy.spin()

    def debug(self, msg):
        pu.log_msg(type='pause_simulator', id=None, msg=msg, debug=self.debug_mode)

if __name__ == '__main__':
    PauseSimulation('pause_simulator').run_operation()


