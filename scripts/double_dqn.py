#!/usr/bin/env python

# !/usr/bin/env python
# Author: Amel Docena

import random
import tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from collections import deque
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error
import keras
import pickle
import numpy as np
import rospy
import actionlib
from loss_fcns import *
import project_utils as pu
from nav_msgs.srv import GetPlan
from std_msgs.msg import Int8, Float32
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from status import areaStatus, battStatus, robotStatus
from reset_simulation import *

SUCCEEDED = 3 #GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html


LAUNCH_FILE = 'simulation.launch' #TO-change
PATH = '~/catkin_ws/src/cs81-fp/'
WORLD = 'PA1.world'
NODES_TO_KILL = ['random_walk', 'finder', 'robot_motion', 'stage_ros']
SLEEP = 5

class DeepQLearning:
    def __init__(self, node_name):
        rospy.init_node(node_name)

        #Subscribers
        self.robot_id = rospy.get_param("~robot_id")
        self.debug_mode = rospy.get_param("/debug_mode")
        # self.robot_velocity = rospy.get_param("/robot_velocity")  # Linear velocity of robot; we assume linear and angular are relatively equal
        self.gamma = rospy.get_param("/gamma")  # discount factor
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery")  # Max battery
        self.battery_reserve = rospy.get_param("/battery_reserve")  # Battery reserve
        self.fsafe, self.fcrit = rospy.get_param("/f_thresh")  # (safe, crit)
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = rospy.get_param("/batt_consumed_per_time")  # (travel, restoration)
        # self.dec_steps = rospy.get_param("/dec_steps")  # STAR
        # self.restoration = rospy.get_param("/restoration")
        # self.noise = rospy.get_param("/noise")
        self.nareas = rospy.get_param("/nareas")  # Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i + 1) for i in range(self.nareas)]  # list of int area IDs
        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data

        # Initialize variables
        charging_station_coords = rospy.get_param("~initial_pose_x"), rospy.get_param("~initial_pose_y")  # rospy.get_param("/charging_station_coords")
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        self.sampled_nodes_poses = [charging_pose_stamped]  # list container for sampled nodes of type PoseStamped

        # Pickle load the sampled area poses
        with open('{}.pkl'.format(rospy.get_param("/file_sampled_areas")), 'rb') as f:
            sampled_areas_coords = pickle.load(f)
        for area_coords in sampled_areas_coords['n{}_p{}'.format(self.nareas, rospy.get_param("/placement"))]:
            pose_stamped = pu.convert_coords_to_PoseStamped(area_coords)
            self.sampled_nodes_poses.append(pose_stamped)

        self.charging_station = 0
        self.curr_loc = self.charging_station  # Initial location robot is the charging station
        self.battery = self.max_battery  # Initialize battery at max, then gets updated by subscribed battery topic
        self.optimal_path = []
        self.dist_matrix = None
        self.mission_area = None
        self.robot_status = robotStatus.IDLE.value
        self.available = True
        self.curr_fmeasures = dict()  # container of current F-measure of areas
        self.decay_rates_dict = dict()  # dictionary for decay rates
        for area in self.areas:
            self.decay_rates_dict[str(area)] = None
        self.decay_rates_counter = 0  # counter for stored decay rates; should be equal to number of areas
        self.decisions_made, self.decisions_accomplished, self.status_history = [], [], []  # record of data

        # We sum this up
        self.environment_status = dict()
        for node in range(self.nareas + 1):
            self.environment_status[node] = 999

        # Publishers/Subscribers
        # Service request to move_base to get plan : make_Plan
        server = '/robot_' + str(self.robot_id) + '/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        rospy.Subscriber('/robot_{}/battery_status'.format(self.robot_id), Int8, self.battery_status_cb)
        rospy.Subscriber('/robot_{}/battery'.format(self.robot_id), Float32, self.battery_level_cb)

        for area in self.areas:
            rospy.Subscriber('/area_{}/decay_rate'.format(area), Float32, self.decay_rate_cb, area)
            rospy.Subscriber('/area_{}/fmeasure'.format(area), Float32, self.area_fmeasure_cb, area)  # REMARK: Here we assume that we have live measurements of the F-measures
            rospy.Subscriber('/area_{}/status'.format(area), Int8, self.area_status_cb, area)

        self.robot_status_pub = rospy.Publisher('/robot_{}/robot_status'.format(self.robot_id), Int8, queue_size=1)
        self.mission_area_pub = rospy.Publisher('/robot_{}/mission_area'.format(self.robot_id), Int8, queue_size=1)

        # Action client to move_base
        self.robot_goal_client = actionlib.SimpleActionClient('/robot_' + str(self.robot_id) + '/move_base', MoveBaseAction)
        self.robot_goal_client.wait_for_server()

        self.gamma = rospy.get_param('/gamma') #dicount factor for future rewards
        self.epsilon = rospy.get_param('/epsilon') #epsilon for greedy selection
        self.number_episodes = rospy.get_param('/number_episodes') #number of training episodes
        self.state_dims = rospy.get_param('/state_dims') #number of states
        self.action_dims = rospy.get_param('/action_dims') #number of actions
        self.replay_buffer_size = rospy.get_param('/replay_buffer_size') #replay buffer size
        self.batch_replay_buffer_size = rospy.get_param('/batch_replay_buffer_size') #sample size of replay buffer for batch training
        self.update_target_network_period = rospy.get_param('/update_target_network_period')#number of training episodes for the network to get updated
        self.length_replay_data = rospy.get_param('/length_replay_data') #number of data points stored; if exceeded, a terminal state

        self.select_action_random = rospy.get_param('/select_action_random') #number of episodes where we choose action randomly for batch replay
        self.select_action_decrease_eps = rospy.get_param('/select_action_decrease_eps') #number of episodes where we decrease eps for eps-greedy action selection
        self.train_epochs = rospy.get_param('/train_epochs') #training epochs of the main network

        #Initialize DQN params/variables
        self.counter_update_target_network = 0

        # this sum is used to store the sum of rewards obtained during each training episode
        self.sum_rewards_episode = []

        # replay buffer
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        # this is the main network
        self.main_network = self.createNetwork()

        # this is the target network
        self.target_network = self.createNetwork()

        # copy the initial weights to target_network
        self.target_network.set_weights(self.main_network.get_weights())

        # this list is used in the cost function to select certain entries of the
        # predicted and true sample matrices in order to form the loss
        self.actions_append = []

    #Navigation methods
    def go_to_target(self, goal_idx):
        """
        Action client to move_base to move to target goal
        Goal is PoseStamped msg
        :param: goal_idx, index of goal in sampled_nodes_poses list
        :return:
        """
        goal = self.sampled_nodes_poses[goal_idx]
        self.send_robot_goal(goal)

    def send_robot_goal(self, goal):
        """
        Sends robot to goal via action client
        :param robot:
        :param goal: PoseStamped object
        :return:
        """
        movebase_goal = MoveBaseGoal()
        movebase_goal.target_pose = goal
        self.available = False
        action_goal_cb = (lambda state, result: self.action_send_done_cb(state, result, self.robot_id))
        self.robot_goal_client.send_goal(movebase_goal, done_cb=action_goal_cb, active_cb=self.action_send_active_cb)

    def action_send_active_cb(self):
        """
        Sets robot as unavailable when pursuing goal
        :return:
        """
        self.available = False
        self.update_robot_status(robotStatus.IN_MISSION)

    def action_send_done_cb(self, state, result, robot_id):
        """

        :param msg:
        :return:
        """
        if state == SUCCEEDED:
            self.curr_loc = self.mission_area
            self.update_robot_status(robotStatus.RESTORING_F)
            if self.mission_area == self.charging_station:
                self.update_robot_status(robotStatus.CHARGING)
            self.decisions_accomplished.append(self.mission_area) #TODO: Here we are storing decisions accomplished. Still needed?

    def think_decisions(self):
        """
        Thinks of the optimal path before starting mission
        :return:
        """
        self.optimal_path = [self.charging_station]  # Default decision is to go back to the charging station
        tree = self.grow_tree(self.dec_steps, self.restoration, self.noise)
        if tree:
            self.optimal_path = self.get_optimal_branch(tree)  # Indices of areas/nodes

    def commence_mission(self):
        """
        Commences mission
        :return:
        """
        if self.send2_next_area() == 0:
            self.update_robot_status(robotStatus.IDLE)

    def send2_next_area(self):
        """
        Sends the robot to the next area in the optimal path:
        :return:
        """
        if len(self.optimal_path):
            self.mission_area = self.optimal_path.pop(0)
            self.mission_area_pub.publish(self.mission_area)
            self.debug('Heading to: {}. {}'.format(self.mission_area, self.sampled_nodes_poses[self.mission_area]))
            self.decisions_made.append(self.mission_area)  # store decisions made
            self.go_to_target(self.mission_area)
            return 1
        return 0

    def update_robot_status(self, status):
        """
        Updates robot status
        :param status:
        :return:
        """
        self.robot_status = status.value

    def battery_level_cb(self, msg):
        """
        Callback for battery level
        :param msg:
        :return:
        """
        self.battery = msg.data

    def battery_status_cb(self, msg):
        """

        :param msg:
        :return:
        """
        self.environment_status[self.charging_station] = msg.data
        if msg.data == battStatus.FULLY_CHARGED.value:
            if self.robot_id < 999: self.debug("Fully charged!")
            self.available = True
            self.update_robot_status(robotStatus.IN_MISSION)

    def area_status_cb(self, msg, area_id):
        """

        :param msg:
        :return:
        """
        self.environment_status[area_id] = msg.data
        if msg.data == areaStatus.RESTORED_F.value:
            if self.robot_id < 999: self.debug("Area fully restored!")
            self.available = True
            self.update_robot_status(robotStatus.IN_MISSION)

    def decay_rate_cb(self, msg, area_id):
        """
        Store decay rate
        :param msg:
        :param area_id:
        :return:
        """
        if self.decay_rates_dict[str(area_id)] == None:
            if self.robot_id < 999: self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates_dict[str(area_id)] = msg.data
            self.decay_rates_counter += 1

    def area_fmeasure_cb(self, msg, area_id):
        """
        Updates fmeasure of area
        :param msg:
        :param area_id:
        :return:
        """
        self.curr_fmeasures[area_id] = msg.data

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Reached {} time operation. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)

    #DQN methods
    def get_curr_state(self):
        """
        Gets current state: robot battery, current location and F-measures of areas
        Returns:

        """
        elements = self.battery, self.curr_loc, self.curr_fmeasures
        state = self.state_fcn(elements)
        return state

    def state_fcn(self, elements):
        """
        Computes the state given the elements:
            Curr battery
            Curr location
            F-measures of areas (stored as tuple)
        Returns:

        """
        curr_battery, curr_loc, curr_fmeasures = elements
        state = (curr_battery, curr_loc, curr_fmeasures) #State: identity function
        assert len(state) == self.state_dims, "Invalid! Incongruent state dimension {} and length of state function return {}".format(self.state_dims, len(state))
        return np.array(state)

    def reward_fcn(self, prev_state, curr_state):
        """
        Reward function of a given state

        Args:
            prev_state: previous state
            curr_state: current state
        Returns:

        """
        prev_battery, prev_loc, prev_fmeasures = prev_state
        curr_battery, curr_loc, curr_fmeasures = curr_state

        #Loss from F-measures
        fmeasure_loss = compute_cost_fmeasures(curr_fmeasures, self.fsafe, self.fcrit)

        #Loss from battery consumption
        batt_consumption = prev_battery - curr_battery
        batt_loss = batt_consumption

        total_loss = fmeasure_loss + batt_loss
        """
        Remarks:
        1. Battery consumption matters especially when we are in the secure zone, perhaps in caution zone, but not really in the critical zone
        2. Potential additions: 
            > The average distance of a node toward other nodes (an additional cost of going/arriving at a node aside from battery consumption)
            > The time/duration an area has been in specific zone, esp. the critical zone.
        """
        reward = -total_loss
        return -reward

    def my_loss_fn(self, y_true, y_pred):

        s1, s2 = y_true.shape
        indices = np.zeros(shape=(s1, 2))
        indices[:, 0] = np.arange(s1)
        indices[:, 1] = self.actions_append
        loss = mean_squared_error(gather_nd(y_true, indices=indices.astype(int)),
                                  gather_nd(y_pred, indices=indices.astype(int)))
        # print(loss)
        return loss

    # create a neural network
    def createNetwork(self): #TODO: Update dims if needed
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_dims, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_dims, activation='linear'))
        model.compile(optimizer=RMSprop(), loss=self.my_loss_fn, metrics=['accuracy'])
        return model

    def is_terminal(self, state):
        """
        Determines whether a state is terminal:
            Battery becomes non-positive and robot not in charging station
        Returns:

        """
        curr_battery, curr_loc, _ = state
        if curr_battery <= 0 and curr_loc != self.charging_station:
            return True
        return False

    def process_action_result(self, action_result): #TODO: What does this actually do?
        """
        Processes action_result into meaningful elements for state function
        These are provided by the action server, which previous was the robot that does the action.
        Right now, since we are using the move_base we just have to get the current state, there's no need for this method
        Args:
            action_result:

        Returns:

        """
        robot_pose = action_result[0], action_result[1], action_result[2]
        object_pose = action_result[3], action_result[4], action_result[5]
        d_front, d_left, d_right = action_result[6], action_result[7], action_result[8]
        processed = robot_pose, object_pose, d_front, d_left, d_right
        return processed

    def trainingEpisodes(self): #CH
        """
        Training episodes

        Returns:

        """

        for indexEpisode in range(self.number_episodes):
            self.debug("Simulating episode {}".format(indexEpisode))
            rewards_episode = []

            # Start simulation
            """
            Q: What are we simulating here?
            The environment
            The robot
            Who thinks?
            We reset. When do we kill? Which nodes do we kill and keep alive?
            When do we reset?
            Are we sure that in killing and resetting, necessary data are stored?
            """
            launch_nodes(launch_file=LAUNCH_FILE, sleep=SLEEP) #TODO: Update the launch file

            #TODO: Change
            while len(self.sampled_nodes_poses) != self.nareas+1:
                self.debug("Topics not yet registered. Sleeping...")
                time.sleep(1)
            self.debug("Topics registered!")
            currentState = self.get_curr_state()

            ndata = 0
            terminalState = False
            while not terminalState:
                # select an action on the basis of the current state, denoted by currentState
                # then get subsequent, state, reward, and bool whether a terminal state

                """
                Potentially: Define a finite state machine here
                IDLE (Not doing anything)
                    > We process the results of the previous action and update the DQN
                    If we selected an action before and now accomplished:
                        > we process the resulting state, computing the reward
                        > terminal state?
                        > append the reward to the rewards episode
                        > append to the replay buffer
                        > potentially train the network once condition of having ample replay buffer size
                        > set the current state as the next state
                        > increase the counter of data replay buffer data length, until full, which becomes a terminal state, 
                                ending the simulation
                    > Transition to READY
                    
                READY     
                    > select action
                    
                IN-MISSION
                    > commence_mission/action
                    
                CHARGING
                    > debug: Charging to full
                    > Note: If fully charged, transition to IDLE 
                RESTORING_F
                    > debug: Restoring F to full
                    > Note: If fully restored, transition to IDLE
                """

                action = self.selectAction(currentState, indexEpisode)
                print("Action request:", action)
                action_result = self.make_action_request(action) #Essentially this collects the current state after conducting the action
                """
                What we can do here if the action is done, we just collect the topics subscribed
                The question is: How can we make it as a finite state machine? Note that in our current configuration, the interplay of information 
                    follows a finite state machine.
                    Q: Do we really need a FSM for this? Yes. I think so.
                    Q: Perhaps we do a run_operation. We only the 
                """
                elements = self.process_action_result(action_result)
                nextState = self.state_fcn(elements)
                reward = self.reward_fcn(nextState)
                terminalState = self.is_terminal(nextState)
                print("Next State: {}, {}, {}. Reward: {}, {}. Terminal state: {}, {}".format(type(nextState),
                                                                                              nextState.shape,
                                                                                              nextState,
                                                                                              type(reward), reward,
                                                                                              type(terminalState),
                                                                                              terminalState))
                rewards_episode.append(reward)

                # add current state, action, reward, next state, and terminal flag to the replay buffer
                self.replay_buffer.append((currentState, action, reward, nextState, terminalState))

                # train network
                self.trainNetwork()

                # set the current state for the next step
                currentState = nextState
                ndata += 1
                if ndata == self.length_replay_data:
                    print("Collected enough datapoints. Terminating...")
                    terminalState = True
            print("Sum of rewards {}".format(np.sum(rewards_episode)))
            self.sum_rewards_episode.append(np.sum(rewards_episode))
            kill_nodes(NODES_TO_KILL, SLEEP)

        return True

    def selectAction(self, state, index): #CH
        """
        Selects an action on the basis of the current state.
        Implements epsilon-greedy approach
        Args:
            state: state for which to compute the action
            index: index of the current episode
        """

        # first index episodes we select completely random actions to have enough exploration
        if index < self.select_action_random:
            return np.random.choice(self.action_dims)

        # epsilon greedy approach
        randomNumber = np.random.random()

        # after index episodes, we slowly start to decrease the epsilon parameter
        if index > self.select_action_decrease_eps:
            self.epsilon = 0.955 * self.epsilon

        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.action_dims)

            # otherwise, we are selecting greedy actions
        else:
            Qvalues = self.main_network.predict(state.reshape(1, self.state_dims))
            return np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])  # we choose the max

    def trainNetwork(self): #VER

        # if the replay buffer has at least batch_replay_buffer_size elements,
        # then train the model
        # otherwise wait until the size of the elements exceeds batch_replay_buffer_size
        if (len(self.replay_buffer) > self.batch_replay_buffer_size):

            # sample a batch from the replay buffer
            randomSampleBatch = random.sample(self.replay_buffer, self.batch_replay_buffer_size)

            # get current state batch and next state batch as inputs for prediction
            currentStateBatch = np.zeros(shape=(self.batch_replay_buffer_size, self.state_dims))
            nextStateBatch = np.zeros(shape=(self.batch_replay_buffer_size, self.state_dims))

            for index, tupleS in enumerate(randomSampleBatch):
                currentStateBatch[index, :] = tupleS[0]
                nextStateBatch[index, :] = tupleS[3]

            # here, use the target network to predict Q-values
            QnextStateTargetNetwork = self.target_network.predict(nextStateBatch)
            # here, use the main network to predict Q-values
            QcurrentStateMainNetwork = self.main_network.predict(currentStateBatch)

            # form batches for training
            # input for training
            inputNetwork = currentStateBatch
            # output for training
            outputNetwork = np.zeros(shape=(self.batch_replay_buffer_size, self.action_dims))

            # list of actions that are selected from the batch
            self.actions_append = []
            for index, (currentState, action, reward, nextState, terminated) in enumerate(randomSampleBatch):
                # if the next state is the terminal state
                if terminated:
                    y = reward  # Q-value pred for terminal state
                else:
                    y = reward + self.gamma * np.max(QnextStateTargetNetwork[index])  # Q-value prediction

                self.actions_append.append(action)

                outputNetwork[index] = QcurrentStateMainNetwork[index]
                outputNetwork[index, action] = y

            # train the network
            self.main_network.fit(inputNetwork, outputNetwork, batch_size=self.batch_replay_buffer_size, verbose=0,
                                 epochs=self.train_epochs)

            # after update_target_network_period training sessions, update the coefficients of the target network
            # increase the counter for training the target network
            self.counter_update_target_network += 1
            if (self.counter_update_target_network > (self.update_target_network_period - 1)):
                # copy the weights to target_network
                self.target_network.set_weights(self.main_network.get_weights())
                print("Target network updated!")
                print("Counter value {}".format(self.counter_update_target_network))
                # reset the counter
                self.counter_update_target_network = 0

    def train_model(self, path, file): #CH
        """
        Trains the robot to follow object via DQN
        Returns:

        """
        done = False
        while not rospy.is_shutdown() and done is not True:
            done = self.trainingEpisodes()

        print("Sum of rewards across episodes:", self.sum_rewards_episode)
        with open(path + file + '.pkl', 'wb') as f:
            pickle.dump(self.sum_rewards_episode, f)

        self.main_network.summary()
        self.main_network.save(path + file + '.h5')

    #Critical: Run/Evaluate DQN
    #Run operation
    def eval_dqn(self, path, file, steps=1, trials=1): #V
        """
        Evaluates trained DQN
        TODO: Define as a finite state machine
        Args:
            file:

        Returns:

        """
        loaded_model = keras.models.load_model(path + file + '.h5', custom_objects={'my_loss_fn': self.my_loss_fn})

        listAveDistances = []

        for iter in range(trials):
            print("Trial:", iter)
            launch_nodes(launch_file=LAUNCH_FILE, sleep=SLEEP)

            while self.curr_robot_pose is None or self.curr_object_pose is None:
                time.sleep(1)
            n = 0
            terminalState = False
            sumDistances = 0
            currentState = self.get_curr_state()

            while not terminalState:
                Qvalues = loaded_model.predict(currentState.reshape(1, self.state_dims))
                # select the action that gives the max Qvalue
                action = np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])
                action_result = self.make_action_request(action)
                elements = self.process_action_result(action_result)
                currentState = self.state_fcn(elements)
                terminalState = self.is_terminal(currentState)
                # sum the distance
                sumDistances += currentState[0]
                n += 1
                if n == steps:
                    terminalState = True
                rospy.sleep(1)
            ave_dist = sumDistances / n
            print("Average distance:", ave_dist)
            listAveDistances.append(ave_dist)
            kill_nodes(NODES_TO_KILL, SLEEP)
        print("Average distance across iterations:", listAveDistances)
        print("Average:", sum(listAveDistances) / trials)
        with open(path + file + '_eval.pkl', 'wb') as f:
            pickle.dump(listAveDistances, f)
        return listAveDistances

    def eval_random_policy(self, path, steps=1, trials=1): #REM
        """
        Policy of chasing after an object that does random actions
        Returns:

        """
        listAveDistances = []

        for iter in range(trials):
            print("Trial:", iter)
            launch_nodes(launch_file=LAUNCH_FILE, sleep=SLEEP)

            while self.curr_robot_pose is None or self.curr_object_pose is None:
                time.sleep(1)
            n = 0
            terminalState = False
            sumDistances = 0
            while not terminalState:
                action = np.random.choice(self.action_dims)
                action_result = self.make_action_request(action)
                elements = self.process_action_result(action_result)
                currentState = self.state_fcn(elements)
                terminalState = self.is_terminal(currentState)
                # sum the distance
                sumDistances += currentState[0]
                n += 1
                if n == steps:
                    terminalState = True
                rospy.sleep(1)
            ave_dist = sumDistances / n
            print("Average distance:", ave_dist)
            listAveDistances.append(ave_dist)
            kill_nodes(NODES_TO_KILL, SLEEP)
        print("Average distance across iterations:", listAveDistances)
        print("Average:", sum(listAveDistances) / trials)
        with open(path + 'random_eval.pkl', 'wb') as f:
            pickle.dump(listAveDistances, f)
        return listAveDistances

    def debug(self, msg):
        pu.log_msg('robot', self.robot_id, msg, self.debug_mode)

if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump') #What is this for? The filename to save results?
    dqn = DeepQLearning('training_dqn')
    # dqn.train_model(path, filename)
    dqn.eval_dqn(path, filename, steps=20, trials=10)
    dqn.eval_random_policy(path, steps=20, trials=10)  # Baseline policy

    """
    What are needed?
    1. The environment (saved placement of areas in the environment)
    2. The robot moving
    3. The training of the DQN for a given number of training episodes:
        > start of the simulation
        > collection of replay buffer
        > training of the network
        > reset the simulation for the next training episode
    4. The evaluation of the trained DQN
    """
















