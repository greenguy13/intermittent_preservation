#!/usr/bin/env python
# Author: Amel Docena

"""
Author: Amel Docena

References:
1. An Overview of the Action Space for Deep Reinforcement Learning., ACML
2. Deep Learning Reinforcement Tutorial: Deep Q Network (DQN), Aleksander Haber
"""
import os
import pickle
import numpy as np
import math
import random
import rospy
import tf
from geometry_msgs.msg import Twist, Pose, PoseStamped
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from collections import deque
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error
from reset_simulation import *
from follower_dqn.msg import make_actionAction, make_actionGoal
import actionlib
import time
import keras
#from transform import Transform

SAFE_DIST = 0.50
COLLISION_DIST = 0.40

MODEL_PARAMS = {'GAMMA': 0.9,
                'EPS': 0.25,
                'NUM_EPISODES': 60,
                'STATE_DIMS': 5,
                'ACTION_DIMS': 8,
                'REPLAY_BUFFER_SIZE': 30,
                'BATCH_REPLAY_BUFFER_SIZE': 10,
                'UPDATE_TARGET_NETWORK_PERIOD': 15,
                'N_STEPS': 10}

LAUNCH_FILE = 'simulation.launch'
PATH = '~/catkin_ws/src/cs81-fp/'
WORLD = 'PA1.world'
NODES_TO_KILL = ['random_walk', 'finder', 'robot_motion', 'stage_ros']
SLEEP = 5

class DeepQLearning:
    def __init__(self, node_name, model_params=MODEL_PARAMS, safe_dist=SAFE_DIST, collision_dist=COLLISION_DIST):
        rospy.init_node(node_name)
        rospy.Subscriber('/robot_0/odom', Odometry, self._set_robot_pose_cb, queue_size=1)
        rospy.Subscriber('/robot_0/d_front', Float32, self._set_d_front_cb, queue_size=1)
        rospy.Subscriber('/robot_0/d_left', Float32, self._set_d_left_cb, queue_size=1)
        rospy.Subscriber('/robot_0/d_right', Float32, self._set_d_right_cb, queue_size=1)
        rospy.Subscriber('/stalked', PoseStamped, self._set_object_pose_cb, queue_size=1)
        #self.transform = Transform() #Encountering problems in TF transformer publisher when resetting simulation. We thus provide the transformation matrix of odom wrt map based on world
        #rospy.sleep(2)
        #self.m_T_o = self.transform.get_transformation_matrix('map', '/robot_0/odom') #Transformation matrix of robot odom wrt map frame
        self.m_T_o = np.array([[1, 0, 0, 5],
                                [0, 1, 0, 5],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        self._make_action_client = actionlib.SimpleActionClient('/make_action_server', make_actionAction)

        self.gamma = model_params['GAMMA']
        self.epsilon = model_params['EPS']
        self.numberEpisodes = model_params['NUM_EPISODES'] #Number of training episodes. How is this related to Experienced Replay?

        # state dimension
        self.stateDimension = model_params['STATE_DIMS'] #No. of states
        # action dimension
        self.actionDimension = model_params['ACTION_DIMS'] #No. of actions
        # this is the maximum size of the replay buffer
        self.replayBufferSize = model_params['REPLAY_BUFFER_SIZE'] #Replay buffer size. We are training 1000 episodes until the terminal state is achieved, and then store this as experienced replay
        # this is the size of the training batch that is randomly sampled from the replay buffer
        self.batchReplayBufferSize = model_params['BATCH_REPLAY_BUFFER_SIZE'] #Batch training

        # number of training episodes it takes to update the target network parameters
        # that is, every updateTargetNetworkPeriod we update the target network parameters
        self.updateTargetNetworkPeriod = model_params['UPDATE_TARGET_NETWORK_PERIOD']
        self.n_steps = model_params['N_STEPS']

        # this is the counter for updating the target network 
        # if this counter exceeds (updateTargetNetworkPeriod-1) we update the network 
        # parameters and reset the counter to zero, this process is repeated until the end of the training process
        self.counterUpdateTargetNetwork = 0

        # this sum is used to store the sum of rewards obtained during each training episode
        self.sumRewardsEpisode = []
        self.safe_dist = safe_dist #Safe distance from obstacle; used for computing reward
        self.collision_dist = collision_dist #Collision distance. If collided, terminal state

        # replay buffer
        self.replayBuffer = deque(maxlen=self.replayBufferSize)

        # this is the main network
        # create network
        self.mainNetwork = self.createNetwork()

        # this is the target network
        # create network
        self.targetNetwork = self.createNetwork()

        # copy the initial weights to targetNetwork
        self.targetNetwork.set_weights(self.mainNetwork.get_weights())

        # this list is used in the cost function to select certain entries of the 
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend = []

        self.curr_robot_pose = None
        self.curr_object_pose = None
        self.d_front, self.d_left, self.d_right = None, None, None

    def _set_object_pose_cb(self, msg):
        #Sets followed object pose
        x, y = msg.pose.position.x, msg.pose.position.y
        orientation = msg.pose.orientation
        _, _, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.curr_object_pose = (x, y, theta)

    def _set_robot_pose_cb(self, msg):
        #Sets curr robot pose
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.curr_robot_pose = (x, y, theta)

    def transform_2d_point(self, transformation_matrix, point):
        x, y = point
        transformed_point = np.matmul(transformation_matrix, np.array([x, y, 0, 1], dtype='float32'))
        return (transformed_point[0], transformed_point[1])

    def _set_d_front_cb(self, msg):
        self.d_front = msg.data

    def _set_d_left_cb(self, msg):
        self.d_left = msg.data

    def _set_d_right_cb(self, msg):
        self.d_right = msg.data

    def _make_action_client_fb(self, msg):
        """
        Feedback for action client to make robot move
        Returns:

        """
        print("Feedback:", msg)



    def make_action_request(self, action):
        """
        Action request to robot to take action
        Args:
            action:

        Returns:

        """
        goal = make_actionGoal()
        goal.action_idx = action
        self._make_action_client.wait_for_server()
        self._make_action_client.send_goal(goal, feedback_cb=self._make_action_client_fb)
        self._make_action_client.wait_for_result()
        result = self._make_action_client.get_result()
        return result.states


    def get_robot_pose(self):
        """
        Returns robot pose
        Returns:

        """
        return self.curr_robot_pose

    def get_object_pose(self):
        """
        Returns object pose
        Returns:

        """
        return self.curr_object_pose

    def get_curr_state(self):
        """
        Gets current state
        Returns:

        """
        robot_pose = self.get_robot_pose()
        object_pose = self.get_object_pose()
        elements = robot_pose, object_pose, self.d_front, self.d_left, self.d_right
        state = self.state_fcn(elements)
        return state

    def my_loss_fn(self, y_true, y_pred):

        s1, s2 = y_true.shape
        indices = np.zeros(shape=(s1, 2))
        indices[:, 0] = np.arange(s1)
        indices[:, 1] = self.actionsAppend
        loss = mean_squared_error(gather_nd(y_true, indices=indices.astype(int)),
                                  gather_nd(y_pred, indices=indices.astype(int)))
        # print(loss)
        return loss
    
    # create a neural network
    def createNetwork(self):
        model = Sequential()
        model.add(Dense(20, input_dim=self.stateDimension, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.actionDimension, activation='linear'))
        model.compile(optimizer=RMSprop(), loss=self.my_loss_fn, metrics=['accuracy'])
        return model

    def is_terminal(self, state):
        """
        Determines whether a state is terminal
        Returns:

        """
        robot_dist_obj, _, d_front, d_left, d_right = state
        if robot_dist_obj < self.collision_dist or d_front < self.collision_dist or d_left < self.collision_dist or d_right < self.collision_dist:
            return True
        return False

    def process_action_result(self, action_result):
        """
        Processes action_result into meaningful elements for state function
        Args:
            action_result:

        Returns:

        """
        robot_pose = action_result[0], action_result[1], action_result[2]
        object_pose = action_result[3], action_result[4], action_result[5]
        d_front, d_left, d_right = action_result[6], action_result[7], action_result[8]
        processed = robot_pose, object_pose, d_front, d_left, d_right
        return processed

    def state_fcn(self, elements):
        """
        Computes the state given the elements:
            Distance and orientation of object from robot wrt map reference frame
            Distance of perceived front, left, right obstacles
        Returns:

        """
        robot_pose, object_pose, d_front, d_left, d_right = elements
        map_robot_pose = self.transform_2d_point(self.m_T_o, (robot_pose[0], robot_pose[1]))
        map_object_pose = self.transform_2d_point(self.m_T_o, (object_pose[0], object_pose[1]))
        robot_dist_obj = math.sqrt((map_object_pose[0]-map_robot_pose[0])**2 + (map_object_pose[1]-map_robot_pose[1])**2)
        robot_orient_obj = math.atan2(map_object_pose[1]-map_robot_pose[1], map_object_pose[0]-map_robot_pose[0])
        state = (robot_dist_obj, robot_orient_obj, d_front, d_left, d_right)
        assert len(state) == self.stateDimension, "Invalid! Incongruent state dimension and length of state function return"
        return np.array(state)

    def reward_fcn(self, state):
        """
        Reward function of a given state
        Args:
            state:

        Returns:

        """
        robot_dist_obj, robot_orient_obj, d_front, d_left, d_right = state

        reward = -0.10 #Penalty for a single move; for move efficiency
        #The distance toward the object. As the robot approaches safe distance, reward increases
        if robot_dist_obj >= self.safe_dist:
            reward += ((9+self.safe_dist)-self.safe_dist) - robot_dist_obj**2

        #We penalize further if within collision distance with object or wall
        if (robot_dist_obj < self.collision_dist) or (d_front < self.collision_dist or d_left < self.collision_dist or d_right < self.collision_dist):
            reward += -20

        return reward

    def trainingEpisodes(self):
        """
        Training episodes

        Returns:

        """

        for indexEpisode in range(self.numberEpisodes):
            rewardsEpisode = []
            print("Simulating episode {}".format(indexEpisode))

            #Start simulation
            launch_nodes(launch_file=LAUNCH_FILE, sleep=SLEEP)
            while self.curr_robot_pose is None or self.curr_object_pose is None or self.d_front is None or self.d_left is None or self.d_right is None:
                print("Topics not yet registered. Sleeping...")
                time.sleep(1)
            print("Robot and object poses registered!")
            currentState = self.get_curr_state()

            n_steps = 0
            terminalState = False
            while not terminalState:
                # select an action on the basis of the current state, denoted by currentState
                # then get subsequent, state, reward, and bool whether a terminal state
                action = self.selectAction(currentState, indexEpisode)
                print("Action request:", action)
                action_result = self.make_action_request(action)
                elements = self.process_action_result(action_result)
                nextState = self.state_fcn(elements)
                reward = self.reward_fcn(nextState)
                terminalState = self.is_terminal(nextState)
                print("Next State: {}, {}, {}. Reward: {}, {}. Terminal state: {}, {}".format(type(nextState), nextState.shape, nextState,
                                                                                          type(reward), reward,
                                                                                          type(terminalState),
                                                                                          terminalState))
                rewardsEpisode.append(reward)

                # add current state, action, reward, next state, and terminal flag to the replay buffer
                self.replayBuffer.append((currentState, action, reward, nextState, terminalState))

                # train network
                self.trainNetwork()

                # set the current state for the next step
                currentState = nextState
                n_steps += 1
                if n_steps == self.n_steps:
                    print("Collected enough datapoints. Terminating...")
                    terminalState = True
            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))
            kill_nodes(NODES_TO_KILL, SLEEP)

        return True

    
    def selectAction(self, state, index):
        """
        Selects an action on the basis of the current state.
        Implements epsilon-greedy approach
        Args:
            state: state for which to compute the action
            index: index of the current episode
        """

        # first index episodes we select completely random actions to have enough exploration
        if index < 30:
            return np.random.choice(self.actionDimension)

        # epsilon greedy approach
        randomNumber = np.random.random()

        # after index episodes, we slowly start to decrease the epsilon parameter
        if index > 40:
            self.epsilon = 0.955 * self.epsilon

        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionDimension)

            # otherwise, we are selecting greedy actions
        else:
            Qvalues = self.mainNetwork.predict(state.reshape(1, self.stateDimension))
            return np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0]) #we choose the max

    def trainNetwork(self):

        # if the replay buffer has at least batchReplayBufferSize elements,
        # then train the model 
        # otherwise wait until the size of the elements exceeds batchReplayBufferSize
        if (len(self.replayBuffer) > self.batchReplayBufferSize):

            # sample a batch from the replay buffer
            randomSampleBatch = random.sample(self.replayBuffer, self.batchReplayBufferSize)

            # get current state batch and next state batch as inputs for prediction
            currentStateBatch = np.zeros(shape=(self.batchReplayBufferSize, self.stateDimension))
            nextStateBatch = np.zeros(shape=(self.batchReplayBufferSize, self.stateDimension))

            for index, tupleS in enumerate(randomSampleBatch):
                currentStateBatch[index, :] = tupleS[0]
                nextStateBatch[index, :] = tupleS[3]

            # here, use the target network to predict Q-values 
            QnextStateTargetNetwork = self.targetNetwork.predict(nextStateBatch)
            # here, use the main network to predict Q-values 
            QcurrentStateMainNetwork = self.mainNetwork.predict(currentStateBatch)

            # form batches for training
            # input for training
            inputNetwork = currentStateBatch
            # output for training
            outputNetwork = np.zeros(shape=(self.batchReplayBufferSize, self.actionDimension))

            # list of actions that are selected from the batch
            self.actionsAppend = []
            for index, (currentState, action, reward, nextState, terminated) in enumerate(randomSampleBatch):
                # if the next state is the terminal state
                if terminated:
                    y = reward #Q-value pred for terminal state
                else:
                    y = reward + self.gamma * np.max(QnextStateTargetNetwork[index]) #Q-value prediction

                self.actionsAppend.append(action)

                outputNetwork[index] = QcurrentStateMainNetwork[index]
                outputNetwork[index, action] = y

            # train the network
            self.mainNetwork.fit(inputNetwork, outputNetwork, batch_size=self.batchReplayBufferSize, verbose=0, epochs=100)

            # after updateTargetNetworkPeriod training sessions, update the coefficients of the target network
            # increase the counter for training the target network
            self.counterUpdateTargetNetwork += 1
            if (self.counterUpdateTargetNetwork > (self.updateTargetNetworkPeriod - 1)):
                # copy the weights to targetNetwork
                self.targetNetwork.set_weights(self.mainNetwork.get_weights())
                print("Target network updated!")
                print("Counter value {}".format(self.counterUpdateTargetNetwork))
                # reset the counter
                self.counterUpdateTargetNetwork = 0

    def train_model(self, path, file):
        """
        Trains the robot to follow object via DQN
        Returns:

        """
        done = False
        while not rospy.is_shutdown() and done is not True:
            done = self.trainingEpisodes()

        print("Sum of rewards across episodes:", self.sumRewardsEpisode)
        with open(path+file+'.pkl', 'wb') as f:
            pickle.dump(self.sumRewardsEpisode, f)

        self.mainNetwork.summary()
        self.mainNetwork.save(path+file+'.h5')

    def eval_dqn(self, path, file, steps=1, trials=1):
        """
        Evaluates trained DQN
        Args:
            file:

        Returns:

        """
        loaded_model = keras.models.load_model(path+file+'.h5', custom_objects={'my_loss_fn': self.my_loss_fn})

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
                Qvalues = loaded_model.predict(currentState.reshape(1, self.stateDimension))
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

    def eval_random_policy(self, path, steps=1, trials=1):
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
                action = np.random.choice(self.actionDimension)
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
            ave_dist = sumDistances/n
            print("Average distance:", ave_dist)
            listAveDistances.append(ave_dist)
            kill_nodes(NODES_TO_KILL, SLEEP)
        print("Average distance across iterations:", listAveDistances)
        print("Average:", sum(listAveDistances)/trials)
        with open(path + 'random_eval.pkl', 'wb') as f:
            pickle.dump(listAveDistances, f)
        return listAveDistances


if __name__ == '__main__':
    path = os.getcwd() + '/src/cs81-fp/'
    filename = 'dqn_exp2'
    dqn = DeepQLearning('training_dqn')
    #dqn.train_model(path, filename)
    dqn.eval_dqn(path, filename, steps=20, trials=10)
    dqn.eval_random_policy(path, steps=20, trials=10) #Baseline policy
















