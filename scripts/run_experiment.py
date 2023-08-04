#!/usr/bin/env python

"""
UPNEXT: Run this script
UPNEXT: Store the data in a rosbag

Runs the experiment through different loops

world: [office, open, cluttered] and charging station. DONE
nareas: [3, 6, 9]
decay_rates: uniform vs non-uniform: 1/3 in (0.0001, 0.0004], 1/3 in (0.0004, 0.0007], 1/3 in (0.0007, 0.001]
placement: a different seed in sampling the nodes in the Voronoi graph, repeated 3 times
decision step: 1, 3, 6, 9 (nareas=3, max 5mins, finished cleanly), 12 (not runnable, >1hr still running), 18 (not runnable, >3hrs)
trials: 5

for each world:
    for each nareas:
            #Here: Create different area launch file

        for each decay_rates:
            # If uniform: all areas have the same decay rates
            # Else: divide the decay rates per group of areas
            for each placement:
                # Seed for placement
                for each decision step:
                    for trials:
                        run trial: roslaunch with the correct parameters
                            > world, nareas, uniform/non-uniform, decision steps?
"""
import pickle
import os
from reset_simulation import *

os.chdir('/root/catkin_ws/src/int_preservation/results')
worlds = ['office', 'open']
nareas = [3, 6]
decay_category = ['uniform', 'non_uniform']
dec_steps = [1, 3]
nplacements = 2
ntrials = 2
tframe = 10

#Different placement of areas
# for w in worlds:
#     for n in nareas:
#         for p in range(nplacements):
#             seed = n*1000 + (p+1)*100 + p*10
#             fileareas = '{}_n{}_p{}'.format(w, n, p)
#             params = ['world:={}'.format(w), 'nareas:={}'.format(n),
#                       'fileareas:={}'.format(fileareas), 'seed:={}'.format(seed)]
#             print("Sampling areas...world: {}, nareas: {}, filename: {}, seed: {}".format(w, n, fileareas, seed))
#             launch_nodes('int_preservation', 'sample_areas.launch', params)
#             kill_nodes(sleep=10)

def sample_nodes_poses(worlds, nareas, nplacements):
    for w in worlds:
        for n in nareas:
            params = ['world:={}'.format(w), 'nareas:={}'.format(n), 'nplacements:={}'.format(nplacements)]
            launch_nodes('int_preservation', 'sample_areas.launch', params)
            kill_nodes(sleep=10)

def open_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        f.close()

    return data
"""
TODO: Refinement of sampling of nodes
For each world: We launch each world once, then sample a number of areas for each narea, taking different placements for p times
"""

#
for w in worlds:
    for n in nareas:
        # Open sampled nodes poses in the world if it exists, otherwise we sample
        filename = '{}_n{}_p{}_sampled_nodes_poses_dict'.format(w, n, nplacements)
        if os.path.exists(filename) is False:
            sample_nodes_poses(w, n, nplacements)

        sampled_nodes_poses_dict = open_data(filename)

        for p in range(nplacements):
            sampled_nodes_poses = sampled_nodes_poses_dict['n{}_p{}'.format(n, p+1)]
            for d in decay_category:
                for k in dec_steps:
                    for i in range(ntrials):
                        filedata = '{}_n{}_p{}_{}_k{}_{}'.format(w, n, p+1, d, k, i+1)
                        #TODO: Update the way we access the sampled nodes poses
                        params = ['world:={}'.format(w), 'nareas:={}'.format(n),
                                  'decay:={}'.format(d), 'dsteps:={}'.format(k),
                                  'tframe:={}'.format(tframe),
                                  'fileareas:={}'.format(fileareas), 'filedata:={}'.format(filedata)]
                        print("Launching...world: {}, nareas: {}, decay: {}, dsteps: {}, tframe: {}, placement: {}".format(w, n, d, k, tframe, fileareas))
                        launch_nodes('int_preservation', 'mission.launch', params)
                        #Ensure we reached the end and saved the desired length of rosbag. perhaps t_operation?
                        # kill_nodes(sleep=5)
