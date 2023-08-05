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

os.chdir('/root/catkin_ws/src/results/int_preservation')
worlds = ['office', 'open']
nareas_list = [3]
decay_category = ['uniform', 'non_uniform']
dec_steps = [1, 3]
nplacements = 2
ntrials = 2
tframe = 10

#Deprecated
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

#TODO: Separate script for sampling of nodes
def sample_nodes_poses(world, nareas, nplacements):
    """
    Create a dictionary of sampled node poses (count=nareas) placed in the world randomly for nplacements.
    Each placement would be the value for this dict.
    :param worlds:
    :param nareas:
    :param nplacements:
    :return:
    """
    params = ['world:={}'.format(world), 'nareas:={}'.format(nareas), 'nplacements:={}'.format(nplacements)]
    launch_nodes('int_preservation', 'sample_areas.launch', params)

def open_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        f.close()

    return data
"""
TODO: Refinement of sampling of nodes
For each world: We launch each world once, then sample a number of areas for each narea, taking different placements for p times
"""

#Create file for the different placement of nodes in the world first
for w in worlds:
    for n in nareas_list:
        fileposes = '{}_n{}_sampled_nodes_poses_dict'.format(w, n)
        if os.path.exists(fileposes) is False:
            sample_nodes_poses(w, n, nplacements)  # TODO: This is imported from another script


#Run the experiment
for w in worlds:
    for n in nareas_list:
        # Open sampled nodes poses in the world if it exists, otherwise we sample
        fileposes = '{}_n{}_sampled_nodes_poses_dict'.format(w, n)

        #NOTE: We will open sampled_nodes_poses_dict in treebased_decision.py
        #We just need to supply it the filename
        #Inside treebased_decision.py, we loop through the proper placement, which is a key to the sampled_nodes_poses_dict
        #Therefore: the looped placement p should likewise be a parameter
        for p in range(nplacements):
            for d in decay_category:
                for k in dec_steps:
                    for i in range(ntrials):
                        fileresult = '{}_n{}_p{}_{}_k{}_{}'.format(w, n, p+1, d, k, i+1)
                        #TODO: Update the way we access the sampled nodes poses
                        params = ['world:={}'.format(w), 'nareas:={}'.format(n),
                                  'decay:={}'.format(d), 'dsteps:={}'.format(k),
                                  'tframe:={}'.format(tframe), 'placement:={}'.format(p+1),
                                  'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult)]
                        print("Launching...world: {}, nareas: {}, decay: {}, dsteps: {}, tframe: {}, placement: {}".format(w, n, d, k, tframe, p+1))
                        launch_nodes('int_preservation', 'mission.launch', params)
