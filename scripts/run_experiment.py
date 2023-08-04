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

from reset_simulation import *

worlds = ['office', 'open']
nareas = [3, 6]
decay_category = ['uniform', 'non_uniform']
dec_steps = [1, 3]
nplacements = 2
ntrials = 2
tframe = 10


for w in worlds:
    for n in nareas:
        trial = 0
        for p in range(nplacements):
            seed = n*1000 + (p+1)*100 + p*10
            #Select nodes from Voronoi graph as areas to preserve, then store using a unique code/filename
            #So if this is the case, then the script will merely be subscribing, sample nodes, then store the nodes
            #Makes sense: After all there's really no need to show the Voronoi graph
            for d in decay_category:
                for k in dec_steps:
                    for i in range(ntrials):
                        trial += 1
                        params = ['world:={}'.format(w), 'nareas:={}'.format(n),
                                  'decay:={}'.format(d), 'dsteps:={}'.format(k),
                                  'tframe:={}'.format(tframe), 'seed:={}'.format(seed),
                                  'trial:={}'.format(trial)] #TODO: Insert the filename for the sampled nodes
                        print("Launching...world: {}, nareas: {}, decay: {}, dsteps: {}, tframe: {}, seed: {}, trial:{}".format(w, n, d, k, tframe, seed, trial))
                        launch_nodes('int_preservation', 'mission.launch', params)
                        #Ensure we reached the end and saved the desired length of rosbag. perhaps t_operation?
                        # kill_nodes(sleep=5)
