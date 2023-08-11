#!/usr/bin/env python

import pickle
import os
from reset_simulation import *


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



#Run the experiment
def batch_experiments(method, worlds, nareas_list, nplacements, decay_category, tframe, dec_steps=None, ntrials=1):

    # Create file for the different placement of nodes in the world first
    for w in worlds:
        for n in nareas_list:
            fileposes = '{}_n{}_sampled_nodes_poses_dict.pkl'.format(w, n)
            if os.path.exists(fileposes) is False:
                sample_nodes_poses(w, n, nplacements)

    # Run experiments
    for w in worlds:
        for n in nareas_list:
            fileposes = '{}_n{}_sampled_nodes_poses_dict'.format(w, n)
            for p in range(nplacements):
                for d in decay_category:
                    if method == 'treebased_decision':
                        for k in dec_steps:
                            for i in range(ntrials):
                                fileresult = '{}_{}_n{}_p{}_{}_k{}_{}'.format(method, w, n, p+1, d, k, i+1)
                                params = ['method:={}'.format(method),
                                          'world:={}'.format(w), 'nareas:={}'.format(n),
                                          'decay:={}'.format(d), 'dsteps:={}'.format(k),
                                          'tframe:={}'.format(tframe), 'placement:={}'.format(p+1),
                                          'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult)]
                                print("Launching...method: {}, world: {}, nareas: {}, decay: {}, dsteps: {}, tframe: {}, placement: {}".format(method, w, n, d, k, tframe, p+1))
                                launch_nodes('int_preservation', 'mission.launch', params)

                    elif method == 'random_decision':
                        for i in range(ntrials):
                            fileresult = '{}_{}_n{}_p{}_{}_{}'.format(method, w, n, p + 1, d, i + 1)
                            params = ['method:={}'.format(method), 'world:={}'.format(w),
                                      'nareas:={}'.format(n), 'decay:={}'.format(d),
                                      'tframe:={}'.format(tframe), 'placement:={}'.format(p + 1),
                                      'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult)]
                            print("Launching...method: {}, world: {}, nareas: {}, decay: {}, tframe: {}, placement: {}".format(method, w, n, d, tframe, p+1))
                            launch_nodes('int_preservation', 'mission.launch', params)


if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    worlds = ['office']
    nareas_list = [3]
    decay_category = ['non_uniform']
    dec_steps = [3]
    nplacements = 4
    ntrials = 5
    tframe = 2100

    #Random decision making
    # batch_experiments(method='random_decision', worlds=worlds, nareas_list=nareas_list, nplacements=nplacements, decay_category=decay_category, tframe=tframe, ntrials=ntrials, dec_steps=None)

    #Treebased decision
    batch_experiments(method='treebased_decision', worlds=worlds, nareas_list=nareas_list, nplacements=nplacements, decay_category=decay_category, tframe=tframe, ntrials=ntrials, dec_steps=dec_steps)

