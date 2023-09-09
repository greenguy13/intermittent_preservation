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

def batch_sample_nodes_poses(worlds, nareas_list, nplacements):
    for w in worlds:
        for n in nareas_list:
            fileposes = '{}_n{}_sampled_nodes_poses_dict.pkl'.format(w, n)
            if os.path.exists(fileposes) is False:
                sample_nodes_poses(w, n, nplacements)



#Run the experiment
def batch_experiments(method, worlds, nareas_list, nplacements, decay_category, tframe, dec_steps=None, ntrials=1, sample_nodes=False, save=False):
    """
    Runs a batch of experiments
    :param method:
    :param worlds:
    :param nareas_list:
    :param nplacements:
    :param decay_category:
    :param tframe:
    :param dec_steps:
    :param ntrials:
    :return:
    """
    # Create file for the different placement of nodes in the world first
    if sample_nodes:
        batch_sample_nodes_poses(worlds, nareas_list, nplacements)

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
                                logfile = fileresult + '.txt'
                                params = ['method:={}'.format(method),
                                          'world:={}'.format(w), 'nareas:={}'.format(n),
                                          'decay:={}'.format(d), 'dsteps:={}'.format(k),
                                          'tframe:={}'.format(tframe), 'placement:={}'.format(p+1),
                                          'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult),
                                          'save:={}'.format(save)]
                                print("Launching...method: {}, world: {}, nareas: {}, decay: {}, dsteps: {}, tframe: {}, placement: {}, save: {}".format(method, w, n, d, k, tframe, p+1, save))
                                launch_nodes('int_preservation', 'mission.launch', params, logfile)

                    elif method == 'random_decision':
                        for i in range(ntrials):
                            fileresult = '{}_{}_n{}_p{}_{}_{}'.format(method, w, n, p + 1, d, i + 1)
                            logfile = fileresult + '.txt'
                            params = ['method:={}'.format(method), 'world:={}'.format(w),
                                      'nareas:={}'.format(n), 'decay:={}'.format(d),
                                      'tframe:={}'.format(tframe), 'placement:={}'.format(p + 1),
                                      'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult),
                                      'save:={}'.format(save)]
                            print("Launching...method: {}, world: {}, nareas: {}, decay: {}, tframe: {}, placement: {}, save: {}".format(method, w, n, d, tframe, p+1, save))
                            launch_nodes('int_preservation', 'mission.launch', params, logfile)

def run_experiment(method, world, nareas, placement, decay, tframe, dec_steps=1, ntrials=1, save=False):
    """
    Runs a single experiment
    :param method:
    :param world:
    :param nareas:
    :param placement:
    :param decay:
    :param tframe:
    :param dec_steps:
    :param ntrials:
    :return:
    """
    fileposes = '{}_n{}_sampled_nodes_poses_dict'.format(world, nareas)
    if method == 'treebased_decision':
        for i in range(ntrials):
            fileresult = '{}_{}_n{}_p{}_{}_k{}_{}'.format(method, world, nareas, placement, decay, dec_steps, i + 1)
            logfile = fileresult + '.txt'
            params = ['method:={}'.format(method),
                      'world:={}'.format(world), 'nareas:={}'.format(nareas),
                      'decay:={}'.format(decay), 'dsteps:={}'.format(dec_steps),
                      'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                      'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult), 'save:={}'.format(save)]
            print("Launching...method: {}, world: {}, nareas: {}, decay: {}, dsteps: {}, tframe: {}, placement: {}, trial: {}, save: {}".format(
                    method, world, nareas, decay, dec_steps, tframe, placement, i+1, save))
            launch_nodes('int_preservation', 'mission.launch', params, logfile)

    elif method == 'random_decision':
        for i in range(ntrials):
            fileresult = '{}_{}_n{}_p{}_{}_{}'.format(method, world, nareas, placement, decay, i + 1)
            logfile = fileresult + '.txt'
            params = ['method:={}'.format(method), 'world:={}'.format(world),
                      'nareas:={}'.format(nareas), 'decay:={}'.format(decay),
                      'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                      'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult), 'save:={}'.format(save)]
            print("Launching...method: {}, world: {}, nareas: {}, decay: {}, tframe: {}, placement: {}, save: {}".format(
                method, world, nareas, decay, tframe, placement, save))
            launch_nodes('int_preservation', 'mission.launch', params, logfile)

if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    # worlds = ['office']
    # nareas_list = [3]
    # decay_category = ['non_uniform']
    # dec_steps = [3]
    # nplacements = 5
    # ntrials = 5
    # tframe = 2100

    #Random decision making
    # batch_experiments(method='random_decision', worlds=worlds, nareas_list=nareas_list, nplacements=nplacements, decay_category=decay_category, tframe=tframe, ntrials=ntrials, dec_steps=None)

    #Treebased decision
    # batch_experiments(method='treebased_decision', worlds=worlds, nareas_list=nareas_list, nplacements=nplacements, decay_category=decay_category, tframe=tframe, ntrials=ntrials, dec_steps=dec_steps)
    # run_experiment(method='random_decision', world='office', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='random_decision', world='office', nareas=3, placement=2, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='random_decision', world='office', nareas=3, placement=3, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='random_decision', world='office', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='random_decision', world='office', nareas=6, placement=2, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='random_decision', world='office', nareas=6, placement=3, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='random_decision', world='office', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='random_decision', world='office', nareas=9, placement=2, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='random_decision', world='office', nareas=9, placement=3, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)


    # #Sample node poses
    # #nareas = 6, 9
    # worlds = ['office', 'open', 'cluttered']
    # nareas_list = [6, 9]
    # nplacements = 5
    # batch_sample_nodes_poses(worlds, nareas_list, nplacements)


    #To-Run
    #Office, placement = 1
    #nareas = 3
    # run_experiment(method='random_decision', world='office', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    run_experiment(method='treebased_decision', world='office', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    run_experiment(method='treebased_decision', world='office', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)
    run_experiment(method='treebased_decision', world='office', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=6, ntrials=5, save=True)
    run_experiment(method='treebased_decision', world='office', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=9, ntrials=5, save=True)


    #nareas = 6
    # run_experiment(method='random_decision', world='office', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    run_experiment(method='treebased_decision', world='office', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    run_experiment(method='treebased_decision', world='office', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)

    #nareas =9
    # run_experiment(method='random_decision', world='office', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    run_experiment(method='treebased_decision', world='office', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    run_experiment(method='treebased_decision', world='office', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)

    #Open
    #nareas = 3
    # run_experiment(method='random_decision', world='open', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='open', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='open', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='open', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=6, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='open', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=9, ntrials=5, save=True)


    #nareas = 6
    # run_experiment(method='random_decision', world='open', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='open', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='open', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)

    #nareas =9
    # run_experiment(method='random_decision', world='open', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='open', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='open', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)

    #Cluttered
    #nareas = 3
    # run_experiment(method='random_decision', world='cluttered', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='cluttered', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='cluttered', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='cluttered', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=6, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='cluttered', nareas=3, placement=1, decay='non_uniform', tframe=2100, dec_steps=9, ntrials=5, save=True)


    #nareas = 6
    # run_experiment(method='random_decision', world='cluttered', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='cluttered', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='cluttered', nareas=6, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)

    #nareas =9
    # run_experiment(method='random_decision', world='cluttered', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=None, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='cluttered', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=1, ntrials=5, save=True)
    # run_experiment(method='treebased_decision', world='cluttered', nareas=9, placement=1, decay='non_uniform', tframe=2100, dec_steps=3, ntrials=5, save=True)
