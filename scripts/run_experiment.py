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
def batch_experiments(method, worlds, nareas_list, nplacements, decay_category, learn_decay_types, tframe, dec_steps=None, ntrials=1, sample_nodes=False, save=False, exp_id=None):
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
                    for l in learn_decay_types:
                        if method == 'treebased_decision':
                            for k in dec_steps:
                                for i in range(ntrials):
                                    fileresult = '{}_{}_n{}_p{}_{}_k{}_{}_exp{}'.format(method, w, n, p+1, d, k, i+1, exp_id)
                                    logfile = fileresult + '.txt'
                                    params = ['method:={}'.format(method),
                                              'world:={}'.format(w), 'nareas:={}'.format(n),
                                              'decay:={}'.format(d), 'learndecay:={}'.format(l),
                                              'dsteps:={}'.format(k),
                                              'tframe:={}'.format(tframe), 'placement:={}'.format(p+1),
                                              'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult),
                                              'save:={}'.format(save)]
                                    print("Launching...method: {}, world: {}, nareas: {}, decay: {}, learn: {}, dsteps: {}, tframe: {}, placement: {}, save: {}, exp: {}".format(method, w, n, d, l, k, tframe, p+1, save, exp_id))
                                    launch_nodes('int_preservation', 'mission.launch', params, logfile)

                        elif method == 'random_decision':
                            for i in range(ntrials):
                                fileresult = '{}_{}_n{}_p{}_{}_{}_exp{}'.format(method, w, n, p + 1, d, i + 1, exp_id)
                                logfile = fileresult + '.txt'
                                params = ['method:={}'.format(method), 'world:={}'.format(w),
                                          'nareas:={}'.format(n), 'decay:={}'.format(d),
                                          'tframe:={}'.format(tframe), 'placement:={}'.format(p + 1),
                                          'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult),
                                          'save:={}'.format(save)]
                                print("Launching...method: {}, world: {}, nareas: {}, decay: {}, tframe: {}, placement: {}, save: {}, exp: {}".format(method, w, n, d, tframe, p+1, save, exp_id))
                                launch_nodes('int_preservation', 'mission.launch', params, logfile)

def run_experiment(method, world, nareas, placement, decay, learn_decay, tframe, dec_steps=1, ntrials=1, save=False, exp_id=None):
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
    if method == 'treebased_decision' or method == 'heuristic_decision':
        for i in range(ntrials):
            fileresult = '{}_{}_n{}_p{}_{}_k{}_{}_exp{}'.format(method, world, nareas, placement, decay, dec_steps, i + 1, exp_id)
            logfile = fileresult + '.txt'
            params = ['method:={}'.format(method),
                      'world:={}'.format(world), 'nareas:={}'.format(nareas),
                      'decay:={}'.format(decay), 'learndecay:={}'.format(learn_decay),
                      'dsteps:={}'.format(dec_steps),
                      'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                      'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult), 'save:={}'.format(save)]
            print("Launching...method: {}, world: {}, nareas: {}, decay: {}, learn: {}, dsteps: {}, tframe: {}, placement: {}, trial: {}, save: {}, exp_id: {}".format(
                    method, world, nareas, decay, learn_decay, dec_steps, tframe, placement, i+1, save, exp_id))
            launch_nodes('int_preservation', 'mission.launch', params, logfile)

    elif method == 'random_decision':
        for i in range(ntrials):
            fileresult = '{}_{}_n{}_p{}_{}_{}_exp{}'.format(method, world, nareas, placement, decay, i + 1, exp_id)
            logfile = fileresult + '.txt'
            params = ['method:={}'.format(method), 'world:={}'.format(world),
                      'nareas:={}'.format(nareas), 'decay:={}'.format(decay),
                      'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                      'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult), 'save:={}'.format(save)]
            print("Launching...method: {}, world: {}, nareas: {}, decay: {}, tframe: {}, placement: {}, trial:{}, save: {}, exp_id: {}".format(
                method, world, nareas, decay, tframe, placement, i+1, save, exp_id))
            launch_nodes('int_preservation', 'mission.launch', params, logfile)

if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')

    #Sample node poses
    # worlds = ['office', 'open', 'cluttered']
    # nareas_list = [4, 8, 12]
    # nplacements = 1000
    # batch_sample_nodes_poses(worlds, nareas_list, nplacements)

    #Office
    run_experiment(method='treebased_decision', world='office', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True, exp_id=40)

    run_experiment(method='heuristic_decision', world='office', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=2, ntrials=3, save=True, exp_id=120)

    run_experiment(method='heuristic_decision', world='office', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True, exp_id=140)

    run_experiment(method='heuristic_decision', world='office', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=6, ntrials=3, save=True, exp_id=160)

    run_experiment(method='heuristic_decision', world='office', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=8, ntrials=3, save=True, exp_id=180)

    run_experiment(method='heuristic_decision', world='office', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=10, ntrials=3, save=True, exp_id=200)

    run_experiment(method='heuristic_decision', world='office', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=12, ntrials=3, save=True, exp_id=220)

    #Cluttered
    run_experiment(method='treebased_decision', world='cluttered', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True, exp_id=41)

    run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=2, ntrials=3, save=True, exp_id=121)

    run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True, exp_id=141)

    run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=6, ntrials=3, save=True, exp_id=161)

    run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=8, ntrials=3, save=True, exp_id=181)

    run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=10, ntrials=3, save=True, exp_id=201)

    run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=12, ntrials=3, save=True, exp_id=221)

    #Open
    run_experiment(method='treebased_decision', world='open', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True, exp_id=42)

    run_experiment(method='heuristic_decision', world='open', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=2, ntrials=3, save=True, exp_id=122)

    run_experiment(method='heuristic_decision', world='open', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True, exp_id=142)

    run_experiment(method='heuristic_decision', world='open', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=6, ntrials=3, save=True, exp_id=162)

    run_experiment(method='heuristic_decision', world='open', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=8, ntrials=3, save=True, exp_id=182)

    run_experiment(method='heuristic_decision', world='open', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=10, ntrials=3, save=True, exp_id=202)

    run_experiment(method='heuristic_decision', world='open', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=2100, dec_steps=12, ntrials=3, save=True, exp_id=222)