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

def run_experiment(method, world, nareas, placement, decay, learn_decay, tframe, dec_steps=1, ntrials=1, save=False):
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
    if method == 'treebased_decision' or method == 'heuristic_decision' or method == 'dynamic_programming' or method == 'replay_decisions':
        for i in range(ntrials):
            fileresult = '{}_{}_n{}_p{}_{}_k{}_{}'.format(method, world, nareas, placement, decay, dec_steps, i + 1)
            logfile = fileresult + '.txt'
            params = ['method:={}'.format(method),
                      'world:={}'.format(world), 'nareas:={}'.format(nareas),
                      'decay:={}'.format(decay), 'learndecay:={}'.format(learn_decay),
                      'dsteps:={}'.format(dec_steps),
                      'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                      'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult), 'save:={}'.format(save)]
            print("Launching...method: {}, world: {}, nareas: {}, decay: {}, learn: {}, dsteps: {}, tframe: {}, placement: {}, trial: {}, save: {}".format(
                    method, world, nareas, decay, learn_decay, dec_steps, tframe, placement, i+1, save))
            launch_nodes('int_preservation', 'mission.launch', params, logfile)

    elif method == 'random_decision':
        for i in range(ntrials):
            fileresult = '{}_{}_n{}_p{}_{}_{}'.format(method, world, nareas, placement, decay, i + 1)
            logfile = fileresult + '.txt'
            params = ['method:={}'.format(method), 'world:={}'.format(world),
                      'nareas:={}'.format(nareas), 'decay:={}'.format(decay),
                      'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                      'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult), 'save:={}'.format(save)]
            print("Launching...method: {}, world: {}, nareas: {}, decay: {}, tframe: {}, placement: {}, trial:{}, save: {}".format(
                method, world, nareas, decay, tframe, placement, i+1, save))
            launch_nodes('int_preservation', 'mission.launch', params, logfile)

if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation')
    os.chdir('/root/catkin_ws/src/results/int_preservation')

    #Sample node poses
    # worlds = ['cluttered']
    # nareas_list = [70]
    # nplacements = 1
    # batch_sample_nodes_poses(worlds, nareas_list, nplacements)

    #Office
    #Adjust acml max_range=20
    #TODO: Run and record for the video simulations
    ##Office quad 4: T4, H4, DP4
    ##Cluttered quad 4: T4, H4, DP4
    ##Open

    # run_experiment(method='treebased_decision', world='office', nareas=4, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=10, dec_steps=4, ntrials=1, save=False)

    #Note: For cluttered adjust acml
    run_experiment(method='replay_decisions', world='office', nareas=4, placement=1, decay='non_uniform',
                   learn_decay=None, tframe=1500, dec_steps=4, ntrials=1, save=False)

    # run_experiment(method='heuristic_decision', world='office', nareas=4, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=1500, dec_steps=4, ntrials=1, save=False)
    #
    # run_experiment(method='dynamic_programming', world='office', nareas=4, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=1500, dec_steps=4, ntrials=1, save=False)


    # run_experiment(method='dynamic_programming', world='office', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=3, ntrials=5, save=True)
    #
    # run_experiment(method='random_decision', world='office', nareas=4, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=None, ntrials=3, save=True)
    #
    #
    # run_experiment(method='random_decision', world='open', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=None, ntrials=3, save=True)

    # Placement 2
    # run_experiment(method='treebased_decision', world='office', nareas=4, placement=2, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='heuristic_decision', world='office', nareas=4, placement=2, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='dynamic_programming', world='office', nareas=4, placement=2, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    #
    # # Placement 3
    # run_experiment(method='treebased_decision', world='office', nareas=4, placement=3, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='heuristic_decision', world='office', nareas=4, placement=3, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='dynamic_programming', world='office', nareas=4, placement=3, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    #
    # # Placement 4
    # run_experiment(method='treebased_decision', world='office', nareas=4, placement=4, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='heuristic_decision', world='office', nareas=4, placement=4, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='dynamic_programming', world='office', nareas=4, placement=4, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    #
    #
    # # Placement 5
    # run_experiment(method='treebased_decision', world='office', nareas=4, placement=5, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='heuristic_decision', world='office', nareas=4, placement=5, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='dynamic_programming', world='office', nareas=4, placement=5, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)

    #Cluttered
    #Adjust acml max_range=10
    #TODO: Run for cluttered. Adjust max_range pls!
    # run_experiment(method='dynamic_programming', world='cluttered', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=3, ntrials=5, save=True)

    # run_experiment(method='random_decision', world='cluttered', nareas=4, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=None, ntrials=3, save=True)



    #
    # run_experiment(method='dynamic_programming', world='cluttered', nareas=4, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    #
    # run_experiment(method='treebased_decision', world='cluttered', nareas=4, placement=2, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=2, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='dynamic_programming', world='cluttered', nareas=4, placement=2, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    #
    # # Placement 3
    # run_experiment(method='treebased_decision', world='cluttered', nareas=4, placement=3, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=3, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='dynamic_programming', world='cluttered', nareas=4, placement=3, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    #
    # # Placement 4
    # run_experiment(method='treebased_decision', world='cluttered', nareas=4, placement=4, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=4, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='dynamic_programming', world='cluttered', nareas=4, placement=4, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    #
    # # Placement 5
    # run_experiment(method='treebased_decision', world='cluttered', nareas=4, placement=5, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=4, placement=5, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)
    # #
    # run_experiment(method='dynamic_programming', world='cluttered', nareas=4, placement=5, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=4, ntrials=3, save=True)

    # run_experiment(method='heuristic_decision', world='cluttered', nareas=70, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=75, dec_steps=10 ** 2, ntrials=1, save=True)
    #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=70, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=75, dec_steps=10 ** 3, ntrials=1, save=True)

    # run_experiment(method='treebased_decision', world='cluttered', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=3, ntrials=3, save=True)
    #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=3, ntrials=3, save=True)
    #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=6, ntrials=3, save=True)
    #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=9, ntrials=3, save=True)
    #
    # run_experiment(method='heuristic_decision', world='cluttered', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=12, ntrials=3, save=True)

    #Open
    #Adjust acml max_range=20
    # run_experiment(method='dynamic_programming', world='open', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=3, ntrials=5, save=True)

    # run_experiment(method='treebased_decision', world='open', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=3, ntrials=3, save=True)
    #
    # run_experiment(method='heuristic_decision', world='open', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=3, ntrials=3, save=True)
    #
    # run_experiment(method='heuristic_decision', world='open', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=6, ntrials=3, save=True)
    #
    # run_experiment(method='heuristic_decision', world='open', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=9, ntrials=3, save=True)
    #
    # run_experiment(method='heuristic_decision', world='open', nareas=9, placement=1, decay='non_uniform',
    #                learn_decay=None, tframe=2100, dec_steps=12, ntrials=3, save=True)