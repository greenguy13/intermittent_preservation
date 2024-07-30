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
def run_experiment(method, world, nareas, placement, decay, tframe, inference=None, dec_steps=1, ntrials=1,
                   history_data=None, history_decisions=None, save=False):
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
    if method != 'random_decision':
        for i in range(ntrials):
            if inference is not None:
                fileresult = '{}_{}_{}_n{}_p{}_{}_k{}_{}'.format(method, inference, world, nareas, placement, decay, dec_steps, i + 1)
                params = ['method:={}'.format(method),
                          'inference:={}'.format(inference),
                          'world:={}'.format(world), 'nareas:={}'.format(nareas),
                          'decay:={}'.format(decay),
                          'dsteps:={}'.format(dec_steps),
                          'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                          'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult), 'save:={}'.format(save)]
                if (method == 'heuristic_uncertainty' and inference == 'timeseries') and (history_data is not None and history_decisions is not None):
                    params.append('history_data:={}'.format(history_data))
                    params.append('history_decisions:={}'.format(history_decisions))
                print(
                    "Launching...method: {}, inference: {}, world: {}, nareas: {}, decay: {}, dsteps: {}, tframe: {}, placement: {}, trial: {}, save: {}".format(
                        method, inference, world, nareas, decay, dec_steps, tframe, placement, i + 1, save))
            else:
                fileresult = '{}_{}_n{}_p{}_{}_k{}_{}'.format(method, world, nareas, placement, decay, dec_steps, i + 1)
                params = ['method:={}'.format(method),
                          'world:={}'.format(world), 'nareas:={}'.format(nareas),
                          'decay:={}'.format(decay),
                          'dsteps:={}'.format(dec_steps),
                          'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                          'fileposes:={}'.format(fileposes), 'fileresult:={}'.format(fileresult),
                          'save:={}'.format(save)]
                print("Launching...method: {}, world: {}, nareas: {}, decay: {}, dsteps: {}, tframe: {}, placement: {}, trial: {}, save: {}".format(
                        method, world, nareas, decay, dec_steps, tframe, placement, i + 1, save))
            logfile = fileresult + '.txt'
            launch_file = 'mission.launch' #'/home/ameldocena/catkin_ws/src/intermittent_preservation/launch/mission.launch'
            launch_nodes('int_preservation', launch_file, params, logfile)
    else:
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

def retrieve_history_data_decisions_filenames(world, method, inference, nareas, placement, dec_steps, trial):
    """
    Retrieves the filenames for history data and decisions accomplished
    :return:
    """
    history_data_filename = '{}_{}_{}_n{}_p{}_non_uniform_k{}_{}_robot0_recorded_data.pkl'.format(method, inference, world, nareas, placement, dec_steps, trial)
    history_decisions_filename = '{}_{}_{}_n{}_p{}_non_uniform_k{}_{}_robot0_decisions_acc_travel.pkl'.format(method, inference, world, nareas, placement, dec_steps, trial)
    return history_data_filename, history_decisions_filename

if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')

    #Sample node poses
    # worlds = ['office']
    # nareas_list = [8]
    # nplacements = 2000
    # batch_sample_nodes_poses(worlds, nareas_list, nplacements)

    #Office
    #Adjust acml.launch, max_range=20
    # Sanity checker
    # history_data, history_decisions = retrieve_history_data_decisions_filenames(world='office', method='heuristic_uncertainty', inference='expected', nareas=8, placement=2, dec_steps=4, trial=1)

    # placement = 1
    # run_experiment('heuristic_uncertainty', 'office', 8, placement, 'non_uniform', 20,
    #                inference='timeseries', dec_steps=1, ntrials=1, save=False)


    placement = 1
    run_experiment('treebased_decision', 'office', 8, placement, 'non_uniform', 3100,
                   inference='oracle', dec_steps=4, ntrials=1, save=True)

    run_experiment('heuristic_uncertainty', 'office', 8, placement, 'non_uniform', 3100,
                   inference='timeseries', dec_steps=4, ntrials=1, save=True)

    run_experiment('multiarmed_ucb', 'office', 8, placement, 'non_uniform', 3100,
                   inference='optimistic', dec_steps=1, ntrials=1, save=True)

    run_experiment('correlated_thompson', 'office', 8, placement, 'non_uniform', 3100,
                   inference='optimistic', dec_steps=1, ntrials=1, save=True)

    run_experiment('correlated_ucb', 'office', 8, placement, 'non_uniform', 3100,
                   inference='optimistic', dec_steps=1, ntrials=1, save=True)

    run_experiment('dynamic_programming', 'office', 8, placement, 'non_uniform', 3100,
                   inference=None, dec_steps=4, ntrials=1, save=True)



    #
    # placement = 32
    # run_experiment('treebased_decision', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='oracle', dec_steps=4, ntrials=1, save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='expected', dec_steps=4, ntrials=1, save=True)
    #
    # run_experiment('multiarmed_ucb', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('correlated_thompson', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('correlated_ucb', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('dynamic_programming', 'office', 8, placement, 'non_uniform', 3100,
    #                inference=None, dec_steps=4, ntrials=1, save=True)
    #
    # placement = 33
    # run_experiment('treebased_decision', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='oracle', dec_steps=4, ntrials=1, save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='expected', dec_steps=4, ntrials=1, save=True)
    #
    # run_experiment('multiarmed_ucb', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('correlated_thompson', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('correlated_ucb', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('dynamic_programming', 'office', 8, placement, 'non_uniform', 3100,
    #                inference=None, dec_steps=4, ntrials=1, save=True)


    #Cluttered
    #Adjust acml max_range=10

    #Open
    #Adjust acml max_range=20