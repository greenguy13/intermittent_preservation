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
def run_experiment(method, world, nareas, placement, decay, tframe, inference=None, dec_steps=1, ntrials=(0, 1), discount=None, exploration=None, nvisits=None,
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
        for i in range(*ntrials):
            if inference is not None:
                params = ['method:={}'.format(method),
                          'inference:={}'.format(inference),
                          'world:={}'.format(world), 'nareas:={}'.format(nareas),
                          'decay:={}'.format(decay),
                          'dsteps:={}'.format(dec_steps),
                          'tframe:={}'.format(tframe), 'placement:={}'.format(placement),
                          'fileposes:={}'.format(fileposes), 'save:={}'.format(save)]
                fileresult = '{}_{}_{}_n{}_p{}_{}_k{}_{}'.format(method, inference, world, nareas, placement, decay, dec_steps, i + 1)

                if method == 'treebased_decision':
                    params.append('discount:={}'.format(discount))

                elif (method == 'heuristic_uncertainty' and inference == 'timeseries') or method == 'heuristic_decision':
                    fileresult = '{}_{}_{}_n{}_p{}_{}_k{}_{}_disc{}_exp{}_nvisits{}'.format(method, inference, world, nareas, placement, decay,
                                                                     dec_steps, i + 1, discount, exploration, nvisits)
                    params.append('discount:={}'.format(discount))
                    params.append('exploration:={}'.format(exploration))
                    params.append('nvisits:={}'.format(nvisits))
                    if (history_data is not None and history_decisions is not None):
                        params.append('history_data:={}'.format(history_data))
                        params.append('history_decisions:={}'.format(history_decisions))

                elif method == 'multiarmed_ucb' or method == 'correlated_ucb':
                    fileresult = '{}_{}_{}_n{}_p{}_{}_k{}_{}_exp{}'.format(method, inference, world,
                                                                                            nareas, placement, decay,
                                                                                            dec_steps, i + 1, exploration)
                    params.append('exploration:={}'.format(exploration))

                params.append('fileresult:={}'.format(fileresult))
                print(
                    "Launching...method: {}, inference: {}, world: {}, nareas: {}, decay: {}, dsteps: {}, discount: {}, exploration: {}, nvisits: {}, tframe: {}, placement: {}, trial: {}, save: {}".format(
                        method, inference, world, nareas, decay, dec_steps, discount, exploration, nvisits, tframe, placement, i + 1, save))
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
    # nareas_list = [12]
    # nplacements = 2000
    # batch_sample_nodes_poses(worlds, nareas_list, nplacements)
    #
    # worlds = ['cluttered']
    # nareas_list = [8, 12]
    # nplacements = 2000
    # batch_sample_nodes_poses(worlds, nareas_list, nplacements)


    #Office
    #Adjust acml.launch, laser_max_range=20
    # Sanity checker
    # history_data, history_decisions = retrieve_history_data_decisions_filenames(world='office', method='heuristic_uncertainty', inference='expected', nareas=8, placement=2, dec_steps=4, trial=1)

    # placement = 1
    # run_experiment('rma_search', 'office', 12, placement, 'non_uniform', 50,
    #                inference=None, dec_steps=4, ntrials=(0, 1), save=True)

    placement = 1
    #
    # run_experiment('treebased_decision', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='oracle', dec_steps=4, ntrials=1, save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='timeseries', dec_steps=4, ntrials=1, save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 8, placement, 'non_uniform', 3100,
    #                inference='timeseries', dec_steps=1, ntrials=1, save=True)
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

    """
    Grid search parameters over e = [0.30, 0.60, 0.90]. For discount 0.0, means k=1
        Fine-tune exploration for reinfocement learning UCB. To re-run: e = 0.90
        Fine-tune exploration for our method for k=4, with equivalent discount as with oracle, to re-run: e = 0.00, 0.30, 0.60
        Fine-tune k, discount, exploration. PENDING
        
        Oracle, trial = 0, 1, 2
    """
    # e= 0.00, 0.30, 0.60, 0.90 params for our proposed method

    #For fine-tuning
    #Heuristic tune dec_steps
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 150,
    #                inference='timeseries', dec_steps=2, discount=0.50, exploration=0.30, nvisits=2, ntrials=(0, 1), save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 50,
    #                inference='timeseries', dec_steps=6, discount=0.75, exploration=0.30, nvisits=2, ntrials=1, save=True)

    #Heuristic fine tune of exploration rate
    # run_experiment('treebased_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.75, ntrials=(0,3), save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=4, discount=0.75, exploration=0.00, nvisits=2, ntrials=(0,3), save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=4, discount=0.75, exploration=0.30, nvisits=2, ntrials=(0,3), save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=4, discount=0.75, exploration=0.60, nvisits=2, ntrials=(0,3), save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=4, discount=0.75, exploration=0.90, nvisits=2, ntrials=(0,3), save=True)

    #UCB tune exploration
    # run_experiment('multiarmed_ucb', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, exploration=0.30, ntrials=1, save=True)
    #
    # run_experiment('multiarmed_ucb', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, exploration=0.60, ntrials=1, save=True)
    #
    # run_experiment('multiarmed_ucb', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, exploration=0.90, ntrials=(0,2), save=True)
    #
    # run_experiment('correlated_ucb', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, exploration=0.30, ntrials=1, save=True)
    #
    # run_experiment('correlated_ucb', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, exploration=0.60, ntrials=1, save=True)
    #
    # run_experiment('correlated_ucb', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, exploration=0.90, ntrials=(0,2), save=True)


    #Full trials
    # run_experiment('heuristic_decision', 'office', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=1, discount=0.00, exploration=None, nvisits=None, ntrials=(0, 3), save=True)
    #
    # run_experiment('treebased_decision', 'office', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.25, exploration=None, nvisits=None, ntrials=(0, 3), save=True)

    # run_experiment('heuristic_decision', 'office', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=1, discount=0.00, exploration=None, nvisits=None, ntrials=(0, 3), save=True)

    run_experiment('heuristic_decision', 'office', 4, placement, 'non_uniform', 2100,
                   inference='oracle', dec_steps=4, discount=0.25, exploration=None, nvisits=None, ntrials=(0, 2), save=True)

    # run_experiment('dynamic_programming', 'office', 4, placement, 'non_uniform', 2100,
    #                inference=None, dec_steps=4, ntrials=(0, 3), save=True)


    # run_experiment('heuristic_decision', 'office', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=3, discount=0.75, exploration=None, nvisits=None, ntrials=(0, 3), save=True)
    #
    # run_experiment('heuristic_decision', 'office', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=3, discount=1.00, exploration=None, nvisits=None, ntrials=(0, 3), save=True)

    #12 areas
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=1, discount=0.00, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)

    # run_experiment('treebased_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.25, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=1, discount=0.00, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)

    run_experiment('heuristic_decision', 'office', 9, placement, 'non_uniform', 2100,
                   inference='oracle', dec_steps=3, discount=0.25, exploration=None, nvisits=None, ntrials=(0, 2),
                   save=True)

    # run_experiment('dynamic_programming', 'office', 12, placement, 'non_uniform', 2100,
    #                inference=None, dec_steps=4, ntrials=(0, 3), save=True)

    #k=3
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=3, discount=0.25, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=3, discount=0.50, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=3, discount=0.75, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=3, discount=1.00, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # #k=4
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.25, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.50, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.75, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=1.00, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # #k=6
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=6, discount=0.25, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=6, discount=0.50, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=6, discount=0.75, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=6, discount=1.00, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)


    # run_experiment('heuristic_decision', 'cluttered', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=1, discount=0.00, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'cluttered', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.25, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'cluttered', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.50, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'cluttered', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=0.75, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_decision', 'cluttered', 4, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, discount=1.00, exploration=None, nvisits=None, ntrials=(0, 3),
    #                save=True)

    #
    # run_experiment('treebased_decision', 'cluttered', 12, placement, 'non_uniform', 2100,
    #                inference='oracle', dec_steps=4, ntrials=(0, 2), discount=0.75, save=True)
    #
    # run_experiment('heuristic_uncertainty', 'cluttered', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=4, discount=0.75, exploration=0.60, nvisits=2, ntrials=(2, 3), save=True)

    # run_experiment('correlated_thompson', 'cluttered', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, ntrials=(0, 3), save=True)

    # run_experiment('correlated_ucb', 'cluttered', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, exploration=0.60, ntrials=(1, 3), save=True)
    #
    # run_experiment('dynamic_programming', 'cluttered', 12, placement, 'non_uniform', 2100,
    #                inference=None, dec_steps=4, ntrials=(2, 3), save=True)
    #
    # run_experiment('rma_search', 'cluttered', 12, placement, 'non_uniform', 2100,
    #                inference=None, dec_steps=4, ntrials=(2, 3), save=True)


    #
    # run_experiment('multiarmed_ucb', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='optimistic', dec_steps=1, ntrials=5, save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=4, discount=0.75, exploration=0.60, nvisits=2, ntrials=(0, 3),
    #                save=True)
    # NOTE: dec_steps=4 has already been run

    # Currently running: Decision steps, slower scenario
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=1, discount=0.00, exploration=0.60, nvisits=2, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=8, discount=0.75, exploration=0.60, nvisits=2, ntrials=(0, 3),
    #                save=True)
    #
    # run_experiment('heuristic_uncertainty', 'office', 12, placement, 'non_uniform', 2100,
    #                inference='timeseries', dec_steps=12, discount=0.75, exploration=0.60, nvisits=2, ntrials=(0, 3),
    #                save=True)

    #Latest in previous runs: 33


    #Cluttered
    #Adjust acml laser_max_range=10

    #TODO: Run slower, inverted correlated areas

    # placement = 1
    #
    # run_experiment('treebased_decision', 'cluttered', 8, placement, 'non_uniform', 3100,
    #                inference='oracle', dec_steps=4, ntrials=1, save=True)
    #
    # run_experiment('heuristic_uncertainty', 'cluttered', 8, placement, 'non_uniform', 3100,
    #                inference='timeseries', dec_steps=4, ntrials=1, save=True)
    #
    # run_experiment('heuristic_uncertainty', 'cluttered', 8, placement, 'non_uniform', 3100,
    #                inference='timeseries', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('multiarmed_ucb', 'cluttered', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('correlated_thompson', 'cluttered', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('correlated_ucb', 'cluttered', 8, placement, 'non_uniform', 3100,
    #                inference='optimistic', dec_steps=1, ntrials=1, save=True)
    #
    # run_experiment('dynamic_programming', 'cluttered', 8, placement, 'non_uniform', 3100,
    #                inference=None, dec_steps=4, ntrials=1, save=True)


    #Open
    #Adjust acml laser_max_range=20