#!/usr/bin/env python

import math
import numpy as np
from loss_fcns import *
import project_utils as pu

max_fmeasure = 100

def get_elapsed_time(max_fmeasure, fmeasures, discount_arr):
    """
    Returns elapsed time since fmeasures has decayed from max_fmeasure by discount_arr
    :param max_fmeasure:
    :param fmeasures:
    :param discount_arr:
    :return: elapsed_time, array of elapsed time
    """
    elapsed_time_dict = dict()
    for area in fmeasures:
        elapsed_time_dict[area] = get_time_given_decay(max_fmeasure, fmeasures[area], discount_arr[area])
    return elapsed_time_dict

def heuristic_forecast_loss(curr_fmeasures, forecast_time_dict, decay_dict, fsafe, fcrit):
    """
    Forecast the fmeasures by some heuristic
    Measure the loss of the forecasted fmeasures
    :param curr_fmeasures:
    :param forecast_time_arr:
    :param decay_arr:
    :param fcrit:
    :return:
    """
    #TODO: Eye
    forecasted_fmeasures = dict()
    for area in curr_fmeasures:
        #Case 1: Keep decaying if curr_fmeasure >= fcrit
        if curr_fmeasures[area] >= fcrit:
            forecast_fmeasure = decay(decay_dict[area], forecast_time_dict[area], curr_fmeasures[area])
        #Case 2: Restore curr_fmeasure to max fmeasure if fcrit < curr_fmeasure
        else:
            forecast_fmeasure = max_fmeasure
        forecasted_fmeasures[area] = forecast_fmeasure

    #Measure loss
    loss = compute_cost_fmeasures(forecasted_fmeasures, fsafe, fcrit)

    #Return forecasted fmeasures and loss
    return forecasted_fmeasures, loss

#Goal: Vectorized operation
#Finish up everything, awaiting sanity check

def heuristic_loss_decision(curr_fmeasures, decay_dict, loss_params, discount, dec_steps, average_duration_decay_dict):
    """
    Measures the heuristic loss of a decision
    :param curr_fmeasures:
    :param decay_dict:
    :param loss_params:
    :param discount:
    :param dec_steps:
    :param average_duration_decay_dict:
    :return:
    """
    fsafe, fcrit = loss_params

    #Measure initial time stamp
    prev_time_dict = get_elapsed_time(max_fmeasure, curr_fmeasures, decay_dict)

    discount_arr = discount**np.array(list(range(1, dec_steps)))

    #Measure discounted loss
    loss_arr = []
    for i in range(1, dec_steps):
        forecast_time_dict = pu.add_entries_dicts(prev_time_dict, average_duration_decay_dict)
        forecasted_fmeasures, loss = heuristic_forecast_loss(curr_fmeasures, forecast_time_dict, decay_dict, fsafe, fcrit)
        loss_arr.append(loss)
        curr_fmeasures = forecasted_fmeasures
        prev_time_dict = forecast_time_dict

    #Sum up discounted losses throughout the decision steps
    loss = np.dot(discount_arr, np.array(loss_arr))
    return loss

