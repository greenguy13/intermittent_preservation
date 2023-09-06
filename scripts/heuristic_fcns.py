#!/usr/bin/env python

import math
from loss_fcns import *

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

def forecast_fmeasures(curr_fmeasures, forecast_time_dict, decay_dict, fcrit):
    """
    Forecast the fmeasures by some assumptions
    :param curr_fmeasures:
    :param forecast_time_arr:
    :param decay_arr:
    :param fcrit:
    :return:
    """
    forecasted_fmeasures = dict()
    for area in curr_fmeasures:
        #Case 1: Keep decaying if curr_fmeasure >= fcrit
        if curr_fmeasures[area] >= fcrit:
            forecast_fmeasure = decay(decay_dict[area], forecast_time_dict[area], curr_fmeasures[area])
        #Case 2: Restore curr_fmeasure to max fmeasure if fcrit < curr_fmeasure
        else:
            forecast_fmeasure = max_fmeasure
            #TODO: Are we updating the time stamps as well?
        forecasted_fmeasures[area] = forecast_fmeasure
    return forecasted_fmeasures

def forecast_loss(curr_fmeasures, decay_dict, loss_params, discount, dec_steps, average_duration_decay_dict):

    fsafe, fcrit = loss_params

    #Measure immediate loss
    loss = compute_cost_fmeasures(curr_fmeasures, fsafe, fcrit)

    #Measure initial time stamp
    time_prev_step_dict = get_elapsed_time(max_fmeasure, curr_fmeasures, decay_dict)

    #Forecast future losses and discount to measure discounted future losses
    for i in range(dec_steps):
        forecast_time_dict = time_prev_step_dict + average_duration_decay_dict
        forecasted_fmeasures = forecast_fmeasures(curr_fmeasures, forecast_time_arr, decay_arr)
        loss += (discount**i)*compute_cost_fmeasures(forecasted_fmeasures, fsafe, fcrit)
        curr_fmeasures = forecasted_fmeasures
        time_prev_step_arr = forecast_time_arr

    return loss

