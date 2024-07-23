#!/usr/bin/env python

import numpy as np
from loss_fcns import *
import project_utils as pu
from infer_decay_parameters import *

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

def heuristic_forecast_cost(curr_fmeasures, average_duration_decay, decay_dict, fsafe, fcrit):
    """
    Forecast the fmeasures by some heuristic
    Measure the cost of the forecasted fmeasures
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
            forecast_fmeasure = decay(decay_dict[area], average_duration_decay[area], curr_fmeasures[area])
        #Case 2: Restore curr_fmeasure to max fmeasure if fcrit < curr_fmeasure
        else:
            forecast_fmeasure = max_fmeasure
        forecasted_fmeasures[area] = forecast_fmeasure

    #Measure cost
    cost = compute_cost_fmeasures(forecasted_fmeasures, fsafe, fcrit)

    #Return forecasted fmeasures and cost
    return forecasted_fmeasures, cost

#Goal: Vectorized operation

def heuristic_cost_decision(curr_fmeasures, decay_dict, loss_params, discount, dec_steps, average_duration_decay_dict):
    """
    Measures the heuristic cost of a decision
    :param curr_fmeasures:
    :param decay_dict:
    :param loss_params:
    :param discount:
    :param dec_steps:
    :param average_duration_decay_dict:
    :return:
    """
    fsafe, fcrit = loss_params

    # #Measure initial time elapsed
    # prev_time_dict = get_elapsed_time(max_fmeasure, curr_fmeasures, decay_dict)
    #
    discount_arr = discount**np.array(list(range(1, dec_steps)))

    #Measure discounted loss
    loss_arr = []
    for i in range(1, dec_steps):
        # forecast_time_dict = pu.add_entries_dicts(prev_time_dict, average_duration_decay_dict)
        # forecasted_fmeasures, loss = heuristic_forecast_loss(curr_fmeasures, forecast_time_dict, decay_dict, fsafe, fcrit)
        forecasted_fmeasures, loss = heuristic_forecast_cost(curr_fmeasures, average_duration_decay_dict, decay_dict, fsafe, fcrit)
        loss_arr.append(loss)
        curr_fmeasures = forecasted_fmeasures
        # prev_time_dict = forecast_time_dict

    #Sum up discounted losses throughout the decision steps
    cost = np.dot(discount_arr, np.array(loss_arr))
    return cost

### Heuristic cost under uncertainty
def heuristic_timeseries_forecast_cost(curr_fmeasures, forecast_time_dict, decay_dict, fsafe, fcrit):
    """
    Forecast the fmeasures by some heuristic
    Measure the cost of the forecasted fmeasures
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
            forecast_fmeasure = decay(decay_dict[area], forecast_time_dict[area], max_fmeasure)
        #Case 2: Restore curr_fmeasure to max fmeasure if fcrit < curr_fmeasure
        else:
            forecast_fmeasure = max_fmeasure
        forecasted_fmeasures[area] = forecast_fmeasure

    #Measure cost
    cost = compute_cost_fmeasures(forecasted_fmeasures, fsafe, fcrit)

    #Return forecasted fmeasures and cost
    return forecasted_fmeasures, cost

def forecast_opportunity_cost(curr_fmeasures, tlapses, latest_obs, model, model_scale, loss_params, discount, dec_steps, average_duration_decay_dict):
    """
    Forecasts the opportunity cost of a decision given dec_steps to look ahead
    :param curr_fmeasures:
    :param loss_params:
    :param discount:
    :param dec_steps:
    :param average_duration_decay_dict:
    :return:
    """
    fsafe, fcrit = loss_params

    #Measure initial time stamp
    discount_arr = discount**np.array(list(range(1, dec_steps)))

    #Forecast decay rates in next decsteps
    forecasted_decay = forecast_decay(model, latest_obs, dec_steps, model_scale) #Note: This already resets index of the forecast in the data frame

    #Measure discounted loss
    loss_arr = []
    for i in range(1, dec_steps):
        forecast_time_dict = pu.add_entries_dicts(tlapses, average_duration_decay_dict)
        decay_dict = forecasted_decay.iloc[i].to_dict()
        forecasted_fmeasures, loss = heuristic_timeseries_forecast_cost(curr_fmeasures, forecast_time_dict, decay_dict, fsafe, fcrit)
        loss_arr.append(loss)
        curr_fmeasures = forecasted_fmeasures.copy()
        tlapses = forecast_time_dict.copy()

    #Sum up discounted losses throughout the decision steps
    cost = np.dot(discount_arr, np.array(loss_arr))
    return cost