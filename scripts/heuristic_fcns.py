#!/usr/bin/env python

import numpy as np
from loss_fcns import *
import project_utils as pu
from infer_lstm import *
from time import process_time

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
    # debug("Discount: {}, {}. Dec steps: {}, {}".format(type(discount), discount, type(dec_steps), dec_steps))
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
        # debug("Area: {}. Fmeasure: {} Forecasted fmeasure: {}".format(area, curr_fmeasures[area], forecasted_fmeasures[area]))

    #Measure cost
    cost = compute_cost_fmeasures(forecasted_fmeasures, fsafe, fcrit)
    # debug("Computed cost: {}".format(cost))

    #Return forecasted fmeasures and cost
    return forecasted_fmeasures, cost

def forecast_opportunity_cost(curr_fmeasures, tlapses, forecast_decay_dict, loss_params, discount, dec_steps, average_duration_decay_dict, forecast_tstep):
    """
    #TODO: Add decay_dict as a parameter, forecast_step. DONE
    #TODO: The decay_dict should be a COPY


    #If we have a forecast model initially, then no need for the data and model


    Forecasts future opportunity cost of a decision given dec_steps to look ahead
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
    #TODO: Replace dec_steps in terms of time steps
    # forecasted_decay = forecast_decay_lstm(model, data, dec_steps) #Note: This already resets index of the forecast in the data frame

    # debug('Forecasted decay: {}'.format(forecasted_decay.to_numpy()))
    forecast_timesteps = np.array([forecast_tstep]*len(curr_fmeasures)).astype(int)
    # debug("Init forecast timesteps: {}".format(forecast_timesteps))
    # debug("Init decayed F for future forecast: {}".format(curr_fmeasures))

    #TODO: We do the forecasting first. We then just index/look them up. In this case, this is linear. A
    #   What we do is we take the max number of forecast time steps by assuming the max duration times decsteps
    # Take the max number of tlapse?

    """
    Idea: Forecast first
        1. Come up with a scalar for forecast timesteps:
            a. Get the max tlapse * max average duration * (decsteps-1)
        2. Use this to forecast lstm
    
    Idea: Look up in forecast
        1. Just feed decay_dict in
    """
    # max_tlapse = np.max(forecast_timesteps)
    # max_average_duration = np.max(list(average_duration_decay_dict.values()))
    # max_forecast_timesteps = int(max_tlapse + max_average_duration * (dec_steps-1))
    #
    # #TODO: Insert timer here
    # forecast_start = process_time()
    # forecast_decay_dict = forecast_decay_lstm(model, data, max_forecast_timesteps)
    # forecast_end = process_time()
    # forecast_process_time = forecast_end - forecast_start
    #
    # debug("Max forecast tsteps: {}. Process time: {}".format(max_forecast_timesteps, forecast_process_time))
    # debug("Forecasted decay data {}:".format(forecast_decay_dict))

    #Measure discounted loss
    # debug("\n Forecasting future opp cost")
    loss_arr = []
    for i in range(1, dec_steps):
        #Forecast the decay rates at this time step
        # decay_dict = forecast_decay_timesteps(model, data, forecast_timesteps)  # Update decay_dict here
        # TODO: We do a lookup here of the decay rate given their respective tlapses in the forecasted decay_dict
        decay_dict = lookup_forecasted_data(forecast_decay_dict, forecast_timesteps)
        # debug("Forecasted decay rate in step {}: {}".format(i, decay_dict))


        #Estimate the average duration areas are decaying for one decision step
        forecast_time_dict = pu.add_entries_dicts(tlapses, average_duration_decay_dict)
        # debug("Forecast future step: {}. Tlapses: {}. Average duration: {}".format(i, tlapses, average_duration_decay_dict))
        # debug("Forecasted time: {}".format(forecast_time_dict))
        # decay_dict = forecasted_decay.iloc[i].to_dict()

        #Forecast the opportunity cost at this decision step
        forecasted_fmeasures, loss = heuristic_timeseries_forecast_cost(curr_fmeasures, forecast_time_dict, decay_dict, fsafe, fcrit)

        #Update counters and looped variables
        loss_arr.append(loss)
        curr_fmeasures = forecasted_fmeasures.copy()
        # debug("Forecasted fmeasures: {}. Loss: {}".format(curr_fmeasures, loss))
        tlapses = forecast_time_dict.copy()

        forecast_timesteps += np.array(list(average_duration_decay_dict.values())).astype(int)
        # debug("Updated forecast timesteps: {}".format(forecast_timesteps))

    #Sum up discounted losses throughout the decision steps
    cost = np.dot(discount_arr, np.array(loss_arr))
    # debug("Discount: {}. Costs: {}".format(discount_arr, loss_arr))
    return cost

def lookup_forecasted_data(forecasted_data, forecast_timesteps):
    """
    Looks up the decay for each area given their respective forecast_timestep
    :param forecasted_data:
    :param forecast_timesteps:
    :return:
    """
    decay_dict = dict()
    for area in forecasted_data.columns:
        # print("Area: {} {}".format(type(area), area))
        timestep = forecast_timesteps[area-1]
        # debug("Timestep: {}".format(timestep))
        # debug("Looking up Area {} in forecast dict. Forecast tstep: {}. Decay area: {}".format(area, timestep, forecasted_data.iloc[timestep][area]))
        decay_dict[area] = forecasted_data.iloc[timestep][area]
    return decay_dict

def forecast_decay_timesteps(model, data, forecast_timesteps):
    """
    Forecast (by time steps) the respective decay on average per area
    :param forecast_timesteps: array of length areas containing time step to forecast for each area
    :return:
    """
    #For each area, extract the average duration that area is decaying
    #Add
    decay_dict = dict()
    for area in range(len(forecast_timesteps)):
        forecast = forecast_decay_lstm(model, data, forecast_timesteps[area])
        decay_dict[area+1] = forecast.iloc[-1][area+1] #TODO: Okay this is the one where we do an iloc

    return decay_dict
def debug(msg, robot_id=0):
    pu.log_msg('robot', robot_id, msg, debug=True)