#!/usr/bin/env python

import math


def decay(rate, t, starting_value):
    """
    Decay function
    :param rate:
    :param t:
    :return:
    """
    decayed_fmeasure = starting_value*(1.0 - rate)**t
    return decayed_fmeasure

def get_time_given_decay(max_fmeasure, decayed_fmeasure, rate):
    """
    Retrieves time given decayed measure and decay rate (by inversion)
    :param decayed_measure:
    :param rate:
    :return:
    """
    t = math.log(decayed_fmeasure/max_fmeasure) / math.log(1.0-rate)

    return t

def beta_rate(rate, rates):
    """
    Computes the normalized F-measure decay rate, which then is the Loss growth rate
    :param rate (float): decay rate
    :param rates (list): list of decay rates
    :return:
    """
    beta = rate / sum(rates)
    return beta

"""
TODO: Refinements
> compute_loss
    + the loss function
> forecast_loss
    + forecast the loss given estimated duration
    + uses compute_loss 
> compute_cost_fmeasures
    + compute the loss for each F-measure
    + sum up all the losses
"""

def compute_loss(max_fmeasure, decayed_fmeasure, fsafe, fcrit, rate, est_duration):
    """
    Computes loss by estimating the decayed fmeasure given the decay rate after a set duration
    :param decayed_fmeasure:
    :param rate:
    :param duration:
    :return:
    """
    t0 = get_time_given_decay(max_fmeasure, decayed_fmeasure, rate)
    t = t0 + est_duration
    est_decayed_fmeasure = decay(rate, t, max_fmeasure)
    print("Current measure: {}. Estimated decayed measure: {}".format(decayed_fmeasure, est_decayed_fmeasure))

    # Safe zone
    if est_decayed_fmeasure >= fsafe:
        loss = 0 #fsafe - est_decayed_fmeasure
    # Caution zone
    elif fcrit <= est_decayed_fmeasure and est_decayed_fmeasure < fsafe:
        loss = 2*(fsafe - est_decayed_fmeasure) #fsafe - est_decayed_fmeasure
    # Crit zone
    elif est_decayed_fmeasure < fcrit:
        loss = (fsafe - est_decayed_fmeasure)**2

    return float(loss)

def compute_cost_fmeasures(fmeasures, fsafe, fcrit):
    """
    Computes the cost, (i.e., the sum of losses), given the F-measures
    :param fmeasures: dict of F-measures of areas
    :return:
    """
    cost = 0
    areas = fmeasures.keys()
    for area in areas:
        if fmeasures[area] >= fsafe:
            loss = 100 - fmeasures[area]
        elif fcrit <= fmeasures[area] and fmeasures[area] < fsafe:
            loss = 2*(100 - fmeasures[area]) #fsafe - fmeasures[area]
        elif fmeasures[area] < fcrit:
            loss = (100 - fmeasures[area])**2
        cost += loss
    return cost

