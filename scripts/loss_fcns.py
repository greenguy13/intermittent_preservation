#!/usr/bin/env python

import math


def decay(rate, t, starting_value):
    """
    Decay function
    :param rate:
    :param t:
    :return:
    """
    decayed_fmeasure = starting_value*(math.exp(-rate*t)) #previous decay: starting_value*(1.0 - rate)**t
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

def get_decay_rate(max_fmeasure, decayed_fmeasure, tlapse):
    """
    Computes the decay rate given F-measure and time elapsed since last update
    :param fmeasure:
    :param time:
    :return:
    """
    rate = 1 - (decayed_fmeasure/max_fmeasure)**(1/tlapse)
    return rate

def beta_rate(rate, rates):
    """
    Computes the normalized F-measure decay rate, which then is the Loss growth rate
    :param rate (float): decay rate
    :param rates (list): list of decay rates
    :return:
    """
    beta = rate / sum(rates)
    return beta

def loss_fcn(max_fmeasure, decayed_fmeasure):
    loss = (max_fmeasure - decayed_fmeasure)**2
    return loss

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

    #previous loss function
    # # Safe zone
    # if est_decayed_fmeasure >= fsafe:
    #     loss = max_fmeasure - est_decayed_fmeasure
    # # Caution zone
    # elif fcrit <= est_decayed_fmeasure and est_decayed_fmeasure < fsafe:
    #     loss = 2*(fsafe - est_decayed_fmeasure) #fsafe - est_decayed_fmeasure
    # # Crit zone
    # elif est_decayed_fmeasure < fcrit:
    #     loss = 4*(fsafe - est_decayed_fmeasure)

    # loss = (max_fmeasure - est_decayed_fmeasure)**2

    loss = loss_fcn(max_fmeasure, est_decayed_fmeasure)
    return float(loss)

def compute_cost_fmeasures(fmeasures, fsafe, fcrit, max_fmeasure=100):
    """
    Computes the cost, (i.e., the sum of losses), given the F-measures
    :param fmeasures: dict of F-measures of areas
    :return:
    """
    cost = 0

    for area in fmeasures:
        #previous loss function
        # if fmeasures[area] >= fsafe:
        #     loss = 100 - fmeasures[area]
        # elif fcrit <= fmeasures[area] and fmeasures[area] < fsafe:
        #     loss = 1.5*(100 - fmeasures[area]) #fsafe - fmeasures[area]
        # elif fmeasures[area] < fcrit:
        #     loss = 2.0*(100 - fmeasures[area])
        loss = (max_fmeasure-fmeasures[area])**2
        cost += loss

    return cost

def marginal_loss():
    """
    Marginal loss (for use if we assume submodular objective function
    :return:
    """
    pass

def compute_trigger_threshold(sensitivity, max_fmeasure=100, type='caution'):
    """
    Computes the equivalent trigger threshold from sensitivity threshold
    :param sensitivity:
    :return:
    """
    if type=='safe':
        L, k = 80, 1.0
    elif type=='caution':
        L, k = 65, 1.5
    elif type=='critical':
        L, k = 50, 2.0

    # delta = 1 - (1 - (L/k*max_fmeasure))**1/tlapse
    # threshold = sensitivity*delta

