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
    rate = 1 - (decayed_fmeasure/max_fmeasure)**(1/(tlapse + 1e-10))
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
        loss = loss_fcn(max_fmeasure, fmeasures[area])
        cost += loss
    return cost

