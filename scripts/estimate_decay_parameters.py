"""
Expected decay
    > Var
    > Confidence interval

Lower bound (optimistic)
CVar (truly pessimistic)
Proposed (pessimistic)
    > Expected between CVar and Expected decay
"""
import scipy.stats as stats
import numpy as np
import project_utils as pu
import math
import pandas as pd
"""
params: alpha
"""

def margin_of_error(data, alpha):
    sd = np.std(data, ddof=1)
    confidence = 1-alpha
    n = len(data)
    df = n-1

    #Standard error
    std_error = sd/np.sqrt(n)

    #Margin of error
    margin_of_error = stats.t.ppf((1 + confidence)/2, df) * std_error
    return margin_of_error

def value_at_risk(data, alpha):
    """
    Quantile corresponding to the chosen confidence level
    Here, the value at risk is on the right tail since we are minimizing the loss
    :param data:
    :param alpha:
    :return:
    """
    VaR = np.percentile(data, (1-alpha)*100)
    return VaR

def simple_average_param(recorded_param_dict, area):
    average = np.nanmean(recorded_param_dict[area]) #Measure the average
    return average

def weighted_average_param(recorded_param_dict):
    pass

def lower_bound_param(recorded_param_dict, area, alpha):
    data = np.array(recorded_param_dict[area])
    m = np.nanmean(data)
    moe = margin_of_error(data, alpha)
    lower_bound = m - moe
    return lower_bound

def CVaR_param(recorded_param_dict, area, alpha):
    """
    Our objective is to minimize loss. And so the risk is values greater than average.
    We thus take the expected value of VaR and beyond
    :param recorded_param_dict:
    :param alpha:
    :return:
    """
    data = np.array(recorded_param_dict[area])
    VaR = value_at_risk(data, alpha)
    CVaR = np.nanmean([x for x in data if x >= VaR])
    return CVaR

def proposed_heuristic(recorded_param_dict, area, alpha):
    """
    The expected value of VaR and the expected value
    :param recorded_param_dict:
    :return:
    """
    data = np.array(recorded_param_dict[area])
    m = np.nanmean(data)
    VaR = value_at_risk(data, alpha)
    proposed = np.nanmean([x for x in data if (x >= m and x < VaR)])
    if math.isnan(proposed): proposed = m
    pu.log_msg('robot', 0, 'data, mean, VaR, proposed: {}'.format((data, m, VaR, proposed)))
    return proposed

def moving_average(recorded_param_dict, area, win_size):
    """
    Forecasts the decay rate by moving average
    :param recorded_param_dict:
    :param area:
    :return:
    """
    data = np.array(recorded_param_dict[area])
    win_size = max(1, win_size)
    forecast = np.mean(data[-win_size:])
    return forecast

"""
UPNEXT: Run an experiment for sanity check
1. Implement the method: 
    > yaml file for the window size
    > heuristic_decision_uncertainty.py
    > run_experiment.py
    > launch file(?) 
2. Check whether the forecast is correct
    > The collected decay list
    > The sensitivity condition and the forecast trigger
    > The forecast
"""