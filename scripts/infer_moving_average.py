#!/usr/bin/env python

"""
Expected decay
    > Var
    > Confidence interval

Lower bound (optimistic)
CVar (truly pessimistic)
PO (pessimistic)
    > Expected between CVar and Expected decay
"""
import scipy.stats as stats
import numpy as np
import project_utils as pu
import math
from statsmodels.tsa.statespace.varmax import VARMAX
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

#### Moving average
def moving_average(recorded_param_dict, area, win_size, alpha, type='expected'):
    """
    Forecasts the decay rate by moving average
    :param recorded_param_dict:
    :param area:
    :return:
    """
    data = np.array(recorded_param_dict[area])
    win_size = max(1, win_size)
    forecast = np.mean(data[-win_size:])
    moe = margin_of_error(data[-win_size:], alpha)
    lower_b, upper_b = forecast - moe, forecast + moe
    if type == 'expected':
        return forecast
    elif type == 'optimistic':
        if lower_b <= 0.0:
            return forecast
        return lower_b
    elif type == 'pessimistic':
        if upper_b >= 1.0:
            return forecast
        return upper_b