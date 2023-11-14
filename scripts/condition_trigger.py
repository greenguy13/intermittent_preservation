"""
> If there is at least one change
> And the change is greater than a set threshold
"""

import numpy as np

"""
params to be set
> sensitivity
> correlation_threshold
"""

def sensitivity_condition(measured_decay_param, current_decay_param, sensitivity):
    """
    The change of Loss wrt change in the decay param
    :return:
    """
    if (measured_decay_param - current_decay_param)/current_decay_param > sensitivity:
        return True
    return False

def correlation_condition(area, correlation_matrix, correlation_threshold):
    """
    Tests the significance of correlation between area
    PO1: Strongly correlated (>0.70)
    PO2: Statistical test

    :return:
    """
    arr = correlation_matrix[area, :]
    condition = arr > correlation_threshold
    count = np.sum(condition)
    if count >= 1:
        return True
    return False