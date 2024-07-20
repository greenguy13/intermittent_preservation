#!/usr/bin/env python

"""
Noise models for travel and restoration
"""


def measure_travel_noise(actual_travel_time, est_travel_time):
    """
    Measures travel noise as a percentage from estimated
    :param actual_travel_time:
    :param est_travel_time:
    :return:
    """
    noise = (actual_travel_time - est_travel_time) / est_travel_time
    return noise

def measure_restoration_rate(restored_amount, end_restore_time, start_restore_time):
    """
    Measures restoration rate
    :param actual_restore_time:
    :param est_restore_time:
    :return:
    """
    restore_time = end_restore_time - start_restore_time
    rate = restored_amount / restore_time
    return rate