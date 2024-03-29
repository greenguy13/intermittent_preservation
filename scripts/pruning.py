#!/usr/bin/env python
"""
Pruning strategy for tree-based decision making
"""


def is_feasible(battery, battery_consumption, battery_reserve):
    """
    If current battery level can cover battery consumption to an area then back to the charging station.
    Here we battery_consumption includes monitoring then back to the charging station.
    :param battery_level: current battery level
    :param battery_consumption: battery consumed to monitor an area then back to charging station
    :return:
    """
    if battery >= max(battery_consumption, battery_reserve):
        return True
    else:
        return False

def prune(battery, battery_consumption, battery_reserve, decayed_fmeasure=None, safe_fmeasure=None):
    """
    Prunes a branch growing from a node if it is infeasible (REMOVED: or if in the next decision step it is still in safe.)
    Equivalently: If feasible and F-measure is below safe, do not prune. Else, prune.

    #PO: Mabunay ha base. We prune staying in charging station when there is an area that is decaying/will be decaying

    :return: bool
    """
    if (is_feasible(battery, battery_consumption, battery_reserve) is False): # or (decayed_fmeasure is not None and decayed_fmeasure >= safe_fmeasure):
        return True

    return False