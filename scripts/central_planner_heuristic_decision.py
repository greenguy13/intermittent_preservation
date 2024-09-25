#!/usr/bin/env python

"""
Implements Clustered BFVG
1. Cluster the areas in the environment based on some attributes
    > In clustering we use K-means clustering algorithm
    > Q: What do you think would these attributes be if we are to construct a simulation?

2. We have available robots and unassigned clusters
    > What defines an available/un-assigned robot? Actually, un-assigned robot would be better
        + If the robot has no currently assigned cluster, either it is heading toward or parked in the charging station
    > What defines an unassigned cluster?
        + If there is no robot assigned to preserve that cluster of areas
    > How do we make the assignment?
        + We evaluate the cluster's value, we then assign them greedily, whichever has the highest value
        + For each cluster, we evaluate among the robots (whose current task is not to charge up) based on their battery level
            and current location. Among those, we evaluate the

Algorithm sketch:
    Among the areas in the environment, cluster them into n-clusters based on their attributes, where n is the number of robots
    Among the unassigned clusters, we evaluate their score, and then assign an available robot to it
        > Q1: Which unassigned clusters gets assignment first?
        > Q2: Which available robot gets assigned to an unassigned cluster?

data = inputs
clusters = Cluster(data) #clusters would be a list containing cluster of areas, where the number of clusters is the number of robots

PO: Evaluate the value of the unassigned clusters, and then store in a priority queue
    + Evaluation would be a forecast of the expected opportunity cost for a given number of future visits
if there is one unassigned cluster, we consider re-planning/re-assignment:
    + Note that this means there is one un-assigned robot whose task is to charge up or just parked
    + PO: Average distance within the cluster, Current location of each robot and their remaining battery,
        and whether their battery level can cover the forecasted number of future visits

Consider re-plan is triggered only when one robot is heading to a charging station

for cluster in unassigned clusters with priority:
    for robot in robots:
        evalute their score for that unassigned cluster
    assign the cluster to the robot with highest score
"""

