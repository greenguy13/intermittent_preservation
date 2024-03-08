#!/usr/bin/env python3

import networkx as nx
import numpy as np
import math

"Node/vertex in a Graph used in dynamic programming"
class Node():
  def __init__(self, name, id, decay_rate, batt_consumption, tlapse=None):
    self.name = name
    self.id = id
    self.batt_consumption = 0 #battery consumed to get here
    self.tlapse = tlapse #TODO: Set as a dictionary instead since different parents will yield in diff tlapses
    self.decay_rate = decay_rate
    self.weight = self.loss(self.tlapse)
    self.loss = self.loss(self.tlapse) #TODO: Set as a dictionary as well
    self.sum = float('inf')
    self.parent = None
    self.path = [] #path from start node to this node container

  def decay(self, time, max_fmeasure):
    fmeasure = max_fmeasure*(math.exp(-self.decay_rate*time))
    return fmeasure

  def loss(self, time, max_fmeasure=100):
    fmeasure = self.decay(time, max_fmeasure)
    loss = (max_fmeasure - fmeasure)**2
    return loss