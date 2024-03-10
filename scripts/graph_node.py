#!/usr/bin/env python3

import networkx as nx
import numpy as np
import math
from loss_fcns import loss_fcn, decay
"Node/vertex in a Graph used in dynamic programming"

class Node():
  def __init__(self, name, id, decay_rate, tlapse_init, tlapse_post_init=0, tlapse_visit=0):
    self.name = name
    self.id = id
    self.tlapse_post_init = tlapse_post_init + tlapse_visit
    self.tlapse = tlapse_init + self.tlapse_post_init
    self.decay_rate = decay_rate
    self.weight = self.loss(self.tlapse)
    self.loss = self.loss(self.tlapse)
    self.sum = float('inf')
    self.parent = None
    self.path = [] #path from start node to this node container

  def loss(self, time, max_fmeasure=100):
    fmeasure = decay(self.decay_rate, time, max_fmeasure)
    loss = loss_fcn(max_fmeasure, fmeasure)
    return loss