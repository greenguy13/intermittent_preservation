#!/usr/bin/env python

import networkx as nx
import numpy as np
import math
from loss_fcns import loss_fcn, decay
import heapq
import project_utils as pu

"Node/vertex in a Graph used in dynamic programming"

class Node():
  def __init__(self, name, id, decay_rate, tlapse_init, tlapse_post_init=0, tlapse_visit=0):
    self.name = name
    self.id = id
    self.tlapse_post_init = 0
    if decay_rate is not None and tlapse_init is not None:
      self.decay_rate = decay_rate
      self.tlapse_post_init = tlapse_post_init + tlapse_visit
      self.tlapse = tlapse_init + self.tlapse_post_init
      self.weight = self.loss(self.tlapse)
      self.loss = self.loss(self.tlapse)
    self.sum = float('inf')
    self.parent = None
    self.path = [] #path from start node to this node container

  def loss(self, time, max_fmeasure=100):
    fmeasure = decay(self.decay_rate, time, max_fmeasure)
    loss = loss_fcn(max_fmeasure, fmeasure)
    return loss

"Label of vertices used in RMA* search"
class Label():
  def __init__(self, vertex, current_loss, tlapse, path, parent=None):
    self.vertex = vertex
    self.tlapse = tlapse
    self.path = path #Note: current location should not be added at initialization?
    self.parent = parent
    self.current_loss = current_loss
    self.heuristic = 0
    self.valuation = self.computeValuation()

  def addToPath(self, add_path):
    self.path.add(add_path)

  def getSuccessors(self, vertices):
    """
    Gets successors given vertices, assuming that the path already traversed will not be visited again.
    This is a main assumption of orienteering problem as cited in the paper.
    """
    succ = vertices - self.path
    return succ

  def computeValuation(self):
    self.valuation = self.current_loss + self.heuristic


#Double criteria priority queue used in RMA* search
class DualCriteriaPriorityQueue:
  def __init__(self):
    self._queue = []
    self._index = 0

  def push(self, label):
    heapq.heappush(self._queue, (-label.valuation, label.tlapse, self._index, label))
    self._index += 1

  def pop(self):
    # Remove and return the highest priority item
    return heapq.heappop(self._queue)[-1]

  def is_empty(self):
    # Return True if the queue is empty
    return len(self._queue) == 0

  def peek(self):
    # Return the highest priority item without removing it
    return self._queue[0][-1] if self._queue else None

#Frontier with dominance rule used in RMA* search
class Frontier():
  def __init__(self, length=math.inf):
    self.frontier = set()
    self.length = length  # Length of the frontier

  def add(self, label):
    self.frontier.add(label)

  def remove(self, label):
    self.frontier.remove(label)

  def filterAddFront(self, label):
    """
    Filters new label by dominance for any labels in the frontier. If it dominates any of the existing, the new label replaces the former.
    If it does not dominate any of the existing label, it is added in the frontier. The frontier, therefore, is a set of non-dominated labels
    """
    for label_prime in self.frontier.copy():
      # THIS IS OK
      if len(label_prime.path) == 0 or (label.path <= label_prime.path and label.current_loss >= label_prime.current_loss):  # label_prime is dominated by label, so we remove the former, add the latter
        # debug("Existing label removed from frontier: {}, {}".format(label_prime.vertex, label_prime.path))
        self.frontier.remove(label_prime)
    if len(self.frontier) < self.length:
      self.frontier.add(label)

def debug(msg, robot_id=0):
  pu.log_msg('robot', robot_id, msg, debug=True)