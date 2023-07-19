#!/usr/bin/env python

from enum import Enum

class areaStatus(Enum):
    IDLE = 0
    DECAYING = 1
    RESTORING_F = 10
    RESTORED_F = 11

class battStatus(Enum):
    IDLE = 0
    DEPLETING = 1
    CHARGING = 10
    FULLY_CHARGED = 11

class robotStatus(Enum):
    IDLE = 0
    READY = 11
    IN_MISSION = 20
    CHARGING = 30
    RESTORING_F = 40

