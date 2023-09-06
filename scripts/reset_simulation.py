#!/usr/bin/env python

# Author: Amel Docena
# Date: May 23, 2023

"""
Acknowledgements to Kizito Masaba for sharing sample codes for killing nodes, although these weren't used eventually.
"""

import os
import subprocess
import signal
import time
import rospy

"""
Running nodes: World file
    
Killing nodes
Reset the simulation
"""

def kill_nodes(sleep):
    """
    Kills all nodes. Note rosout restarts after being killed
    Args:
        nodes_to_kill:
        sleep:

    Returns:

    """
    os.popen('rosnode kill -a')
    os.popen('rosnode cleanup purge')
    time.sleep(sleep)

def launch_nodes(package, launch_file, params, logfile, sleep=None):
    """
    Runs the launch file with params
    Returns:

    """
    inp = ['roslaunch', package, launch_file]
    inp.extend(params)
    main_process = subprocess.Popen(inp)
    main_process.wait()
    if sleep: time.sleep(sleep)

def reset_simulation(launch_file, nodes_to_kill, sleep):
    """
    Kills nodes then launches the world and runs python scripts
    Args:
        path:
        nodes_to_kill:
        world:
        python_scripts:
        sleep:

    Returns:

    """
    kill_nodes(nodes_to_kill, sleep)
    launch_nodes(launch_file, sleep)