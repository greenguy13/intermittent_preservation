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
import roslaunch

"""
Running nodes: World file
    
Killing nodes
Reset the simulation
"""

# def kill_nodes(sleep):
#     """
#     Kills all nodes. Note rosout restarts after being killed
#     Args:
#         nodes_to_kill:
#         sleep:
#
#     Returns:
#
#     """
#     os.popen('rosnode kill -a')
#     os.popen('rosnode cleanup purge')
#     time.sleep(sleep)
#
# def launch_nodes(package, launch_file, params, logfile=None, sleep=None):
#     """
#     Runs the launch file with params
#     Returns:
#
#     """
#     inp = ['roslaunch', package, launch_file]
#     inp.extend(params)
#     print(inp)
#     #main_process = subprocess.Popen(inp, shell=True)
#     #main_process.wait()
#
#     os.system(" ".join(inp))
#     if sleep: time.sleep(sleep)
#
# def reset_simulation(launch_file, nodes_to_kill, sleep):
#     """
#     Kills nodes then launches the world and runs python scripts
#     Args:
#         path:
#         nodes_to_kill:
#         world:
#         python_scripts:
#         sleep:
#
#     Returns:
#
#     """
#     kill_nodes(nodes_to_kill, sleep)
#     launch_nodes(launch_file, sleep)

## Alternative: Using roslaunch
def kill_nodes(sleep):
    """
    Kills all nodes. Note rosout restarts after being killed
    Args:
        nodes_to_kill:
        sleep:

    Returns:

    """
    # launch.shutdown()
    os.popen('rosnode kill -a')
    os.popen('rosnode cleanup purge')
    time.sleep(sleep)

def launch_nodes(package, launch_file, params, logfile=None, sleep=None):
    """
    Runs the launch file with params
    Returns:

    """
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    cli_args = [launch_file]
    cli_args.extend(params)
    print("cli_args", cli_args)
    roslaunch_args = cli_args[1:]

    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
    parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    parent.start()

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