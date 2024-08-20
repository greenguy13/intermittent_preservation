# Persistent Preservation of a Spatio-temporal Environment Under Uncertainty
Authors: anonymized

# Requirements

Here are the requirements to run the full experiments:

- ROS noetic
- stageros
- move_base
- amcl

and add this ROS package to a `catkin_ws`.


# Example run
First, sample a number of areas from the map in `run_experiment.py`. Choose between 'office' and 'cluttered', which are 'sdr-b' 
and 'grass' referred to in the paper, respectively.
```
worlds = ['office', 'cluttered]
nareas_list = [12] #Number of areas
nplacements = 2000 #Number of random placements of the areas
batch_sample_nodes_poses(worlds, nareas_list, nplacements)
```
Once sampled, comment the block of codes.

To modify the settings of the experiments, modify the line:

```
run_experiment('heuristic_uncertainty', 'cluttered', 12, placement, 'non_uniform', 2100,
                   inference='timeseries', dec_steps=4, discount=0.75, exploration=0.60, nvisits=2, ntrials=(0,3), save=True)
```

To run the experiments in batch:
```
python ~/catkin_ws/src/intermittent_preservation/scripts/run_experiment.py
```

## Nodes
### Robot navigation
Takes care of the navigation, which uses `move_base` and `amcl`

### Robot decision-making: treebased, heuristic_uncertainty, correlated_ucb, dynamic_programming, rma
The mind of the robot: Determines the areas of interest for restoration

### Robot battery
Battery node that depletes as the robot moves, and charges up when in charging station

### Area
Area node where F decays by some decay function, if simulation is commenced (paused while robot is thinking).
If robot starts restoring F, F is restored by some restoration model.