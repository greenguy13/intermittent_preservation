# Search-based Strategy for Preservation of Temporal Environmental Properties
Author: Amel Docena

## Nodes
### Robot navigation
Takes care of the navigation, which uses <code>move_base</code>

### Robot decision-making: Tree-based
The mind of the robot:
Determines the areas of interest for restoration  
Subscribes to the F-measures  
Forecasts the decayed F-measure of areas for a given time  
Forecasts the consumed battery in fulfilling the mission/operation  
Computes the loss of the decisions  
Comes up with the optimal set of decisions for a given length of decision steps. The decision consists of travelling to an area, 
which either to restore the F of an area, or charge up battery.

    Publisher:
        voronoi (for determining locations in the map)
        robot navigation goal  
        robot status (PO: IDLE, READY, IN_MISSION, REQUEST2CHARGE, CHARGING, RESTORING_F)
    (PO) Server:
        commence simulation (sends notice to areas to pause or commence simulation)
    Subscriber:
        map
        F-measure from areas
        battery
    Process:
        Think decisions, which includes growing the tree and searching for optimal set of decisions

### Robot battery
Battery node that depletes as the robot moves, and charges up when in charging station

    Publisher: 
        battery level
    Subscriber: 
        robot location
        (PO: server) charge request

### Area
Area node where F decays by some decay function, if simulation is commenced (can be paused while robot is thinking).
If robot starts restoring F, F is restored by some restoration model.

    Publisher:
        F-measure
    Subscriber:
        robot location
        (PO: server) restoring device is ON

## Program methods
### Treebased-decision
*run_operation*
  > robot states (state machine) and act accordingly

*think_decisions*
  > think decisions via grow_tree  
  > send notice to areas to pause simulation

*commence_mission*  
  > send notice to areas to resume simulation  
  > updates robot status as IN-MISSION  
  > if optimal set of decisions is empty, we set status as IDLE  

*send2_next_area*  
  > sends robot to the next area in the optimal path  

*update_robot_status*  
  > updates robot status (self.robot_status)  

*go_to_target*  
  > goes to cartesian (x, y) coords

*compute_gvg*  
  > computes GVG (Voronoi) for exploration

*prune_leaves*  
  > prunes leaves for GVG exploration  

*publish_edges*  
  > publishes edges for GVG exploration  

*select_preservation_areas*  
  > selects areas to preserve from potential nodes collected from GVG

*build_dist_matrix*  
  > builds distance matrix among selected areas for preservation

*static_map_callback*  
  > callback for grid  

*is_charging_station*
  > check whether we are within the radius of the charging station

*robot_nav_callback*  
  > callback after navigator informs arrival of robot to set goal  
  > if charging station: we update status to REQUEST2CHARGE  
  > else: RESTORING_F

*request_charge*  
  > FOR CONFIRMATION/UPDATING: charge up battery

*charge_battery_feedback_cb*  
  > FOR DELETION: feedabck for action request to charge up battery  

*restore_f_request_feedback_cb*  
  > FOR DELETION: feedback for restoring F  

*restore_f_request*  
  > FOR CONFIRMATION/UPDATING: action request to restore F-measure  

*request_fmeasure*  
  > FOR CONFIRMATION/UPDATING: service request to get information on F  

*grow_tree*  
  > grow a decision tree of depth dec_steps starting from where the robot is

*compute_duration*  
  > (for grow_tree) computes (time) duration of operation, which includes travelling distance plus restoration, if any

*consume_battery*  
  > (for grow_tree) consumes curr_battery for the duration of the operation. 
  this duration includes the distance plus F-measure restoration, if any

*adjust_fmeasures*  
  > (for grow_tree) adjusts the F-measures of all areas. The visit area will be restored to max, while the other areas will decay for
        t duration. Note that the charging station is not part of the areas to monitor. And so, if the visit_area is the
        charging station, then all of the areas will decay as duration passes by

*compute_cost_path*  
  > (for grow_tree) computes the cost, (i.e., the sum of losses) of the path  

*get_optimal_branch*  
  > returns the optimal branch of the tree. This shall be the optimal decision path for the robot

### Area
*run_operation*  
  > states of the area: PO paused, decaying, restoring

*robot_status_cb*  
  > FOR CONFIRMATION/UPDATING: call for robot status

*publish_fmeasure*  
  > publishes F-measure as a topic

*raise_fmeasure_cb*  
  > FOR CONFIRMATION/UPDATING: restore F-measure

*report_flevel_cb*  
  > FOR CONFIRMATION/UPDATING: callback as service server for F-measure

*decay*  
  > decay function (PO: modularize)

### Battery
*run_operation*  
  > states of the battery: PO idle, depleting, charging

*charge_battery_cb*  
  > FOR CONFIRMATION/UPDATING: callback as action server to charge up battery. Simply sets a delay in charging the robot up
