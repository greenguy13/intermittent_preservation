#!/usr/bin/env python


import rospy
from time import process_time
import pickle
import numpy as np
import actionlib
from pruning import *
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from std_msgs.msg import Int8, Float32
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# from int_preservation.srv import clusterAssignment
from int_preservation.srv import clusterAssignment2
from int_preservation.srv import assignmentAccomplishment, assignmentAccomplishmentResponse
from int_preservation.srv import registerRobot, registerRobotResponse
from int_preservation.srv import pauseSimulation
from int_preservation.srv import collectedEnoughData, collectedEnoughDataResponse
from status import centralStatus, battStatus, robotStatus, robotAssignStatus
from reset_simulation import *
from heuristic_fcns import *
from loss_fcns import *
import math
from math import ceil
import numpy as np
from collections import defaultdict

# SciPy clustering & assignment
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3  # GoalStatus ID for succeeded, http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html
SHUTDOWN_CODE = 99

def _exp_decay(F_max, rate, t):
    """Exponential decay g(δ, t)."""
    return F_max * math.exp(-rate * max(0.0, t))

def _mean_duration_excluding_col(M, j):
    """
    taū_j := average duration entries when the 'committed' choice is NOT j.
    (Delete column j per our definition and average the remainder.)
    If you have per-robot matrices, average across them outside.
    """
    sub = np.delete(M, j, axis=1)
    return float(np.mean(sub)) if sub.size > 0 else 0.0

class CentralPlanner:
    def __init__(self, node_name):
        """

        :param node_name:
        """

        rospy.init_node(node_name, anonymous=True)

        # Parameters
        self.nrobots = rospy.get_param("/nrobots")
        self.robot_ids = [i for i in range(self.nrobots)]
        self.debug_mode = rospy.get_param("/debug_mode")
        self.robot_velocity = rospy.get_param("/robot_velocity")  # Linear velocity of robot; we assume linear and angular are relatively equal
        self.gamma = rospy.get_param("/gamma")  # discount factor
        self.max_fmeasure = rospy.get_param("/max_fmeasure")  # Max F-measure of an area
        self.max_battery = rospy.get_param("/max_battery")  # Max battery
        self.battery_reserve = rospy.get_param("/battery_reserve")  # Battery reserve
        self.tolerance = rospy.get_param("/move_base_tolerance")
        self.charging_station = 0  # charging station index

        f_thresh = rospy.get_param("/f_thresh")
        self.fsafe, self.fcrit = f_thresh  # (safe, crit)

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_consumed_per_travel_time, self.batt_consumed_per_restored_f = batt_consumed_per_time  # (travel, restoration)

        self.dec_steps = rospy.get_param("/dec_steps")  # STAR
        self.restoration = rospy.get_param("/restoration")
        self.noise = rospy.get_param("/noise")
        self.nareas = rospy.get_param("/nareas")  # Sample nodes from voronoi equal to area count #STAR
        self.areas = [int(i + 1) for i in range(self.nareas)]  # list of int area IDs

        # Crisis mitigation with surge binding release
        # surge_bindings: robot_id -> {'areas': set[int], 't_bind': int, 'home_cid': str|None, 'dwell_cap': int}
        self.surge_bindings = {}
        self.area_locks = {}  # area_id -> rid that currently locks it (surge)
        self.last_reset_step = {aid: -1 for aid in self.areas}  # Track last time an area was restored (tlapse reset)

        # self.check_pause()
        self.requested_pause = False  # indicator variable whether requested Stage to pause simulation

        self.debug("Nareas {}. Areas list: {}".format(self.nareas, self.areas))
        # self.tolerance = rospy.get_param("/move_base_tolerance")
        self.t_operation = rospy.get_param("/t_operation")  # total duration of the operation
        self.save = rospy.get_param("/save")  # Whether to save data
        self.crisis_mitigation = rospy.get_param("/crisis_mitigation") # Crisis mitigation toggle
        self.debug("Crisis mitigation: {}".format(self.crisis_mitigation))

        # Pickle load the sampled area poses
        self.areas_poses = list()
        with open('{}.pkl'.format(rospy.get_param("/file_sampled_areas")), 'rb') as f:
            sampled_areas_coords = pickle.load(f)
        for area_coords in sampled_areas_coords['n{}_p{}'.format(self.nareas, rospy.get_param("/placement"))]:
            pose_stamped = pu.convert_coords_to_PoseStamped(area_coords)
            self.areas_poses.append(pose_stamped)

        self.dist_matrices = dict()  # Initialize distance matrices of each registered robot to the areas + charging station
        for robot in range(self.nrobots):
            self.dist_matrices[robot] = None

        self.tau_bar = None

        self.charging_station = 0

        # Initialize variables/containers
        self.mission_areas = dict()  # Mission areas of robots
        for robot_id in self.robot_ids:
            self.mission_areas[robot_id] = None

        self.assign_statuses = dict()  # Assignment statuses of robots
        for robot_id in self.robot_ids:
            self.assign_statuses[robot_id] = None

        self.tlapses = dict()  # Tlapses of areas
        for area in self.areas:
            self.tlapses[area] = 0

        self.robot_statuses = dict()  # Robot statuses
        for robot_id in self.robot_ids:
            self.robot_statuses[robot_id] = None

        self.robots_location = dict()  # Robots location
        for robot_id in self.robot_ids:
            self.robots_location[robot_id] = None

        self.robots_battery = dict()  # Robots battery
        for robot_id in self.robot_ids:
            self.robots_battery[robot_id] = None

        self.decay_rates = dict()  # Decay rates
        for area in self.areas:
            self.decay_rates[area] = None

        self.collected_enough_data = dict()  # Whether we have collected enough data for shutdown
        for area in self.areas:
            self.collected_enough_data[area] = False

        self.clusters = None  # Clustering of areas
        self.clusters_assignment = dict()  # Assignment of clusters (keys) to robots (values)
        self.robots_assignment = dict()  # Assignment of robots (keys) to clusters (values)
        self.unassigned_clusters = list()  # List of unassigned clusters
        self.unassigned_robots = list()  # List of unassigned robots

        # Server
        self.robots_registry_server = rospy.Service('/robots_registry_server', registerRobot, self.register_robots_cb)
        self.assignment_accomplishment_server = rospy.Service('/assignment_accomplishment_server',
                                                              assignmentAccomplishment,
                                                              self.assignment_accomplishment_cb)
        self.shutdown_server = rospy.Service('/shutdown_server', collectedEnoughData, self.collected_enough_data_cb)

        # Publishers/Subscribers
        self.central_status_pub = rospy.Publisher('/central_status', Int8, queue_size=1)

        # Service request to move_base to get plan : make_Plan
        server = '/robot_0/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug("Getplan service: {}".format(self.get_plan_service))

        for robot_id in self.robot_ids:
            # rospy.Subscriber('/robot_{}/assignment_status'.format(robot_id), Int8, self.assign_status_cb, robot_id) #TODO: To be updated. This is for re-assignment
            rospy.Subscriber('/robot_{}/mission_area'.format(robot_id), Int8, self.mission_area_cb, robot_id)
            rospy.Subscriber('/robot_{}/robot_status'.format(robot_id), Int8, self.robot_status_cb, robot_id)
            rospy.Subscriber('/robot_{}/location'.format(robot_id), Int8, self.robot_location_cb, robot_id)
            rospy.Subscriber('/robot_{}/battery'.format(robot_id), Float32, self.robot_battery_cb, robot_id)

        # Here: It is assumed oracle knoweldge of decay rates
        for area in self.areas:
            rospy.Subscriber('/area_{}/decay_rate'.format(area), Float32, self.decay_rate_cb, area)

    def register_robots_cb(self, msg):
        """
        Register robots id
        :return:
        """
        robot_id = msg.robot_id  # robot id for registration
        init_x = msg.init_x
        init_y = msg.init_y

        self.debug("Registry request received (id, x, y): {}, {}, {}".format(robot_id, init_x, init_y))

        # Build distance matrix for that robot
        self.dist_matrices[robot_id] = self.build_dist_matrix(robot_id, float(init_x), float(init_y))
        self.robots_location[robot_id] = 0  # initial location

        # Debug that robot has been registered
        self.debug("Robot registered: {}. Dist matrix: {}".format(robot_id, self.dist_matrices[robot_id]))

        return registerRobotResponse(True)

    # METHODS: Node poses and distance matrix
    def get_plan_request(self, start_pose, goal_pose, tolerance):
        """
        Sends a request to GetPlan service to create a plan for path from start to goal without actually moving the robot
        :param start_pose:
        :param goal_pose:
        :param tolerance:
        :return:
        """
        req = GetPlan()
        req.start = start_pose
        req.goal = goal_pose
        req.tolerance = tolerance
        server = self.get_plan_service
        result = server(req.start, req.goal, req.tolerance)
        path = result.plan.poses
        return path

    def decouple_path_poses(self, path):
        """
        Decouples a path of PoseStamped poses; returning a list of x,y poses
        :param path: list of PoseStamped
        :return:
        """
        list_poses = list()
        for p in path:
            x, y = p.pose.position.x, p.pose.position.y
            list_poses.append((x, y))
        return list_poses

    def compute_path_total_dist(self, list_poses):
        """
        Computes the total path distance
        :return:
        """
        total_dist = 0
        for i in range(len(list_poses) - 1):
            dist = math.dist(list_poses[i], list_poses[i + 1])
            total_dist += dist
        return total_dist

    def compute_dist_bet_areas(self, area_i, area_j, tolerance):
        """
        Computes the distance between area_i and area_j:
            1. Call the path planner between area_i and area_j
            2. Decouple the elements of path planning
            3. Compute the distance then total distance
        :param area_i: PoseStamped
        :param area_j: PoseStamped
        :return:
        """
        path = self.get_plan_request(area_i, area_j, tolerance)
        list_poses = self.decouple_path_poses(path)
        total_dist = self.compute_path_total_dist(list_poses)
        return total_dist

    def build_dist_matrix(self, robot_id, init_x, init_y):
        """
        Builds the distance matrix among areas
        :return:
        """
        charging_station_coords = (init_x, init_y)
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        nodes_poses = [charging_pose_stamped]
        nodes_poses.extend(self.areas_poses)
        self.debug("Robot: {}. Nodes_poses: {}".format(robot_id, nodes_poses))

        n = len(nodes_poses)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                area_i, area_j = nodes_poses[i], nodes_poses[j]
                if area_i != area_j:
                    dist = self.compute_dist_bet_areas(area_i, area_j, self.tolerance)
                    dist_matrix[i, j] = dist

        # self.debug("Dist matrix: {}".format(self.dist_matrix))
        return dist_matrix

    def robot_location_cb(self, msg, robot_id):
        """
        Receives and stores robot location
        :param msg:
        :param robot_id:
        :return:
        """
        curr_loc = msg.data
        self.robots_location[robot_id] = int(curr_loc)

    def robot_battery_cb(self, msg, robot_id):
        """
        Receives and stores robot battery
        :param msg:
        :param robot_id:
        :return:
        """
        battery = msg.data
        self.robots_battery[robot_id] = int(battery)

    def assignment_accomplishment_cb(self, msg):
        """
        Receives area accomplishment from robots. Central then updates tlapse for that area
        :return:
        """
        # Tlapse reset if area is assigned area
        robot_id = msg.robot_id
        area_id = msg.area_accomplished
        # Reset tlapse and stamp for surge-release logic
        self.tlapses[area_id] = 0
        self.on_area_restored(area_id)
        self.note_surge_visit(rid=int(robot_id), aid=int(area_id))
        self.debug("Received notice from Robot: {} restored Area: {}. Tlapse reset: {}".format(robot_id, area_id,
                                                                                               self.tlapses[area_id]))
        return assignmentAccomplishmentResponse(True)

    def assign_status_cb(self, msg, robot_id):
        """
        Updates the assignment status of robot from subscribed topic
        :param msg:
        :param robot_id:
        :return:
        """
        assign_status = msg.data
        self.assign_statuses[robot_id] = int(assign_status)

    def mission_area_cb(self, msg, robot_id):
        """
        Updates the mission area of robot from subscribed topic
        :param msg:
        :param robot_id:
        :return:
        """
        mission_area = msg.data
        self.mission_areas[robot_id] = int(mission_area)

    def robot_status_cb(self, msg, robot_id):
        """
        Updates the robots statuses.
        Moreover, updates the tlapses of areas based on robot's status and mission area/assignment status
        :param msg:
        :param robot_id:
        :return:
        """
        robot_status = msg.data
        self.robot_statuses[robot_id] = int(robot_status)

    def decay_rate_cb(self, msg, area_id):
        """
        Store decay rate
        :param msg:
        :param area_id:
        :return:
        """
        if self.decay_rates[area_id] == None:
            self.debug("Area {} decay rate: {}".format(area_id, msg.data))
            self.decay_rates[area_id] = msg.data

    def retrieve_tlapses(self, areas):
        """
        Retrieves the tlapses of areas
        :param areas:
        :return:
        """
        tlapses = list()
        for area in areas:
            tlapses.append(int(self.tlapses[area]))
        return tlapses

    def retrieve_decay_rates(self, areas):
        """
        Retrieves the decay rates of areas
        :param areas:
        :return:
        """
        decay_rates = list()
        for area in areas:
            decay_rates.append(self.decay_rates[area])
        return decay_rates

    def update_tlapses_areas(self, dt=1):
        """Advance tlapses for all areas by dt. (No crisis logic here.)"""
        for aid in self.areas:
            self.tlapses[aid] += int(dt)

    # Different methods for Heirarchichal approach with crisis mitigation
    ## Clustering
    def create_clusters(self, alpha_decay=1.0, seed=42, n_clusters=None):
        """
        Build jurisdictions using SciPy kmeans2 on features [x, y, alpha*decay].
        Returns: dict {cluster_name: [area_ids]}
        """
        n_clusters = n_clusters or self.nrobots

        # Build feature matrix
        feats, ids = [], []
        for aid in self.areas:
            rate = float(self.decay_rates.get(aid, 0.0))
            p = self.areas_poses[aid - 1].pose.position  # areas are 1..N; poses list is 0-based
            feats.append([p.x, p.y, alpha_decay * rate])
            ids.append(aid)
        X = np.asarray(feats, dtype=float)

        # SciPy kmeans2
        # kmeans2 can be a bit sensitive to duplicate points; use minit='++' where available
        centroids, labels = kmeans2(X, k=n_clusters, minit='++', iter=50, thresh=1e-05)

        clusters = {}
        for c in range(n_clusters):
            clusters[f"C{c + 1}"] = [ids[i] for i, lab in enumerate(labels) if int(lab) == c]

        self.clusters = clusters
        return clusters

    def create_urgent_clusters(self, rho, K, alpha_decay=1.0, seed=42):
        """
        Cluster the urgent set rho into K groups using SciPy kmeans2 on [x, y, alpha*decay].
        Returns: dict { 'SURG_1': [area_ids], ... }
        """
        if not rho or K <= 0:
            return {}

        pts, idx2aid = [], []
        for aid in rho:
            p = self.areas_poses[aid - 1].pose.position
            pts.append([p.x, p.y, alpha_decay * float(self.decay_rates[aid])])
            idx2aid.append(aid)

        X = np.asarray(pts, dtype=float)
        K = min(K, max(1, len(rho)))
        _, labels = kmeans2(X, k=K, minit='++', iter=50, thresh=1e-05)

        clusters = {}
        for c in range(K):
            clusters[f"SURG_{c + 1}"] = [idx2aid[i] for i, lab in enumerate(labels) if int(lab) == c]
        return clusters

    def _npv_cluster_loss(self, area_ids, z, gamma, tau_bar, decay_rates, F_max, delta_t=1):
        """
        NPV_c = Σ_j Σ_{h=1..H_j} gamma^h * max(0, z - g(δ_j, h*delta_t)),
        with H_j = |c|
        """
        if not area_ids:
            return 0.0
        size = max(1, len(area_ids))
        tot = 0.0
        self.debug("Calculating NPV cluster: {}".format(area_ids))
        for aid in area_ids:
            H = int(size)
            rate = decay_rates[aid]
            for h in range(1, H + 1):
                F_pred = _exp_decay(F_max, rate, h * float(tau_bar.get(aid, 0.0)))
                # loss = max(0.0, z - F_pred)
                loss = loss_fcn(max_fmeasure=F_max, decayed_fmeasure=F_pred)
                tot += (gamma ** h) * loss
                self.debug("Area: {}. Loss: {}. Running total: {}".format(aid, loss, tot))
        return tot

    # ------------------------------------------------------------------
    # NEW: Predict Highly Urgent Areas (predicted subsumes actual)
    # ------------------------------------------------------------------
    def predict_highly_urgent_areas(self, k=1):
        """
        Predict areas that will be at/below critical threshold within k * tau_bar[a].
        Uses current tlapse as starting point. Returns a SET of area ids.
        """
        if self.tau_bar is None:
            self.tau_bar = self._tau_bar_from_robot_mats()

        rho = set()
        for aid in self.areas:
            tau = float(self.tau_bar.get(aid, 0.0))
            horizon = max(0.0, k * tau)
            rate = float(self.decay_rates[aid])
            t_future = float(self.tlapses[aid]) + horizon
            F_future = self.max_fmeasure * math.exp(-rate * t_future)
            if F_future <= self.fcrit:
                rho.add(aid)
        self.debug(f"[URGENT-PRED] k={k} -> |rho|={len(rho)}")
        return rho

    # ------------------------------------------------------------------
    # NEW: Internal Overload (strict majority > 50% + ≥1 actual urgent)
    # ------------------------------------------------------------------
    def internal_overload(self, cid, rho):
        """
        A cluster C is overloaded if:
          - strictly more than half of its areas are predicted highly urgent, and
          - there exists at least one actually urgent area in C.
        """
        areas = self.clusters.get(cid, [])
        if not areas:
            return False

        urgent = [a for a in areas if a in rho]
        urgent_count = len(urgent)
        cluster_size = len(areas)
        majority_urgent = urgent_count > (cluster_size / 2.0)
        actual_urgent = any(self._area_current_F(a) <= self.fcrit for a in areas)

        overloaded = majority_urgent and actual_urgent
        self.debug(f"[INTERNAL-OVERLOAD] cid={cid} majority={majority_urgent} "
                   f"actual={actual_urgent} | urgent={urgent_count}/{cluster_size} -> {overloaded}")
        return overloaded

    # ------------------------------------------------------------------
    # NEW: External Feasibility (τ̄ from assigned robot's matrix average)
    # ------------------------------------------------------------------
    def external_feasibility(self, cid, rho):
        """
        True if at least one external robot can arrive before the home robot would
        finish visiting all predicted highly urgent areas (h = |U(C)|).
        τ̄(C) is taken as the average of the assigned home robot's duration matrix.
        """
        areas = self.clusters.get(cid, [])
        if not areas:
            return False

        # h = number of predicted highly urgent in this cluster
        h = len([a for a in areas if a in rho])
        if h == 0:
            return False

        home_rid = self.clusters_assignment.get(cid, None)
        if home_rid is None:
            # If we can't identify a home robot, we can't compare rates
            return False

        M_home = self.dist_matrices.get(home_rid, None)
        if M_home is None:
            return False

        tau_bar = float(np.mean(M_home)) if M_home.size else float('inf')

        # Anchor = most urgent predicted (min predicted F); fallback to min current F
        anchor = self._most_urgent_anchor(areas)
        if anchor is None:
            return False

        feasible_arrivals = []
        for rid in self.robot_ids:
            if rid == home_rid:
                continue
            M = self.dist_matrices.get(rid, None)
            src = self.robots_location.get(rid, None)
            if M is None or src is None:
                continue
            src = int(src)
            if 0 <= src < M.shape[0] and 0 <= anchor < M.shape[1]:
                travel_time = float(M[src, anchor])
                # Convert travel time to battery cost; require sufficient battery
                req_batt = travel_time * float(self.batt_consumed_per_travel_time)
                have_batt = float(self.robots_battery.get(rid, 0.0))
                if have_batt >= req_batt:
                    feasible_arrivals.append(travel_time)

        if not feasible_arrivals:
            self.debug(f"[EXTERNAL-FEAS] cid={cid} no feasible external arrivals")
            return False

        t_ext = min(feasible_arrivals)
        visits_home = t_ext / max(1e-6, tau_bar)
        feasible = (visits_home < float(h))
        self.debug(f"[EXTERNAL-FEAS] cid={cid} h={h} t_ext={t_ext:.3f} tau_bar={tau_bar:.3f} "
                   f"visits_home≈{visits_home:.2f} -> {feasible}")
        return feasible

    # ------------------------------------------------------------------
    # NEW: Form Surge Robots (base + potential; battery & economics)
    # ------------------------------------------------------------------
    def form_surge_robots(self, marked_cids, rho):
        """
        Base surge robots: home robots of marked clusters.
        Potential surge robots: not in base, with NPV_home < avg surge NPV per available robot,
        and with enough expected energy budget for surge.
        Returns (S, Gamma) where S=list[rids], Gamma=set[area ids] predicted urgent in marked clusters.
        """
        # Base robots
        base = []
        for C in marked_cids:
            rid = self.clusters_assignment.get(C, None)
            if rid is not None and rid not in base:
                base.append(int(rid))

        # Γ = predicted highly urgent across marked clusters
        Gamma = set()
        for C in marked_cids:
            Gamma.update(a for a in self.clusters.get(C, []) if a in rho)

        # If Γ size <= |base|, only base is needed
        if len(Gamma) <= len(base):
            self.debug(f"[SURGE-FORM] Gamma={len(Gamma)} <= |base|={len(base)} -> base only")
            return base, Gamma

        # Compute NPV(Γ)
        npv_gamma = self._npv_cluster_loss(list(Gamma), self.fcrit, self.gamma,
                                           self.tau_bar or self._tau_bar_from_robot_mats(),
                                           self.decay_rates, self.max_fmeasure)
        # Available non-base robots
        avail = [rid for rid in self.robot_ids if rid not in base]
        A = max(1, len(avail))
        # Average surge NPV per available robot (cap by min(R, |Gamma|) implicitly later)
        avg_surge = npv_gamma / float(A)

        def home_npv(rid):
            home_cid = self.robots_assignment.get(rid, None)
            areas = self.clusters.get(home_cid, []) if home_cid else []
            return self._npv_cluster_loss(areas, self.fcrit, self.gamma,
                                          self.tau_bar or self._tau_bar_from_robot_mats(),
                                          self.decay_rates, self.max_fmeasure)

        # Expected surge energy budget per robot (approx):
        # expected cluster size ~ ceil(|Γ| / A); cost ~ mean tau over Γ mapped to battery
        taus = [float(self.tau_bar.get(a, 0.0)) for a in Gamma] if Gamma else [0.0]
        avg_tau = float(np.mean(taus)) if taus else 0.0
        expected_size = int(math.ceil(len(Gamma) / float(A)))
        expected_cost = expected_size * (avg_tau * float(self.batt_consumed_per_travel_time))

        potential = []
        for rid in avail:
            hn = home_npv(rid)
            have_batt = float(self.robots_battery.get(rid, 0.0))
            if (hn < avg_surge) and (have_batt >= expected_cost):
                potential.append((hn, rid))

        potential.sort(key=lambda x: x[0])  # ascending by NPV_home
        S = base + [rid for _, rid in potential]

        # Cap by |Γ| (minimum 1 area per surge robot)
        cap = len(Gamma)
        if len(S) > cap:
            S = S[:cap]

        self.debug(f"[SURGE-FORM] base={base} potential={[rid for _, rid in potential]} "
                   f"npvΓ={npv_gamma:.3f} avg_surge≈{avg_surge:.3f} expected_cost≈{expected_cost:.3f} "
                   f"-> S={S} |Γ|={len(Gamma)}")
        return S, Gamma

    # ------------------------------------------------------------------
    # NEW: Surge Binding (uses existing assign_clusters; no duplicate code)
    # ------------------------------------------------------------------
    def bind_surge_robots(self, assignment, dwell_cap='default', tag_prefix="SURG_"):
        """
        assignment: rid -> surge_area_ids (already grouped into surge clusters).
        We re-use assign_clusters(..., surge=True) for dispatch; this method is kept
        for symmetry/clarity but it simply calls assign_clusters appropriately.
        """
        # Convert rid->areas to {cid: areas} keyed by SURG_# so Hungarian can run if needed
        clusters = {}
        for idx, (rid, areas) in enumerate(assignment.items(), start=1):
            clusters[f"{tag_prefix}{idx}"] = list(map(int, areas))
        self.assign_clusters(clusters,
                             method='hungarian',
                             candidate_rids=list(assignment.keys()),
                             tag_prefix=tag_prefix,
                             surge=True,
                             dwell_cap=dwell_cap)

    # ------------------------------------------------------------------
    # NEW: Release Surge Robots (OR condition; Hungarian re-dispatch home)
    # ------------------------------------------------------------------
    def release_surge_robots(self, rho, dwell_cap_override=None):
        """
        Release when EITHER:
          - surge cluster is no longer a highly urgent cluster (NOT internal_overload), OR
          - dwell cap (time-based) elapsed: sim_t >= t_bind + dwell_cap.
        On release: unlock surge areas and re-dispatch robot to UPDATED home jurisdiction
        (exclude areas still locked by other surge robots) using Hungarian.
        """
        if not self.surge_bindings:
            return

        to_release = []
        for rid, b in list(self.surge_bindings.items()):
            curr_cid = self.robots_assignment.get(rid, None)
            if curr_cid is None:
                continue

            # Check urgency of the current surge cluster
            still_urgent = self.internal_overload(curr_cid, rho)

            # Time-based dwell cap
            cap = int(dwell_cap_override) if (dwell_cap_override is not None) else int(b.get('dwell_cap', len(b.get('areas', [])) or 1))
            t_bind = int(b.get('t_bind', self.sim_t))
            dwell_elapsed = (int(self.sim_t) >= int(t_bind) + cap)

            if (not still_urgent) or dwell_elapsed:
                reason = 'no_longer_urgent' if (not still_urgent) else 'dwell_cap'
                to_release.append((rid, reason))

        for rid, reason in to_release:
            b = self.surge_bindings.pop(rid, None)
            if not b:
                continue

            # Unlock areas owned by this surge robot
            for aid in list(b.get('areas', set())):
                if self.area_locks.get(int(aid)) == int(rid):
                    self.area_locks.pop(int(aid), None)

            home_cid = b.get('home_cid', None)
            if home_cid is None:
                # No home cluster recorded; mark unassigned
                if rid not in self.unassigned_robots:
                    self.unassigned_robots.append(rid)
                self.robots_assignment[rid] = None
                self.debug(f"[SURGE-RELEASE] rid={rid} reason={reason} -> no home cluster; unassigned")
                continue

            # Updated home: exclude areas still locked by other surge robots
            full_home = list(map(int, self.clusters.get(home_cid, [])))
            still_locked = {a for bb in self.surge_bindings.values() for a in bb.get('areas', set())}
            updated_home = [a for a in full_home if a not in still_locked]

            self.debug(f"[SURGE-RELEASE] rid={rid} reason={reason} -> home={home_cid} "
                       f"free={len(updated_home)}/{len(full_home)} sim_t={self.sim_t}")

            # Re-dispatch (Hungarian) home with updated areas
            if updated_home:
                self.assign_clusters({home_cid: updated_home},
                                     method='hungarian',
                                     candidate_rids=[rid],
                                     tag_prefix="C",
                                     surge=False)
            else:
                # If nothing free, mark unassigned; next cycles will fill when locks clear
                if rid not in self.unassigned_robots:
                    self.unassigned_robots.append(rid)
                self.robots_assignment[rid] = None

    # ------------------------------------------------------------------
    # CRISIS CYCLE (replaces median-NPV detection for surge triggering)
    # ------------------------------------------------------------------
    def _crisis_cycle(self, k=1, eta=1.0, b=3, delta_t=1.0, alpha_decay=1.0, surge_assign_method='hungarian',
                      dwell_cap=None):
        """
        One crisis-mitigation pass:
          - Predict highly urgent areas ρ
          - Mark clusters needing external help: InternalOverload && ExternalFeasibility
          - Form surge team (base + potential)
          - Create surge clusters over Γ = union of predicted urgent among marked clusters
          - Assign via Hungarian and bind
          - On subsequent ticks, release surge robots under OR condition
        """
        # Build tau_bar once
        if self.tau_bar is None:
            self.tau_bar = self._tau_bar_from_robot_mats()

        # Always predict current ρ
        rho = self.predict_highly_urgent_areas(k=k)

        if not self.surge_bindings:
            # Detect marked clusters
            marked = []
            for cid, area_ids in (self.clusters or {}).items():
                if not cid or cid.startswith("SURG_"):
                    continue  # consider only home jurisdictions
                if self.internal_overload(cid, rho) and self.external_feasibility(cid, rho):
                    marked.append(cid)

            if not marked:
                return  # nothing to do this tick

            # Form surge team
            S, Gamma = self.form_surge_robots(marked, rho)
            if not S:
                self.debug("[SURGE] No feasible surge team")
                return

            # Create surge subclusters from Γ and assign to |S|
            urg_list = list(Gamma)
            K = min(max(1, len(S)), len(urg_list))
            surg_clusters = self.create_urgent_clusters(urg_list, K, alpha_decay=alpha_decay, seed=42)

            if not surg_clusters:
                self.debug("[SURGE] Failed to create surge clusters")
                return

            # Assign surge clusters to surge robots via Hungarian
            self.assign_clusters(clusters=surg_clusters,
                                 method='hungarian',
                                 candidate_rids=S,
                                 tag_prefix="SURG_",
                                 surge=True,
                                 dwell_cap=(len(urg_list) if (dwell_cap in (None, 'default')) else int(dwell_cap)))
        else:
            # Try to release surge robots (OR condition)
            self.release_surge_robots(rho=rho, dwell_cap_override=dwell_cap)

    ## Cluster-robot assignment
    def _area_current_F(self, aid):
        rate = float(self.decay_rates[aid])
        t = float(self.tlapses[aid])
        return self.max_fmeasure * math.exp(-rate * t)

    def _most_urgent_anchor(self, area_ids):
        if not area_ids:
            return None
        # Anchor as min predicted F over 1*tau horizon (ranking consistency)
        best = None
        bestF = float('inf')
        for a in area_ids:
            tau = float(self.tau_bar.get(a, 0.0))
            rate = float(self.decay_rates.get(a, 0.0))
            F0 = self._area_current_F(a)
            F_pred = F0 * math.exp(-rate * max(0.0, tau))
            if F_pred < bestF:
                bestF = F_pred
                best = a
        return best

    def _closest_robot_to_area(self, candidate_rids, area_id):
        best, best_cost = None, float('inf')
        for rid in candidate_rids:
            M = self.dist_matrices.get(rid, None)
            src = self.robots_location.get(rid, None)
            if M is None or src is None:
                continue
            cost = float(M[int(src), int(area_id)])
            if cost < best_cost:
                best, best_cost = rid, cost
        return best, best_cost

    def assign_clusters(self,
                        clusters,
                        method='greedy',
                        candidate_rids=None,
                        tag_prefix="C",
                        surge=False,
                        dwell_cap='default'):
        """
        Assign clusters to robots.

        - method: 'greedy' or 'hungarian'
        - candidate_rids: optional subset of robot ids to consider (e.g., surge set Ω).
          If None, uses all currently unassigned robots (or all robots if none tracked).
        - tag_prefix: prefix for cluster names when recording, e.g., "C" or "SURG_"
        - surge: if True, we record sticky surge bindings with dwell/visit fields and area locks
        - dwell_cap: for surge bindings:
            * 'default' or None  -> len(area_ids)
            * int                -> explicit cap (timesteps if used with release_surge_robots)
        """
        self.debug(f"Assigning clusters (method={method}, surge={surge}): {clusters}")

        # 1) Choose an anchor (most urgent area) for each cluster
        cluster_ids = list(clusters.keys())
        anchors = {cid: self._most_urgent_anchor(clusters[cid]) for cid in cluster_ids}
        valid_cids = [cid for cid in cluster_ids if anchors[cid] is not None]
        if not valid_cids:
            self.debug("No valid clusters to assign (no anchors).")
            return

        # 2) Which robots can we use?
        if candidate_rids is None:
            candidate_rids = (self.unassigned_robots.copy() or self.robot_ids.copy())

        # Ensure lock/binding maps exist
        if surge and not hasattr(self, "surge_bindings"):
            self.surge_bindings = {}
        if not hasattr(self, "area_locks"):
            self.area_locks = {}

        # 3) Utility to actually send an assignment to a robot
        def _dispatch(rid, cid, area_ids):
            try:
                # For non-surge dispatches, avoid sending areas that are currently locked by surge robots
                if not surge and self.area_locks:
                    filtered = [int(a) for a in area_ids if self.area_locks.get(int(a)) is None]
                    if not filtered:
                        self.debug(f"[DISPATCH-SKIP] rid={rid}: all requested areas locked by surge robots.")
                        return False
                    area_ids = filtered

                rospy.wait_for_service(f"/cluster_assignment_server_{rid}")
                srv = rospy.ServiceProxy(f"/cluster_assignment_server_{rid}", clusterAssignment2)

                tlapses = self.retrieve_tlapses(area_ids)
                rates = self.retrieve_decay_rates(area_ids)

                # Flatten this robot's distance matrix
                M = self.dist_matrices.get(rid, None)
                if M is None:
                    rospy.logwarn(f"assign_clusters: no distance matrix for robot {rid}")
                    return False
                dist_matrix = M.astype(np.float32).ravel().tolist()

                resp = srv(area_ids, tlapses, rates, dist_matrix)
                if not resp.availability:
                    return False

                # If surge: remember where the robot came from and bind
                named_cid = cid if (tag_prefix == "" or str(cid).startswith(tag_prefix)) else f"{tag_prefix}{cid}"
                home_cid = self.robots_assignment.get(rid, None) if surge else None

                # Bookkeeping
                self.clusters[named_cid] = list(map(int, area_ids))
                self.robots_assignment[rid] = named_cid
                self.clusters_assignment[named_cid] = rid
                if rid in self.unassigned_robots:
                    self.unassigned_robots.remove(rid)

                if surge:
                    # Record surge binding (time-based dwell cap)
                    cap_val = (len(area_ids) if (dwell_cap in (None, 'default')) else int(dwell_cap))
                    self.surge_bindings[rid] = {
                        'home_cid': home_cid,
                        'areas': set(int(a) for a in area_ids),
                        't_bind': int(getattr(self, "sim_t", rospy.get_time())),
                        'dwell_cap': max(1, int(cap_val)),
                    }
                    # Lock these areas for this surge robot
                    for aid in area_ids:
                        self.area_locks[int(aid)] = int(rid)

                    self.debug(f"[SURGE-BIND] Robot {rid} → {named_cid} areas={area_ids} dwell_cap={cap_val} (home={home_cid})")
                else:
                    self.debug(f"[ASSIGN] Robot {rid} → {named_cid} areas={area_ids}")

                return True
            except rospy.ServiceException as e:
                rospy.logerr(f"assign_clusters: service call failed for robot {rid}: {e}")
                return False
            except Exception as e:
                rospy.logerr(f"assign_clusters: dispatch error for robot {rid}: {e}")
                return False

        # 4) Build assignment either via Hungarian or greedy
        if method.lower() == "hungarian":
            try:
                rids = candidate_rids.copy()
                cids = valid_cids.copy()

                # Cost(i,j) = duration from robot i current node -> anchor of cluster j
                C = np.full((len(rids), len(cids)), np.inf, dtype=float)
                for i, rid in enumerate(rids):
                    M = self.dist_matrices.get(rid, None)
                    src = self.robots_location.get(rid, None)
                    if M is None or src is None:
                        continue
                    src = int(src)
                    for j, cid in enumerate(cids):
                        aid_anchor = int(anchors[cid])
                        if 0 <= src < M.shape[0] and 0 <= aid_anchor < M.shape[1]:
                            C[i, j] = float(M[src, aid_anchor])

                rr, cc = linear_sum_assignment(C)
                used = set()
                for i, j in zip(rr, cc):
                    if not np.isfinite(C[i, j]):
                        continue
                    rid = rids[i]
                    cid = cids[j]
                    if rid in used:
                        continue
                    if _dispatch(rid, cid, clusters[cid]):
                        used.add(rid)

                # Update unassigned list
                self.unassigned_robots = [rid for rid in self.robot_ids if rid not in used]

            except Exception as e:
                rospy.logwarn(f"Hungarian assignment failed ({e}); falling back to greedy.")
                method = "greedy"  # fall through

        if method.lower() == "greedy":
            # sort clusters by urgency (min current F at anchor)
            def cluster_urgency(cid):
                aid = anchors[cid]
                return self._area_current_F(aid) if aid is not None else float('inf')

            cids_sorted = sorted(valid_cids, key=cluster_urgency)
            free = candidate_rids.copy()

            for cid in cids_sorted:
                if not free:
                    break
                anchor = anchors[cid]
                rid, _ = self._closest_robot_to_area(free, anchor)
                if rid is None:
                    continue
                if _dispatch(rid, cid, clusters[cid]):
                    free.remove(rid)

            # any robot not used remains "unassigned"
            used = set(candidate_rids) - set(free)
            self.unassigned_robots = [rid for rid in self.robot_ids if rid not in used]


    def _dispatch_cluster_to_robot(self, rid, cid, area_ids, surge=False, dwell_cap='default', tag_prefix=""):
        """
        Directly dispatch cluster (cid -> area_ids) to robot rid via the ROS service.
        Updates all planner bookkeeping. If surge=True, records a surge binding (with dwell_cap).
        """

        try:
            rospy.wait_for_service(f"/cluster_assignment_server_{rid}")
            srv = rospy.ServiceProxy(f"/cluster_assignment_server_{rid}", clusterAssignment2)

            tlapses = self.retrieve_tlapses(area_ids)
            rates = self.retrieve_decay_rates(area_ids)

            M = self.dist_matrices.get(rid, None)
            if M is None:
                raise RuntimeError(f"No distance matrix for robot {rid}")
            dist_matrix = M.astype(np.float32).ravel().tolist()

            resp = srv(area_ids, tlapses, rates, dist_matrix)
            if not getattr(resp, "availability", True):
                return False

            # Bookkeeping
            named_cid = cid if (not tag_prefix or str(cid).startswith(tag_prefix)) else f"{tag_prefix}{cid}"
            home_cid = self.robots_assignment.get(rid, None) if surge else None

            self.clusters[named_cid] = list(map(int, area_ids))
            self.robots_assignment[rid] = named_cid
            self.clusters_assignment[named_cid] = rid
            if hasattr(self, "unassigned_robots") and rid in self.unassigned_robots:
                self.unassigned_robots.remove(rid)

            if surge:
                if not hasattr(self, "surge_bindings"):
                    self.surge_bindings = {}
                cap_val = (len(area_ids) if (dwell_cap in (None, 'default')) else int(dwell_cap))
                self.surge_bindings[rid] = {
                    'home_cid': home_cid,
                    'areas': set(int(a) for a in area_ids),
                    't_bind': int(getattr(self, "sim_t", rospy.get_time())),
                    'dwell_cap': max(1, int(cap_val)),
                }
                # Lock surge areas for this robot
                if not hasattr(self, "area_locks"):
                    self.area_locks = {}
                for aid in area_ids:
                    self.area_locks[int(aid)] = int(rid)
                self.debug(f"[SURGE-BIND] Robot {rid} → {named_cid} areas={area_ids} (home={home_cid})")
            else:
                self.debug(f"[ASSIGN] Robot {rid} → {named_cid} areas={area_ids}")

            return True

        except Exception as e:
            rospy.logerr(f"_dispatch_cluster_to_robot failed for robot {rid}: {e}")
            return False

    def release_completed_surge_bindings(self, *args, **kwargs):
        """
        (Deprecated) Kept for compatibility if referenced; not used.
        Use release_surge_robots(rho, dwell_cap_override) instead.
        """
        self.debug("[WARN] release_completed_surge_bindings is deprecated. Use release_surge_robots().")

    def note_surge_visit(self, rid: int, aid: int):
        """Record a visit by a surge robot to an area in its surge binding (optional for analytics)."""
        try:
            b = getattr(self, 'surge_bindings', {}).get(rid)
            if not b:
                return
            if aid in b.get('areas', set()):
                self.debug("Robot: {} accomplished surge Area {} visit".format(rid, aid))
        except Exception as e:
            rospy.logwarn(f"note_surge_visit failed for rid={rid}, aid={aid}: {e}")

    def _tau_bar_from_robot_mats(self):
        mats = [m for m in self.dist_matrices.values() if m is not None]
        if not mats:
            return {}
        M = np.stack(mats, axis=0)  # (R, N, N)
        M_mean = np.mean(M, axis=0)  # (N, N)
        J = M_mean.shape[0] - 1  # 0=charging
        tau_bar = {}
        for aid in range(1, J + 1):
            tau_bar[aid] = _mean_duration_excluding_col(M_mean, aid)
        return tau_bar

    def on_area_restored(self, area_id):
        self.tlapses[area_id] = 0
        self.last_reset_step[area_id] = self.sim_t

    def _binding_completed(self, binding):
        t_bind = int(binding['t_bind'])
        return all(int(self.last_reset_step.get(aid, -1)) > t_bind for aid in binding['areas'])

    def _init_jurisdictions(self, assign_method='hungarian', alpha_decay=1.0):
        if not self.clusters:
            # Optionally pause sim during planning if you prefer
            self.request_pause(True)
            self.clusters = self.create_clusters(alpha_decay=alpha_decay)
            self.request_pause(False)

            # Assign jurisdictions to all robots (uses greedy or Hungarian)
            self.unassigned_robots = self.robot_ids.copy()
            self.assign_clusters(self.clusters, method=assign_method, tag_prefix="C", surge=False)

    def _crisis_cycle_legacy(self, *args, **kwargs):
        """Kept only for reference; do not use."""
        pass

    def _crisis_cycle(self, k=1, eta=1.0, b=3, delta_t=1.0, alpha_decay=1.0, surge_assign_method='hungarian',
                      dwell_cap=None):
        """
        One crisis-mitigation pass (Final Algorithm):
          - Predict highly urgent areas ρ
          - Mark clusters needing external help: InternalOverload && ExternalFeasibility
          - Form surge team (base + potential)
          - Create surge clusters over Γ = union of predicted urgent among marked clusters
          - Assign via Hungarian and bind (time-based dwell cap)
          - Else (if already mitigating): release surge robots under OR condition
        """
        # Build tau_bar once
        if self.tau_bar is None:
            self.tau_bar = self._tau_bar_from_robot_mats()

        # Always predict current ρ
        rho = self.predict_highly_urgent_areas(k=k)

        if len(self.surge_bindings) == 0:
            self.debug("Detecting potential crisis...")

            # Detect marked clusters
            marked = []
            for cid, area_ids in (self.clusters or {}).items():
                if not cid or cid.startswith("SURG_"):
                    continue  # consider only home jurisdictions
                if self.internal_overload(cid, rho) and self.external_feasibility(cid, rho):
                    marked.append(cid)

            if not marked:
                self.debug("[CRISIS] No marked clusters this tick.")
                return

            # Form surge team
            S, Gamma = self.form_surge_robots(marked, rho)
            if not S:
                self.debug("[SURGE] No feasible surge team.")
                return

            # Create surge subclusters from Γ and assign to |S|
            urg_list = list(Gamma)
            K = min(max(1, len(S)), len(urg_list))
            surg_clusters = self.create_urgent_clusters(urg_list, K, alpha_decay=alpha_decay, seed=42)

            if not surg_clusters:
                self.debug("[SURGE] Failed to create surge clusters.")
                return

            # Assign surge clusters to surge robots via Hungarian
            self.assign_clusters(clusters=surg_clusters,
                                 method='hungarian',
                                 candidate_rids=S,
                                 tag_prefix="SURG_",
                                 surge=True,
                                 dwell_cap=(len(urg_list) if (dwell_cap in (None, 'default')) else int(dwell_cap)))

        else:
            # Try to release surge robots (OR condition)
            self.debug("Still mitigating crisis. Surge bindings: {}".format(self.surge_bindings))
            self.release_surge_robots(rho=rho, dwell_cap_override=dwell_cap)

    def wait_nodes_to_register(self):
        """
        Wait for area nodes to register
        :return:
        """
        na_counts = True
        while na_counts is True:
            decay_rates, dist_matrices = list(self.decay_rates.values()), list(self.dist_matrices.values())
            self.debug("Decay rates: {}. Dist matrices: {}".format(decay_rates, dist_matrices))
            is_notna_decay_rates = all(element is not None for element in decay_rates)
            is_notna_dist_matrices = all(element is not None for element in dist_matrices)
            if is_notna_decay_rates is True and is_notna_dist_matrices is True:
                na_counts = False
            rospy.sleep(1)
        self.debug("Sufficent data. Decay rates: {}. Dist matrices: {}".format(self.decay_rates, self.dist_matrices))

    def run_operation(self, filename, freq=1):
        rospy.sleep(10)
        self.wait_nodes_to_register()
        self.unassigned_robots = self.robot_ids.copy()
        self.status = centralStatus.IDLE.value
        self.sim_t = 0

        while not rospy.is_shutdown():
            self.central_status_pub.publish(self.status)
            self.print_state()

            if self.status == centralStatus.IDLE.value:
                # Build jurisdictions once, then switch to mission
                self.debug("Idle central state. Creating and assigning clusters")
                #TODO: Record process time
                self._init_jurisdictions(assign_method='hungarian',
                                         alpha_decay=1.0)  # Simulation is paused while thinking to create jurisdictions
                # self.assign_clusters(self.clusters)

                if len(self.unassigned_robots) == 0:
                    self.update_central_status(centralStatus.IN_MISSION)

            elif self.status == centralStatus.IN_MISSION.value:
                self.debug("Central in mission...")

                # Advance all tlapse by 1
                self.update_tlapses_areas(dt=1)
                self.sim_t += 1

                # Crisis detection and mitigation
                #TODO: Record process time
                if self.crisis_mitigation:
                    self._crisis_cycle(k=1, eta=1.0, b=3, delta_t=1.0,
                                       alpha_decay=1.0, surge_assign_method='hungarian', dwell_cap=None)

            elif self.status == centralStatus.CONSIDER_REPLAN.value:
                self.debug("Central considers re-assignment...")

            self.shutdown_check()

            #TODO: Save process time: init_jurisdiction, crisis_mitigation

            rospy.sleep(1)

    def shutdown_check(self):
        """
        Checks whether all areas have collected enough data points. If True, we shutdown all nodes
        :return:
        """
        self.debug("Shutdown check: {}".format(self.collected_enough_data))
        if False not in list(self.collected_enough_data.values()):
            self.shutdown(sleep=10)

    def collected_enough_data_cb(self, msg):
        """
        Updates area that has collected enough data
        :return:
        """
        area = msg.area_id
        collected_enough = msg.collected_enough
        self.collected_enough_data[area] = collected_enough
        self.debug("Received notice Area {}. Collected enough data".format(area))
        return collectedEnoughDataResponse(True)

    def check_pause(self):
        """
        Checks the pause request to Stage
        :return:
        """
        # Check time whether it is ticking
        # Then check time again after pausing the simulation and whether it is likewise ticking
        time_new = rospy.get_time()
        for i in range(5):
            time_prev = time_new
            time_new = rospy.get_time()
            tlapse = time_new - time_prev
            self.debug("Ticking instance: {}. Rospy time: {}. Tlapse: {}".format(i, time_new, tlapse))

            rospy.sleep(1)

        self.request_pause()

        time_prev = time_new
        time_new = rospy.get_time()
        tlapse = time_new - time_prev
        self.debug("Ticking instance: {}. Rospy time: {}. Tlapse: {}".format(i, time_new, tlapse))
        sum = 1 + 1
        self.debug("Sum: {}".format(sum))

        self.request_pause(False)

        self.check_pause()

    def request_pause(self, is_pause=True):
        """
        Sends pause request to pause_simulation
        :return:
        """
        self.requested_pause = is_pause
        rospy.wait_for_service('/pause_queue_server')
        try:
            pause_request = rospy.ServiceProxy('/pause_queue_server', pauseSimulation)
            agent_id = 999
            resp = pause_request(is_pause, agent_id)
            return resp.pause_result
        except rospy.ServiceException as e:
            rospy.logerr(f"Pause service call failed: {e}")

    def print_state(self):
        """
        Prints current state of the environment based on info collected by central
        :return:
        """
        state = (self.sim_t, self.robots_location, self.robots_battery, self.tlapses, self.decay_rates)
        self.debug("State: {}".format(state))

    def update_central_status(self, status):
        """
        Updates task scheduler status
        :param status:
        :return:
        """
        self.status = status.value

    def debug(self, msg):
        pu.log_msg(type='task_scheduler', id=None, msg=msg, debug=self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Collected enough data. Shutting down...".format(self.t_operation))
        kill_nodes(sleep)


if __name__ == '__main__':
    # os.chdir('/home/ameldocena/.ros/int_preservation/results')
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    CentralPlanner('central_planner').run_operation(filename)
    # CentralPlanner('central_planner').check_pause()