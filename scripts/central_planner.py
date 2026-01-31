#!/usr/bin/env python

import os
import math
import pickle
from math import ceil

import rospy
import numpy as np
import actionlib

# Project modules (assumed available in your workspace)
from pruning import *
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from std_msgs.msg import Int8, Float32
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from int_preservation.srv import clusterAssignment2
from int_preservation.srv import assignmentAccomplishment, assignmentAccomplishmentResponse
from int_preservation.srv import registerRobot, registerRobotResponse
from int_preservation.srv import pauseSimulation
from int_preservation.srv import collectedEnoughData, collectedEnoughDataResponse
from status import centralStatus, battStatus, robotStatus, robotAssignStatus
from reset_simulation import *
from heuristic_fcns import *
from loss_fcns import *
import project_utils as pu

# SciPy clustering & assignment
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SUCCEEDED = 3  # GoalStatus ID for succeeded
SHUTDOWN_CODE = 99


# -----------------------------
# Utilities for F-decay & taū
# -----------------------------
# def _exp_decay(F_max: float, rate: float, t: float) -> float:
#     """Exponential decay from F_max with absolute tlapse t: F(t) = F_max * exp(-rate * t)."""
#     return float(F_max) * math.exp(-float(rate) * max(0.0, float(t)))


# def _mean_duration_excluding_col(M: np.ndarray, j: int) -> float:
#     """
#     taū_j := average duration entries when the 'committed' choice is NOT j.
#     (Delete column j per our definition and average the remainder.)
#     """
#     sub = np.delete(M, j, axis=1)
#     return float(np.mean(sub)) if sub.size > 0 else 0.0


# ============================
# Central Planner
# ============================
class CentralPlanner:
    def __init__(self, node_name: str):
        """ROS central planner with refined crisis mitigation (no linger)."""
        rospy.init_node(node_name, anonymous=True)

        # --- Parameters ---
        self.nrobots = int(rospy.get_param("/nrobots"))
        self.robot_ids = [i for i in range(self.nrobots)]
        self.debug_mode = bool(rospy.get_param("/debug_mode"))
        self.robot_velocity = float(rospy.get_param("/robot_velocity"))  # linear velocity (magnitude)
        if rospy.get_param("/gamma") != 'None':
            self.gamma = float(rospy.get_param("/gamma")) # discount factor
        self.max_fmeasure = float(rospy.get_param("/max_fmeasure"))  # Max F-measure of an area
        self.max_battery = float(rospy.get_param("/max_battery"))  # Max battery
        self.battery_reserve = float(rospy.get_param("/battery_reserve"))  # Battery reserve
        self.tolerance = float(rospy.get_param("/move_base_tolerance"))
        self.t_operation = int(rospy.get_param("/t_operation"))
        self.charging_station = 0  # index 0

        f_thresh = rospy.get_param("/f_thresh")
        self.fsafe, self.fcrit = float(f_thresh[0]), float(f_thresh[1])  # (safe, crit)

        batt_consumed_per_time = rospy.get_param("/batt_consumed_per_time")
        self.batt_consumed_per_travel_time = float(batt_consumed_per_time[0])
        self.batt_consumed_per_restored_f = float(batt_consumed_per_time[1])

        self.dec_steps = int(rospy.get_param("/dec_steps"))  # STAR
        self.restoration = float(rospy.get_param("/restoration"))
        self.noise = float(rospy.get_param("/noise"))
        self.nareas = int(rospy.get_param("/nareas"))
        self.areas = [int(i + 1) for i in range(self.nareas)]  # 1..N

        # Crisis mitigation state
        # surge_bindings: rid -> {'areas': set[int], 't_bind': int, 'home_cid': str|None, 'dwell_cap': int}
        self.surge_bindings = {}
        self.area_locks = {}  # area_id -> rid (if locked by a surge robot)
        self.last_reset_step = {aid: -1 for aid in self.areas}  # last time an area was restored (tlapse reset)

        self.requested_pause = False

        self.debug(f"Nareas {self.nareas}. Areas list: {self.areas}")
        self.t_operation = int(rospy.get_param("/t_operation"))
        self.save = bool(rospy.get_param("/save"))
        self.crisis_mitigation = bool(rospy.get_param("/crisis_mitigation"))
        self.debug(f"Crisis mitigation: {self.crisis_mitigation}")

        # --- Load sampled area poses ---
        self.areas_poses = []
        with open('{}.pkl'.format(rospy.get_param("/file_sampled_areas")), 'rb') as f:
            sampled_areas_coords = pickle.load(f)
        for area_coords in sampled_areas_coords['n{}_p{}'.format(self.nareas, rospy.get_param("/placement"))]:
            pose_stamped = pu.convert_coords_to_PoseStamped(area_coords)
            self.areas_poses.append(pose_stamped)

        # --- Per-robot distance matrices ---
        self.dist_matrices = {robot: None for robot in range(self.nrobots)} #This is a global dist matrix for each robot's charging station to the rest of the areas

        self.tau_bar = None  # area_id -> taū_j
        self.charging_station = 0

        # --- Runtime state ---
        self.mission_areas = {robot_id: None for robot_id in self.robot_ids}
        self.assign_statuses = {robot_id: None for robot_id in self.robot_ids}
        self.tlapses = {area: 0 for area in self.areas}
        self.robot_statuses = {robot_id: None for robot_id in self.robot_ids}
        self.robots_location = {robot_id: None for robot_id in self.robot_ids}
        self.robots_battery = {robot_id: None for robot_id in self.robot_ids}
        self.decay_rates = {area: None for area in self.areas}
        self.collected_enough_data = {area: False for area in self.areas}

        self.clusters = None  # dict: cid -> [area_ids]
        self.clusters_assignment = {}  # cid -> rid
        self.robots_assignment = {}    # rid -> cid
        self.unassigned_clusters = []
        self.unassigned_robots = []

        # --- Servers ---
        self.robots_registry_server = rospy.Service('/robots_registry_server', registerRobot, self.register_robots_cb)
        self.assignment_accomplishment_server = rospy.Service('/assignment_accomplishment_server',
                                                              assignmentAccomplishment,
                                                              self.assignment_accomplishment_cb)
        self.shutdown_server = rospy.Service('/shutdown_server', collectedEnoughData, self.collected_enough_data_cb)

        # --- Publishers/Subscribers ---
        self.central_status_pub = rospy.Publisher('/central_status', Int8, queue_size=1)

        # move_base make_plan
        server = '/robot_0/move_base_node/make_plan'
        rospy.wait_for_service(server)
        self.get_plan_service = rospy.ServiceProxy(server, GetPlan)
        self.debug(f"Getplan service: {self.get_plan_service}")

        for robot_id in self.robot_ids:
            rospy.Subscriber(f'/robot_{robot_id}/mission_area', Int8, self.mission_area_cb, robot_id)
            rospy.Subscriber(f'/robot_{robot_id}/robot_status', Int8, self.robot_status_cb, robot_id)
            rospy.Subscriber(f'/robot_{robot_id}/location', Int8, self.robot_location_cb, robot_id)
            rospy.Subscriber(f'/robot_{robot_id}/battery', Float32, self.robot_battery_cb, robot_id)

        # Decay rates (oracle)
        for area in self.areas:
            rospy.Subscriber(f'/area_{area}/decay_rate', Float32, self.decay_rate_cb, area)

    # -----------------------------
    # Registration & Callbacks
    # -----------------------------
    def register_robots_cb(self, msg):
        """Register robot id and build its distance matrix."""
        robot_id = int(msg.robot_id)
        init_x = float(msg.init_x)
        init_y = float(msg.init_y)

        self.debug(f"Registry request received (id, x, y): {robot_id}, {init_x}, {init_y}")
        self.dist_matrices[robot_id] = self.build_dist_matrix(robot_id, init_x, init_y)
        self.robots_location[robot_id] = 0  # initial

        dm = self.dist_matrices[robot_id]
        self.debug(f"Robot registered: {robot_id}. Dist matrix shape: {None if dm is None else dm.shape}")

        return registerRobotResponse(True)

    def robot_location_cb(self, msg, robot_id):
        self.robots_location[int(robot_id)] = int(msg.data)

    def robot_battery_cb(self, msg, robot_id):
        self.robots_battery[int(robot_id)] = float(msg.data)

    def assignment_accomplishment_cb(self, msg):
        """Robot reports area restored: reset tlapse and note for surge analytics."""
        robot_id = int(msg.robot_id)
        area_id = int(msg.area_accomplished)
        self.tlapses[area_id] = 0
        self.on_area_restored(area_id)
        # self.note_surge_visit(rid=robot_id, aid=area_id)
        self.debug(f"Received notice Robot {robot_id} restored Area {area_id}. Tlapse reset: {self.tlapses[area_id]}")
        return assignmentAccomplishmentResponse(True)

    def assign_status_cb(self, msg, robot_id):
        self.assign_statuses[int(robot_id)] = int(msg.data)

    def mission_area_cb(self, msg, robot_id):
        self.mission_areas[int(robot_id)] = int(msg.data)

    def robot_status_cb(self, msg, robot_id):
        self.robot_statuses[int(robot_id)] = int(msg.data)

    def decay_rate_cb(self, msg, area_id):
        if self.decay_rates[int(area_id)] is None:
            self.decay_rates[int(area_id)] = float(msg.data)
            self.debug(f"Area {area_id} decay rate: {msg.data}")

    # -----------------------------
    # Planner: geometry helpers
    # -----------------------------
    def get_plan_request(self, start_pose, goal_pose, tolerance):
        """Use move_base make_plan service to get a path without moving the robot."""
        req = GetPlan()
        req.start = start_pose
        req.goal = goal_pose
        req.tolerance = tolerance
        result = self.get_plan_service(req.start, req.goal, req.tolerance)
        return result.plan.poses

    def decouple_path_poses(self, path):
        """Decouple a path of PoseStamped poses into list[(x, y)]."""
        out = []
        for p in path:
            out.append((p.pose.position.x, p.pose.position.y))
        return out

    def compute_path_total_dist(self, list_poses):
        """Compute total path length over a sequence of (x,y) points."""
        total = 0.0
        for i in range(len(list_poses) - 1):
            total += math.dist(list_poses[i], list_poses[i + 1])
        return total

    def compute_dist_bet_areas(self, area_i, area_j, tolerance):
        """Distance between area_i and area_j via planner."""
        path = self.get_plan_request(area_i, area_j, tolerance)
        pts = self.decouple_path_poses(path)
        return self.compute_path_total_dist(pts)

    def build_dist_matrix(self, robot_id, init_x, init_y):
        """Build the distance matrix among areas (includes charging station as index 0)."""
        charging_station_coords = (init_x, init_y)
        charging_pose_stamped = pu.convert_coords_to_PoseStamped(charging_station_coords)
        nodes_poses = [charging_pose_stamped]
        nodes_poses.extend(self.areas_poses)
        self.debug(f"Robot: {robot_id}. Nodes_poses count: {len(nodes_poses)}")

        n = len(nodes_poses)
        dist_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist_matrix[i, j] = self.compute_dist_bet_areas(nodes_poses[i], nodes_poses[j], self.tolerance)

        return dist_matrix

    def retrieve_tlapses(self, areas):
        return [int(self.tlapses[a]) for a in areas]

    def retrieve_decay_rates(self, areas):
        return [float(self.decay_rates[a]) for a in areas]

    def update_tlapses_areas(self, dt=1):
        for aid in self.areas:
            self.tlapses[aid] += int(dt)

    # -----------------------------
    # Clustering (jurisdictions)
    # -----------------------------
    def create_clusters(self, alpha_decay=1.0, seed=42, n_clusters=None):
        """
        Build jurisdictions using SciPy kmeans2 on features [x, y, alpha*decay].
        Returns: dict {cluster_name: [area_ids]}
        """
        n_clusters = n_clusters or self.nrobots

        feats, ids = [], []
        for aid in self.areas:
            rate = float(self.decay_rates.get(aid, 0.0))
            p = self.areas_poses[aid - 1].pose.position  # 1-based ids
            feats.append([p.x, p.y, alpha_decay * rate])
            ids.append(aid)
        X = np.asarray(feats, dtype=float)
        _, labels = kmeans2(X, k=n_clusters, minit='++', iter=50, thresh=1e-05)

        clusters = {}
        for c in range(n_clusters):
            clusters[f"C{c + 1}"] = [ids[i] for i, lab in enumerate(labels) if int(lab) == c]

        self.clusters = clusters
        return clusters

    def create_urgent_clusters(self, rho, K, alpha_decay=1.0, seed=42):
        """
        Cluster the urgent set rho into K groups using kmeans2 on [x, y, alpha*decay].
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

    # -----------------------------
    # Present value of discounted losses
    # -----------------------------
    def _npv_cluster_loss(self, area_ids, gamma, tau_bar, decay_rates):
        """
        NPV_c = Σ_j Σ_{h=1..H_j} gamma^h * max(0, z - g(δ_j, h*delta_t)),
        with H_j = |c|
        """
        if not area_ids:
            return 0.0
        h = max(1, len(area_ids))
        tot = 0.0
        self.debug("Calculating NPV cluster: {}".format(area_ids))
        for aid in area_ids:
            rate = decay_rates[aid]
            curr_tlapse = self.tlapses.get(aid, 0)
            est_tlapse = curr_tlapse + h * float(tau_bar.get(aid, 0.0))
            F_pred = decay(rate, est_tlapse, self.max_fmeasure)
            loss = loss_fcn(max_fmeasure=self.max_fmeasure, decayed_fmeasure=F_pred)
            tot += 1 / (1 + gamma ** h) * loss
            self.debug("Area: {}. Loss: {}. Running total: {}".format(aid, loss, tot))
        return tot

    # -----------------------------
    # Urgency & overload tests
    # -----------------------------
    def _area_current_F(self, aid):
        rate = float(self.decay_rates[aid])
        t = float(self.tlapses[aid])
        return decay(rate, t, self.max_fmeasure)

    def predict_highly_urgent_areas(self, k=1):
        """
        Predict areas that will be ≤ fcrit within k * tau_bar[a].
        Uses absolute tlapse: tlapse_now + horizon, decaying from F_max.
        Returns a set of area ids.
        """
        if self.tau_bar is None:
            self.tau_bar = self._tau_bar_from_robot_mats()

        rho = set()
        for aid in self.areas:
            tau = float(self.tau_bar.get(aid, 0.0))
            horizon = max(0.0, k * tau)
            rate = float(self.decay_rates[aid])
            tlapse_now = float(self.tlapses[aid])
            F_future = decay(rate, tlapse_now + horizon, self.max_fmeasure) #_exp_decay(self.max_fmeasure, rate, tlapse_now + horizon)
            if F_future <= self.fcrit:
                rho.add(aid)
        self.debug(f"[URGENT-PRED] k={k} -> |rho|={len(rho)}")
        return rho

    def internal_overload(self, cid, rho):
        """
        Overload if > 50% of areas are predicted highly urgent AND there exists at least one actually urgent area.
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
        self.debug(f"[INTERNAL-OVERLOAD] cid={cid} majority={majority_urgent} actual={actual_urgent} "
                   f"| urgent={urgent_count}/{cluster_size} -> {overloaded}")
        return overloaded

    def external_feasibility(self, cid, rho):
        """
        True if at least one external robot can arrive before the home robot would
        finish visiting all predicted highly urgent areas (h = |U(C)|).
        τ̄(C) is taken as the average of the assigned home robot's duration matrix.
        """
        areas = self.clusters.get(cid, [])
        if not areas:
            return False

        h = len([a for a in areas if a in rho])
        if h == 0:
            return False

        home_rid = self.clusters_assignment.get(cid, None)
        if home_rid is None:
            return False

        M_home = self.dist_matrices.get(home_rid, None)
        if M_home is None or M_home.size == 0:
            return False
        tau_bar = float(np.mean(M_home))

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
                req_batt = travel_time * self.batt_consumed_per_travel_time
                have_batt = float(self.robots_battery.get(rid, 0.0))
                if have_batt >= req_batt:
                    feasible_arrivals.append(travel_time)

        if not feasible_arrivals:
            self.debug(f"[EXTERNAL-FEAS] cid={cid} no feasible external arrivals")
            return False

        t_ext = min(feasible_arrivals)
        visits_home = t_ext / max(1e-6, tau_bar)
        feasible = (visits_home < float(h))
        self.debug(f"[EXTERNAL-FEAS] cid={cid} h={h} t_ext={t_ext:.3f} taū={tau_bar:.3f} visits≈{visits_home:.2f} -> {feasible}")
        return feasible

    # -----------------------------
    # Surge team formation
    # -----------------------------
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

        if len(Gamma) <= len(base):
            self.debug(f"[SURGE-FORM] |Γ|={len(Gamma)} <= |base|={len(base)} -> base only")
            return base, Gamma

        # Compute NPV(Γ)
        npv_gamma = self._npv_cluster_loss(list(Gamma), self.gamma,
                                           self.tau_bar or self._tau_bar_from_robot_mats(),
                                           self.decay_rates)

        # Available non-base robots
        avail = [rid for rid in self.robot_ids if rid not in base]
        A = max(1, len(avail))
        avg_surge = npv_gamma / float(A)

        def home_npv(rid):
            home_cid = self.robots_assignment.get(rid, None)
            areas = self.clusters.get(home_cid, []) if home_cid else []
            return self._npv_cluster_loss(areas, self.gamma,
                                          self.tau_bar or self._tau_bar_from_robot_mats(),
                                          self.decay_rates)

        # Expected surge energy budget (rough): expected cluster size × avg tau of Γ × travel cost factor
        taus = [float(self.tau_bar.get(a, 0.0)) for a in Gamma] if Gamma else [0.0]
        avg_tau = float(np.mean(taus)) if taus else 0.0
        expected_size = int(math.ceil(len(Gamma) / float(A)))
        expected_cost = expected_size * (avg_tau * self.batt_consumed_per_travel_time)

        potential = []
        for rid in avail:
            hn = home_npv(rid)
            have_batt = float(self.robots_battery.get(rid, 0.0))
            if (hn < avg_surge) and (have_batt >= expected_cost):
                potential.append((hn, rid))

        potential.sort(key=lambda x: x[0])  # ascending by NPV_home
        S = base + [rid for _, rid in potential]

        # Cap by |Γ| (≥ 1 area per surge robot)
        cap = len(Gamma)
        if len(S) > cap:
            S = S[:cap]

        self.debug(f"[SURGE-FORM] base={base} potential={[rid for _, rid in potential]} "
                   f"npvΓ={npv_gamma:.3f} avg_surge≈{avg_surge:.3f} expected_cost≈{expected_cost:.3f} -> S={S}")
        return S, Gamma

    # -----------------------------
    # Assignment helpers
    # -----------------------------
    def _most_urgent_anchor(self, area_ids):
        """Anchor as min predicted F at a 1*tau horizon (ranking consistency)."""
        if not area_ids:
            return None
        best = None
        bestF = float('inf')
        for a in area_ids:
            tau = float(self.tau_bar.get(a, 0.0)) if self.tau_bar else 0.0
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
            cost = float(M[int(src), int(area_id)]) #distance cost
            if cost < best_cost:
                best, best_cost = rid, cost
        return best, best_cost

    def assign_clusters(self,
                        clusters,
                        method='hungarian',
                        candidate_rids=None,
                        tag_prefix="C",
                        surge=False,
                        dwell_cap='default'):
        """
        Assign clusters to robots.

        - method: 'greedy' or 'hungarian'
        - candidate_rids: optional subset of robot ids to consider
        - tag_prefix: "C" or "SURG_"
        - surge: if True, record surge bindings (locks + dwell_cap based on timesteps)
        - dwell_cap: for surge bindings:
            * 'default' or None  -> len(area_ids)
            * int                -> explicit cap (timesteps for release_surge_robots)
        """
        self.debug(f"Assigning clusters (method={method}, surge={surge}): {clusters}")
        #TODO: Measure time here

        # 1) Anchors
        cluster_ids = list(clusters.keys())
        anchors = {cid: self._most_urgent_anchor(clusters[cid]) for cid in cluster_ids}
        valid_cids = [cid for cid in cluster_ids if anchors[cid] is not None]
        #TODO: Measure time up to here

        if not valid_cids:
            self.debug("No valid clusters to assign (no anchors).")
            return

        # 2) Robots
        if candidate_rids is None:
            candidate_rids = (self.unassigned_robots.copy() or self.robot_ids.copy())

        # Ensure dicts exist
        if surge and not hasattr(self, "surge_bindings"):
            self.surge_bindings = {}
        if not hasattr(self, "area_locks"):
            self.area_locks = {}

        # 3) Dispatch helper
        def _dispatch(rid, cid, area_ids):
            try:
                # For non-surge dispatches, avoid sending areas currently locked by surge robots
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

                M = self.dist_matrices.get(rid, None)
                if M is None:
                    rospy.logwarn(f"assign_clusters: no distance matrix for robot {rid}")
                    return False
                dist_matrix = M.astype(np.float32).ravel().tolist()

                resp = srv(area_ids, tlapses, rates, dist_matrix)
                if not resp.availability:
                    return False

                named_cid = cid if (tag_prefix == "" or str(cid).startswith(tag_prefix)) else f"{tag_prefix}{cid}"
                home_cid = self.robots_assignment.get(rid, None) if surge else None

                # Bookkeeping
                self.clusters[named_cid] = list(map(int, area_ids))
                self.robots_assignment[rid] = named_cid
                self.clusters_assignment[named_cid] = rid
                if rid in self.unassigned_robots:
                    self.unassigned_robots.remove(rid)

                if surge:
                    cap_val = (int(self.t_operation) if (dwell_cap in (None, 'default')) else int(dwell_cap))
                    self.surge_bindings[rid] = {
                        'home_cid': home_cid,
                        'areas': set(int(a) for a in area_ids),
                        't_bind': int(getattr(self, "sim_t", rospy.get_time())),
                        'dwell_cap': max(1, int(cap_val)),
                    }
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

        # 4) Assignment
        if method.lower() == "hungarian":
            #TODO: Measure time here
            try:
                rids = candidate_rids.copy()
                cids = valid_cids.copy()

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

                self.unassigned_robots = [rid for rid in self.robot_ids if rid not in used]
                #TODO: Measure time here, potentially return it

            except Exception as e:
                rospy.logwarn(f"Hungarian assignment failed ({e}); falling back to greedy.")
                method = "greedy"  # fall through

        if method.lower() == "greedy":
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
                cap_val = (len(self.t_operation) if (dwell_cap in (None, 'default')) else int(dwell_cap))
                self.surge_bindings[rid] = {
                    'home_cid': home_cid,
                    'areas': set(int(a) for a in area_ids),
                    't_bind': int(getattr(self, "sim_t", rospy.get_time())),
                    'dwell_cap': max(1, int(cap_val)),
                }
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

    # -----------------------------
    # Surge release (double dispatch)
    # -----------------------------
    def _rebuild_home_of_area(self):
        """
        Build/refresh a map from area_id -> home_cid using only base (non-SURG_) clusters.
        Call this after initial jurisdiction creation or whenever base clusters are changed.
        """
        self.home_of_area = {}
        for cid, area_ids in (self.clusters or {}).items():
            if isinstance(cid, str) and cid.startswith("SURG_"):
                continue
            for a in area_ids:
                self.home_of_area[int(a)] = cid
        self.debug(f"[HOME-MAP] Built area→home_cid for {len(self.home_of_area)} areas.")

    def release_surge_robots(self, k=2, dwell_cap=None):
        """
        Release surge robots under disjunction:
          (a) the SURG_ cluster is no longer a highly urgent cluster (per InternalOverload), OR
          (b) dwell cap (in timesteps) has elapsed since binding.

        On release:
          - Unlock areas previously locked by this surge robot.
          - Dispatch the surge robot back to its home jurisdiction with currently FREE areas.
          - For EVERY unlocked area, update its original home jurisdiction: if that home robot is NOT in surge,
            re-dispatch the home jurisdiction with all of its currently FREE areas.
        """
        if not getattr(self, "surge_bindings", None):
            return
        if not hasattr(self, "area_locks"):
            self.area_locks = {}
        if not hasattr(self, "home_of_area") or not self.home_of_area:
            self._rebuild_home_of_area()

        # Need rho to evaluate InternalOverload for SURG_ clusters
        if self.tau_bar is None:
            self.tau_bar = self._tau_bar_from_robot_mats()
        rho = self.predict_highly_urgent_areas(k=k)

        now = int(self.sim_t)
        to_release = []

        # Identify surge robots to release
        for rid, binding in list(self.surge_bindings.items()):
            surge_cid = self.robots_assignment.get(rid, None)
            if not (isinstance(surge_cid, str) and surge_cid.startswith("SURG_")):
                continue

            still_overloaded = self.internal_overload(surge_cid, rho)  # True => keep; False => can release
            # t_bind = int(binding.get("t_bind", now))
            # default_cap = max(1, len(binding.get("areas", [])) or 1)
            # cap = int(dwell_cap) if (dwell_cap is not None) else int(binding.get("dwell_cap", default_cap))
            # dwell_elapsed = (now >= (t_bind + cap))

            # if (not still_overloaded) or dwell_elapsed:
            if not still_overloaded:
                reason = "no_longer_urgent" #if (not still_overloaded) else "dwell_cap"
                to_release.append((rid, surge_cid, binding, reason))

        if not to_release:
            return

        for rid, surge_cid, binding, reason in to_release:
            home_cid = binding.get("home_cid", None)
            surge_areas = list(map(int, binding.get("areas", set())))

            self.debug(f"[SURGE-RELEASE] rid={rid} from {surge_cid} reason={reason} |areas|={len(surge_areas)}")

            # ---- A) Unlock this robot's surge areas ----
            unlocked = []
            for aid in surge_areas:
                if self.area_locks.get(aid) == rid:
                    self.area_locks.pop(aid, None)
                    unlocked.append(aid)

            # Clean surge cluster mapping and binding
            self.surge_bindings.pop(rid, None)
            self.clusters.pop(surge_cid, None)
            self.clusters_assignment.pop(surge_cid, None)

            # ---- B) Dispatch the releasing robot back to its own home (FREE areas only) ----
            if home_cid and home_cid in self.clusters:
                full_home = list(map(int, self.clusters[home_cid]))
                free_home = [a for a in full_home if self.area_locks.get(a) is None]
                if free_home:
                    ok = self._dispatch_cluster_to_robot(
                        rid=rid,
                        cid=home_cid,
                        area_ids=free_home,
                        surge=False,
                        dwell_cap='default',
                        tag_prefix="C",
                    )
                    if ok:
                        self.debug(f"[SURGE-RETURN] rid={rid} → {home_cid} with {len(free_home)} free areas.")
                    else:
                        self.debug(f"[SURGE-RETURN-FAIL] rid={rid} could not be dispatched to {home_cid}.")
                else:
                    self.robots_assignment[rid] = None
                    if rid not in self.unassigned_robots:
                        self.unassigned_robots.append(rid)
                    self.debug(f"[SURGE-RETURN] rid={rid} no free areas in {home_cid}; set unassigned.")
            else:
                self.robots_assignment[rid] = None
                if rid not in self.unassigned_robots:
                    self.unassigned_robots.append(rid)
                self.debug(f"[SURGE-RETURN] rid={rid} has no valid home_cid; set unassigned.")

            # ---- C) Update all affected home jurisdictions for newly unlocked areas ----
            affected_by_home = {}
            for aid in unlocked:
                hc = self.home_of_area.get(aid, None)
                if hc is not None:
                    affected_by_home.setdefault(hc, set()).add(aid)

            for hc, _unlocked_set in affected_by_home.items():
                if hc not in self.clusters:
                    continue  # cluster removed or invalid

                full = list(map(int, self.clusters[hc]))
                free = [a for a in full if self.area_locks.get(a) is None]
                if not free:
                    continue

                home_rid = self.clusters_assignment.get(hc, None)
                if home_rid is None:
                    continue

                curr_assign = self.robots_assignment.get(home_rid, "")
                if isinstance(curr_assign, str) and curr_assign.startswith("SURG_"):
                    self.debug(f"[SURGE-UPDATE-SKIP] home_rid={home_rid} for {hc} still in surge ({curr_assign}); defer update.")
                    continue

                if (home_rid == rid) and (hc == home_cid):
                    continue  # already dispatched above

                ok = self._dispatch_cluster_to_robot(
                    rid=home_rid,
                    cid=hc,
                    area_ids=free,
                    surge=False,
                    dwell_cap='default',
                    tag_prefix="C",
                )
                if ok:
                    self.debug(f"[SURGE-UPDATE] Updated home cluster {hc} for robot {home_rid} with {len(free)} free areas.")
                else:
                    self.debug(f"[SURGE-UPDATE-FAIL] Could not update {hc} for robot {home_rid}.")

        self.debug(f"[SURGE-RELEASE] Completed. Active surge robots: {list(self.surge_bindings.keys())}")

    def note_surge_visit(self, rid: int, aid: int):
        """Record a visit by a surge robot to an area in its surge binding."""
        try:
            b = getattr(self, 'surge_bindings', {}).get(rid)
            if not b:
                return
            if aid in b.get('areas', set()):
                self.debug("Robot: {} accomplished surge Area {} visit".format(rid, aid))
                b['visit_count'] = int(b.get('visit_count', 0)) + 1
                va = b.get('visited_areas', set())
                va.add(int(aid))
                b['visited_areas'] = va
        except Exception as e:
            rospy.logwarn(f"note_surge_visit failed for rid={rid}, aid={aid}: {e}")

    # -----------------------------
    # Crisis cycle (final algorithm)
    # -----------------------------
    def _tau_bar_from_robot_mats(self):
        mats = [m for m in self.dist_matrices.values() if m is not None]
        if not mats:
            return {}
        M = np.stack(mats, axis=0)  # (R, N, N)
        M_mean = np.mean(M, axis=0)  # (N, N)
        J = M_mean.shape[0] - 1  # 0=charging
        tau_bar = {}
        for aid in range(1, J + 1):
            tau_bar[aid] = pu._mean_duration_excluding_col(M_mean, aid)
        return tau_bar

    def on_area_restored(self, area_id):
        self.tlapses[area_id] = 0
        self.last_reset_step[area_id] = getattr(self, "sim_t", 0)

    def _crisis_cycle(self, k=2, alpha_decay=1.0, assign_method='hungarian',
                      dwell_cap=None):
        """
        One crisis-mitigation pass (Final Algorithm):
          - Predict highly urgent areas ρ
          - Mark clusters needing external help: InternalOverload && ExternalFeasibility
          - Form surge team (base + potential)
          - Create surge clusters over Γ = union of predicted urgent among marked clusters
          - Assign via Hungarian and bind (time-based dwell cap)
          - Else (already mitigating): release surge robots under OR condition
        """
        if self.mitigating is False:
            self.debug("Detecting potential crisis...")
            #TODO: Measure time here
            if self.tau_bar is None:
                self.tau_bar = self._tau_bar_from_robot_mats()

            rho = self.predict_highly_urgent_areas(k=k)

            # Detect marked clusters
            marked = []
            for cid, area_ids in (self.clusters or {}).items():
                if not cid or (isinstance(cid, str) and cid.startswith("SURG_")):
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

            self.mitigating = True
            self.debug("Currently mitigating crisis: {}".format(self.mitigating))

            # Assign surge clusters to surge robots via Hungarian
            self.assign_clusters(clusters=surg_clusters,
                                 method=assign_method,
                                 candidate_rids=S,
                                 tag_prefix="SURG_",
                                 surge=True,
                                 dwell_cap=(len(urg_list) if (dwell_cap in (None, 'default')) else int(dwell_cap)))
            self.debug("Currently mitigating crisis: {}".format(self.mitigating))
            #TODO: Measure time here
        else:
            self.debug(f"Still mitigating crisis. Surge bindings: {self.surge_bindings}")
            self.release_surge_robots(dwell_cap=dwell_cap, k=k)
            if len(self.surge_bindings) == 0: #All surge robots have been released
                self.debug("All surge robots released! Resetting self.mitigating to False.")
                self.mitigating = False

    # -----------------------------
    # Runtime & admin
    # -----------------------------
    def wait_nodes_to_register(self):
        """Wait for area nodes to register (dist matrices + decay rates)."""
        na_counts = True
        while na_counts:
            decay_rates = list(self.decay_rates.values())
            dist_matrices = list(self.dist_matrices.values())
            self.debug(f"Decay rates: {decay_rates}. Dist matrices: {dist_matrices}")
            is_notna_decay = all(x is not None for x in decay_rates)
            is_notna_dists = all(x is not None for x in dist_matrices)
            if is_notna_decay and is_notna_dists:
                na_counts = False
            rospy.sleep(1)
        self.debug(f"Sufficient data. Decay rates: {self.decay_rates.keys()} | Dist mats known: {sum(m is not None for m in self.dist_matrices.values())}/{self.nrobots}")

    def run_operation(self, filename, freq=1):
        rospy.sleep(10)
        self.wait_nodes_to_register()
        self.unassigned_robots = self.robot_ids.copy()
        self.status = centralStatus.IDLE.value
        self.sim_t = 0
        self.mitigating = False

        while not rospy.is_shutdown():
            self.central_status_pub.publish(self.status)
            self.print_state()

            if self.status == centralStatus.IDLE.value:
                # Build jurisdictions once, then switch to mission
                self.debug("Idle central state. Creating and assigning clusters")
                self._init_jurisdictions(assign_method='hungarian', alpha_decay=1.0)

                if len(self.unassigned_robots) == 0:
                    self.update_central_status(centralStatus.IN_MISSION)

            elif self.status == centralStatus.IN_MISSION.value:
                self.debug("Central in mission...")
                # Advance all tlapse by 1
                self.update_tlapses_areas(dt=1)
                self.sim_t += 1

                # Crisis detection and mitigation
                if self.crisis_mitigation:
                    self._crisis_cycle(k=3,
                                       alpha_decay=1.0, assign_method='hungarian', dwell_cap=None)

            elif self.status == centralStatus.CONSIDER_REPLAN.value:
                self.debug("Central considers re-assignment...")

            self.shutdown_check()
            rospy.sleep(1)

    def shutdown_check(self):
        """Check whether all areas have collected enough data to shutdown."""
        self.debug(f"Shutdown check: {self.collected_enough_data}")
        if False not in list(self.collected_enough_data.values()):
            self.shutdown(sleep=10)

    def collected_enough_data_cb(self, msg):
        """Update area that has collected enough data."""
        area = int(msg.area_id)
        collected_enough = bool(msg.collected_enough)
        self.collected_enough_data[area] = collected_enough
        self.debug(f"Received notice Area {area}. Collected enough data")
        return collectedEnoughDataResponse(True)

    def check_pause(self):
        """Diagnostic helper to verify pause mechanism (unused)."""
        time_new = rospy.get_time()
        for i in range(5):
            time_prev = time_new
            time_new = rospy.get_time()
            tlapse = time_new - time_prev
            self.debug(f"Ticking instance: {i}. Rospy time: {time_new}. Tlapse: {tlapse}")
            rospy.sleep(1)

        self.request_pause()
        time_prev = time_new
        time_new = rospy.get_time()
        tlapse = time_new - time_prev
        self.debug(f"Ticking instance: {i}. Rospy time: {time_new}. Tlapse: {tlapse}")
        self.request_pause(False)
        self.check_pause()

    def request_pause(self, is_pause=True):
        """Pause Stage simulation via service."""
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
        """Print current state for debugging."""
        state = (self.sim_t, self.robots_location, self.robots_battery, self.tlapses, self.decay_rates)
        self.debug(f"State: {state}")

    def update_central_status(self, status):
        self.status = status.value

    def debug(self, msg):
        pu.log_msg(type='task_scheduler', id=None, msg=msg, debug=self.debug_mode)

    def shutdown(self, sleep):
        self.debug("Collected enough data. Shutting down...")
        kill_nodes(sleep)

    # -----------------------------
    # Init jurisdictions
    # -----------------------------
    def _init_jurisdictions(self, assign_method='hungarian', alpha_decay=1.0):
        if not self.clusters:
            self.request_pause(True)
            self.clusters = self.create_clusters(alpha_decay=alpha_decay)
            self.request_pause(False)

            # Assign jurisdictions to all robots
            self.unassigned_robots = self.robot_ids.copy()
            self.assign_clusters(self.clusters, method=assign_method, tag_prefix="C", surge=False)

            # Build area -> home cluster map for release logic
            self._rebuild_home_of_area()


if __name__ == '__main__':
    os.chdir('/root/catkin_ws/src/results/int_preservation')
    filename = rospy.get_param('/file_data_dump')
    CentralPlanner('central_planner').run_operation(filename)