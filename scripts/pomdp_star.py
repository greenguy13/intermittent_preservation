#!/usr/bin/env python

"""
This implements STAR as a POMDP and uses POMCP to solve the POMDP
"""

import pomdp_py
import random
import numpy as np
import random

random.seed(1234) #For reproducibility

#### State
class STARState(pomdp_py.State):
    def __init__(self, F_values, location):
        self.F_values = F_values  # Dictionary {area_index: F_value}
        self.location = location  # Current robot location (integer index)

    def __hash__(self):
        """ Hash function for storing state in sets/dictionaries. """
        return hash((tuple(sorted(self.F_values.items())), self.location))

    def __eq__(self, other):
        """ Equality comparison between two states. """
        return isinstance(other, STARState) and self.F_values == other.F_values and self.location == other.location

    def __repr__(self):
        """ String representation of the state. """
        F_str = ", ".join(f"Area {i}: {F:.2f}" for i, F in self.F_values.items())
        return f"STARState(Location: {self.location}, F_values: [{F_str}])"

#### Action
class STARAction(pomdp_py.Action):
    def __init__(self, target_area):
        self.target_area = target_area  # Destination for travel and restoration
        self.name = f"Restore Area {target_area}"
    def __hash__(self):
        """ Hash function for storing action in sets/dictionaries. """
        return hash(self.target_area)

    def __eq__(self, other):
        """ Equality comparison between two actions. """
        return isinstance(other, STARAction) and self.target_area == other.target_area

    def __repr__(self):
        """ String representation of the action. """
        return f"STARAction(Travel_Restore to Area {self.target_area})"

#### Transition
class STARTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, duration_matrix, decay_rates, F_max):
        """
        Args:
        - duration_matrix (dict): Dictionary { (i, j): travel_time } for all areas.
        - decay_rates (dict): Dictionary { i: decay_rate } for all areas.
        - F_max (float): Maximum environmental measure F.
        """
        self.duration_matrix = duration_matrix
        self.decay_rates = decay_rates
        self.F_max = F_max

    def probability(self, next_state, state, action):
        """ Since transitions are deterministic, return probability 1.0 if next_state is correct. """
        expected_next_state = self.sample(state, action)
        return 1.0 if next_state == expected_next_state else 0.0 #deterministic

    def sample(self, state, action):
        """ Samples the next state given the current state and action. """
        target_area = action.target_area
        travel_time = self.duration_matrix[state.location-1, target_area-1]

        # Update F values (decay for all areas, restoration sets target to F_max)
        new_F_values = {}
        for i, F_i in state.F_values.items():
            if i == target_area:
                new_F_values[i] = self.F_max  # Reset to max upon restoration
            else:
                new_F_values[i] = max(0, F_i - self.decay_rates[i] * travel_time)  # Other areas decay by the duration based on belief decay rates

        # Update robot's location
        new_location = target_area

        return STARState(new_F_values, new_location)

#### Observation
class STARObservation(pomdp_py.Observation):
    def __init__(self, observed_F, area):
        """
        Represents an observation in the STAR POMDP.

        Args:
        - observed_F (float): The measured environmental measure F.
        """
        self.observed_F = observed_F  # The exact F value observed upon restoration.
        self.area = area

    def __hash__(self):
        """ Hash function for storing observations in sets/dictionaries. """
        # return hash(self.observed_F)
        return hash((self.observed_F, self.area))

    def __eq__(self, other):
        """ Equality comparison between two observations. """
        return isinstance(other, STARObservation) and self.observed_F == other.observed_F and self.area == other.area

    def __repr__(self):
        """ String representation of the observation. """
        return f"STARObservation(Area={self.area}, F_observed={self.observed_F:.2f})"


#### Observation Model
class STARObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise_std=5.0):
        """
        Stochastic Observation Model for STAR POMDP.

        Args:
        - noise_std (float): Standard deviation of observation noise.
        """
        self.noise_std = noise_std  # Controls observation uncertainty

    def probability(self, observation, next_state, action):
        """ Returns P(o | s', a). """
        if isinstance(action, STARAction) and observation.area == action.target_area and next_state.location == observation.area:
            restored_area = action.target_area
            true_F = next_state.F_values[restored_area]
            observed_F = observation.observed_F

            # Compute probability using Gaussian likelihood, avoid deterministic probability
            prob = max(0.05, np.exp(-0.5 * ((observed_F - true_F) / self.noise_std) ** 2) / (self.noise_std * np.sqrt(2 * np.pi)))
            return prob

        return 0.0 # If observation is from a different area, probability is zero, not possible

    def sample(self, next_state, action, add_noise=True):
        """ Samples an observation given the next state and action. """
        # print("Sampling observation. Next state: {}. Action: {}".format(next_state, action))
        restored_area = action.target_area
        true_F = next_state.F_values[restored_area]

        if add_noise:
            observed_F = np.random.normal(true_F, self.noise_std) #We introduce some gaussian noise
            observed_F = max(0, min(100, observed_F))  # Ensure valid range

            return STARObservation(observed_F, restored_area)
        return STARObservation(true_F, restored_area)

#### Reward Model
class STARRewardModel(pomdp_py.RewardModel):
    def __init__(self, duration_matrix, F_threshold, travel_cost):
        """
        Args:
        - duration_matrix
        - F_threshold (float): F crit
        - travel_cost (float): Cost per unit travel time F
        """
        self.duration_matrix = duration_matrix
        self.F_threshold = F_threshold
        self.travel_cost = travel_cost
        self.fmax = 100

    def sample(self, state, action, next_state):
        """
        Computes the reward R(s, a).

        Args:
        - state (STARState): The current state before taking action.
        - action (STARAction): The action taken.
        - next_state (STARState): The resulting state.

        Returns:
        - float: The computed reward.
        """
        target_area = action.target_area

        # As computed in original STAR formulation, equivalently opportunity cost
        reward = (self.fmax - state.F_values[target_area]) - sum((self.fmax - F_i) for F_i in next_state.F_values.values())

        return reward

#### Particle belief
class STARParticleBelief(pomdp_py.Particles):
    def __init__(self, num_particles, num_areas, mu_0, sigma_0, initial_location, noise_std=5.0):
        """
        Initializes the belief with sampled particles.

        Args:
        - num_particles (int): Number of particles.
        - num_areas (int): Number of environmental areas.
        - mu_0 (float): Initial mean estimate of F.
        - sigma_0 (float): Standard deviation of initial F estimate.
        - initial_location (int): The robot's starting location.
        - noise_std (float): Additional noise to ensure belief diversity.
        """
        self.num_particles = num_particles
        self.noise_std = noise_std
        particles = []
        for _ in range(num_particles):
            F_values = {i+1: max(0, min(100, random.gauss(mu_0, sigma_0) + random.gauss(0, noise_std))) for i in range(num_areas)}
            particles.append(STARState(F_values, initial_location))
        super().__init__(particles)

    def update(self, action, observation, observation_model):
        """
        Updates the belief using a resampling-based particle filter.

        Args:
        - action (STARAction): Action taken.
        - observation (STARObservation): Observation received.
        - observation_model (STARObservationModel): The observation model.
        """
        new_particles = []
        num_particles = len(self.particles)

        for _ in range(num_particles):
            sampled_state = random.choice(self.particles)
            prob = observation_model.probability(observation, sampled_state, action)

            # Instead of removing particles, re-weight them
            if random.random() < max(prob, 0.20):  # Ensure at least 20% chance of survival
                perturbed_F_values = {
                    i: max(0, min(100, F + random.gauss(0, 5)))  # Add small noise
                    for i, F in sampled_state.F_values.items()
                }
                new_particles.append(STARState(perturbed_F_values, sampled_state.location))

        # Ensure particles are not lost
        if len(new_particles) < self.num_particles // 2:
            print(f"[DEBUG] Particle deprivation detected! Resampling and injecting noise for diversity...")
            missing_particles = self.num_particles - len(new_particles)
            for _ in range(missing_particles):
                perturbed_state = random.choice(self.particles)
                perturbed_F_values = {
                    i: max(0, min(100, F + random.gauss(0, 10)))  # Inject more noise
                    for i, F in perturbed_state.F_values.items()
                }
                new_particles.append(STARState(perturbed_F_values, perturbed_state.location))

        # Replace with the new set of particles
        print("Length of updated particles", len(new_particles))
        super().__init__(new_particles)

#### Policy Model
class STARPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, num_areas):
        """
        Policy model used for rollout in POMCP.

        Args:
        - num_areas (int): Total number of areas.
        """
        self.num_areas = num_areas
        self.actions = [STARAction(i+1) for i in range(num_areas)]
        self.exploration_prob = 0.1
    def sample(self, state, **kwargs):
        """
        Samples a random action from available actions.

        Args:
        - state (STARState): The current state.

        Returns:
        - STARAction: A randomly chosen action.
        """
        possible_actions = [i+1 for i in range(self.num_areas) if i != state.location]
        return STARAction(random.choice(possible_actions))

    def rollout(self, state, history=None):
        return STARAction(random.choice(range(1, self.num_areas+1)))

    def get_all_actions(self, state=None, history=None):
        """
        Returns all possible actions for the given state.
        Overriding the base class method to provide actions.
        """
        return self.actions

#### STAR Environment
class STAREnvironment(pomdp_py.Environment):
    def __init__(self, init_state, transition_model, reward_model):
        """
        Custom environment class for STAR POMDP.

        Args:
        - init_state (STARState): The initial true state of the environment.
        - transition_model (STARTransitionModel): The transition model.
        - reward_model (STARRewardModel): The reward model.
        """
        super().__init__(init_state, transition_model, reward_model)
        self._state = init_state

    def update_state(self, new_state):
        """Allows updating the environment state."""
        self._state = new_state

    @property
    def state(self):
        return self._state


#### STAR Problem
class STARProblem(pomdp_py.POMDP):
    def __init__(self, init_true_state, init_belief, policy_model, transition_model, observation_model, reward_model, env_transition_model):
        """
        STAR POMDP problem definition.

        Args:
        - obs_noise (float): Observation noise (not used in STAR but kept for flexibility).
        - init_true_state (STARState): The true initial state.
        - init_belief (pomdp_py.Particles): The initial belief distribution.
        - num_areas (int): Number of areas.
        """
        agent = pomdp_py.Agent(
            init_belief,
            policy_model=policy_model,
            transition_model = transition_model,
            observation_model = observation_model,
            reward_model = reward_model
        )

        # env = pomdp_py.Environment(init_true_state, env_transition_model, reward_model)
        env = STAREnvironment(init_true_state, env_transition_model, reward_model)

        super().__init__(agent, env, name="STARProblem")


#### Test planner
def test_planner(STAR_problem, planner, nsteps=3):
    """
    Runs the action-feedback loop of STAR problem POMDP

    Args:
        STAR_problem (STARProblem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    schedule = list()
    for i in range(nsteps):
        action = planner.plan(STAR_problem.agent)
        schedule.append(action.target_area)

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {STAR_problem.env.state}")
        print(f"Belief: {STAR_problem.agent.cur_belief}")
        print(f"Action: {action}")
        next_state = STAR_problem.env.transition_model.sample(STAR_problem.env.state, action)
        reward = STAR_problem.env.reward_model.sample(STAR_problem.env.state, action, next_state)
        print("Reward:", reward)
        real_observation = STAR_problem.agent.observation_model.sample(STAR_problem.env.state, action)
        print(">> Observation:", real_observation)

        # Ensure the environment progresses
        STAR_problem.env.update_state(next_state)
        STAR_problem.agent.update_history(action, real_observation)
        STAR_problem.agent.cur_belief.update(action, real_observation, STAR_problem.agent.observation_model)

        if action.name.startswith("restore"):
            print("\n")
    return schedule


