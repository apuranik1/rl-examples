import argh
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Environment():
    """An abstract base class defining the interface for a finite environment.
    Useful for problems with a tractable number of distinct states that can be
    easily simulated, but not easily described by an MDP.
    """

    @property
    def state(self):
        raise NotImplementedError()

    def state_list(self):
        raise NotImplementedError()

    def get_actions(self, state=None):
        """Return a dictionary of available actions at the current state"""
        raise NotImplementedError()

    def take_action(self, action):
        """Respond to an actor taking an action, returning the reward"""
        raise NotImplementedError()


class Agent():
    """An base class for an agent learning in an environment"""

    def __init__(self, environment):
        self.environment = environment

    def action(self):
        raise NotImplementedError()

    def update(self, oldstate, action, reward, t):
        raise NotImplementedError()

    def episode_end(self):
        pass


class MonteCarloES(Agent):
    """A greedy Monte Carlo control agent which assumes exploring starts.
    Only tracks averages after the first state visit in an epoch.
    """
    def __init__(self, environment):
        super().__init__(environment)
        self.estimates = {state: 0 for state in environment.states}
        states = environment.state_list()
        self.observation_counts = {(state, act): 0
                                   for state in states
                                   for act in environment.get_actions(state)}
        policy = {}
        for state in states:
            # get an arbitrary action for state
            policy[state] = next(iter(environment.get_actions(state)))
        self.policy = policy
        self.first_state = True
        self.first_in_episode = {}  # maps (s, a) to first index in episode
        self.episode_data = []

    def action(self):
        state = self.environment.state
        actions = self.environment.get_actions()
        if self.first_state == True:
            # pick an action at random (exploring start)
            return np.random.choice(list(actions.keys()))
        # pick action according to policy
        return self.policy[state]

    def update(self, oldstate, action, reward, t):
        # record the data, but don't act on it
        newstate = self.environment.state
        index = (oldstate, action)
        if index not in self.first_in_episode:
            self.first_in_episode[index] = len(self.episode_data)
        self.episode_data.append((oldstate, action, reward))

    def episode_end(self):
        # recompute values and reset saved state from episode
        self.first_state = False
