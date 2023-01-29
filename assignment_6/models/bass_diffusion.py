import random

import networkx as nx
from copy import copy
import numpy as np


class BassDiffusionModel:
    """ BassDiffusionModel.
    Based on: http://prac.im.pwr.edu.pl/~szwabin/assets/diff/14.pdf
    """

    def __init__(self, network_size: int, innovators: int, imitators: int):
        self.innovators = innovators
        self.imitators = imitators
        self.network_size = network_size

        # prepare network
        self.network = None
        self.initialize_state()

        # statistics storage
        self.numbers_over_time = [[innovators, imitators]]

        self.run = True

    def initialize_state(self):
        """ """
        innovators_array = np.ones(self.innovators)
        imitators_array = 2 * np.ones(self.imitators)
        # not_adapted_array = np.zeros(self.network_size - self.innovators - self.imitators)

        self.network = np.concatenate([innovators_array, imitators_array])

    def single_step(self, p: float, q: float):
        """ Single event according to the link.
        Args:
            p (flaot): 0 <= p <= 1.
            q (float): 0 <= f <= 1.
        """
        # number of not yet adopted nodes
        potential_adapters = self.network_size - len(self.network)

        if not potential_adapters:
            # no one to adopt, break the loop
            self.run = False
            return

        for _ in range(potential_adapters):
            rnd = np.random.random()
            if rnd < p:
                # become an innovator
                self.network = np.append(self.network, 1)
            elif rnd < p + q * len(self.network)/self.network_size:
                # become an imitator
                self.network = np.append(self.network, 2)

        self.numbers_over_time.append([self._get_innovators(), self._get_imitators()])
        return

    def simulate(self, p: float, q: float):
        """ Method simulating the opinion spread: <num_of_events> steps.
        Args:
            p (flaot): 0 <= p <= 1.
            q (float): 0 <= f <= 1.
        """

        while self.run:
            self.single_step(p, q)
        return self.numbers_over_time

    def _get_innovators(self):
        return np.sum(self.network == 1)

    def _get_imitators(self):
        return np.sum(self.network == 2)


if __name__ == "__main__":
    """ simple check of methods."""
    params = {
        "network_size": 1000,
        "innovators": 0,
        "imitators": 0
    }
    q_voter = BassDiffusionModel(**params)

    vals = q_voter.simulate(0.01, 0.3)
    print(vals)
    print(len(vals))
    #
    # mag = q_voter.simulate(num_of_events=100, p=0, q=4, f=0.5)
    # print(mag)
