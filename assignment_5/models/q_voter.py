import networkx as nx
from copy import copy
import numpy as np


class QVoter:
    """ q-voter model with NN influence group. """

    def __init__(self, init_network: nx.Graph):
        self.init_network = init_network
        self.network_size = init_network.size()

        self.operating_network = None
        self.operating_opinion = None
        self.operating_concentration = []

    def reload_operating_network(self):
        """ Operating network is needed for Monte Carlo trajectories. """
        self.operating_network = copy(self.init_network)

    def reload_operating_opinion(self):
        """ Method initializing opinion of the spinsons to 1. In future this could be changed and improved. """
        self.operating_opinion = {node: 1 for node in self.init_network.nodes}

    def reload_operating_magnetization(self):
        self.operating_concentration = []

    def influence_choice(self, spinson: int, q: int, type_of_influence: str = 'NN') -> list:
        """ Method returning spinsons from the network to affect given <spinson (int)> according to given theoretical
            <type_of_influence (int)>
        Args:
            spinson (int):  given spinson.
            q (int): number of people in the influence group
            type_of_influence (str): type of choice of the influence group.
        """
        if type_of_influence == 'NN':
            # 'q randomly chosen nearest neighbours of the target spinson are in the group.'
            return np.random.choice([neighbour for neighbour in self.operating_network.neighbors(spinson)], q)
        else:
            # in the future there may be other ways of choice implemented as well
            raise NotImplementedError

    def unanimous_check(self, group: list[int]):
        """ Method checking if the group is unanimous.
        Args:
            group (list[int]): Given group"""
        # only if (all are equal to 1) v (all are equal to -1)  <==> abs(sum(group_opinions)) = len(group)
        opinions = [self.operating_opinion[member] for member in group]
        return abs(sum(opinions)) == len(group)

    def single_step(self, p: float, q: int, f: float, type_of_influence: str = 'NN'):
        """ Single event accroding to the paper.
        Args:
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q (int): number of people in the influence group
            f (float): 0 <= f <= 1. Flexibility. In the case of independent behavior, with probability f the spinson changes its
                       opinion and with 1-f stays with the current opinion.
            type_of_influence (str): type of choice of the influence group.
        """
        # pick network size spinsons at random
        spinsons = np.random.choice(self.operating_network.nodes, self.network_size)

        for spinson in spinsons:
            if np.random.random() < p:
                # independent state, change its opinion with probability f
                if np.random.random() < f:
                    opinion = self.operating_opinion[spinson]
                    self.operating_opinion[spinson] = -1 * opinion
            else:
                # randomly chosen group of influence.
                influence_group = self.influence_choice(spinson, q, type_of_influence)
                if self.unanimous_check(influence_group):
                    # if not independent, let the spinson take the opinion of its group of influence.
                    self.operating_opinion[spinson] = self.operating_opinion[list(influence_group)[0]]

    def simulate(self, num_of_events: int, p: float, q: int, f: float, type_of_influence: str = 'NN'):
        """ Method simulating the opinion spread: <num_of_events> steps.
        Args:
            num_of_events: number of iterations (time).
            p (float): 0 <= p <= 1. Probability for spinson to be independent
            q (int): number of people in the influence group
            f (float): 0 <= f <= 1. Flexibility. In the case of independent behavior, with probability f the spinson changes its
                       opinion and with 1-f stays with the current opinion.
            type_of_influence (str): type of choice of the influence group.
        """
        self.initialize_simulation()

        for event in range(num_of_events):
            # single iteration
            self.single_step(p, q, f, type_of_influence)
            # add current concentration to the list
            self.update_concentration_list()

        return self.operating_concentration

    def calculate_global_concentration(self):
        """ Method calculating global concentration. Positive/all"""
        return len([opinion for opinion in self.operating_opinion.values() if opinion == 1]) / len(
               self.operating_opinion)

    def calculate_magnetization(self):
        """ Method calculating magnetization. """
        return np.mean(list(self.operating_opinion.values()))

    def update_concentration_list(self):
        """ Method updating magnetization list with current magnetization. """
        self.operating_concentration.append(self.calculate_global_concentration())

    def initialize_simulation(self):
        """ Method initializing operating values, i.e. clearing them. """
        # cleaning operating network
        self.reload_operating_network()
        # cleaning operating opinion
        self.reload_operating_opinion()
        # cleaning magnetization
        self.reload_operating_magnetization()


if __name__ == "__main__":
    """ simple check of methods."""
    n = 100
    network = nx.complete_graph(n)
    # print(network)

    q_voter = QVoter(network)

    mag = q_voter.simulate(num_of_events=100, p=0, q=4, f=0.5)
    print(mag)