# Piotr Rogula, 249801
import itertools
import numpy as np
# from assignment_3.utils import to_list, from_list

# try:
from utils import to_list, from_list
# except:
#     from .utils import to_list, from_list
import random
from copy import copy
from gif_tool import GifTool

# 1 iteration - get all unhappy agents, move it one by one

# loop, 1 iteration:
# 1. get list of all agents
# 2. shuffle the list
# 3. iterate over the list:
# 3a. check if agent is unhappy
# 3b. if so, move him

# use periodic boundary condition (ostatni jest sąsiadem pierwszego)


class SchellingModel:
    def __init__(self, size: int = 100, neighbourhood_range: int = 1, n_agents: tuple[int, int] = (250, 250),
                 j_t: tuple[float, float] = (0.5, 0.5), gif_tool: None | GifTool = None):
        self.n_agents = n_agents
        self.size = size
        self.neighbourhood_range = neighbourhood_range
        self.j_t = j_t

        self.lattice = self._generate_lattice()
        # self.m = self._get_m_parameter()
        self.empty_spots = []
        self.agents = []

        self.run = True

        self.gif_tool = gif_tool

    def _generate_lattice(self):
        self.lattice = np.zeros(self.size ** 2 - np.sum(self.n_agents))
        for ind, it in enumerate(self.n_agents):
            self.lattice = np.append(self.lattice, np.ones(it) * (ind + 1))
        np.random.shuffle(self.lattice)
        self.lattice = self.lattice.reshape((self.size, self.size))
        return self.lattice

    def _update_empty_spots(self):
        self.empty_spots = np.argwhere(self.lattice == 0).tolist()
        return self.empty_spots

    def _update_agents(self):
        self.agents = np.argwhere(self.lattice != 0).tolist()
        return self.agents

    def get_neighbours(self, agent_index: tuple[int, int]):
        """ Get all the neighbours of given agent_index (row_num, col_num)
        args:
            agent_index: given agent_index
        Returns:
            neighbour list. """
        nb_row = [it % self.size for it in range(agent_index[0] - self.neighbourhood_range,
                  (agent_index[0] + self.neighbourhood_range + 1))]
        nb_col = [it % self.size for it in range(agent_index[1] - self.neighbourhood_range,
                  (agent_index[1] + self.neighbourhood_range + 1))]
        neighbours = itertools.product(nb_row, nb_col)
        return [it for it in neighbours if it != agent_index]

    def get_happiness(self, agent_index: tuple[int, int], agent_type: int):
        neighbours = from_list(self.get_neighbours(agent_index))
        occupied_cells = np.sum(self.lattice[neighbours[0], neighbours[1]] != 0)
        similar_cells = np.sum(self.lattice[neighbours[0], neighbours[1]] == agent_type)
        # print(f'neighbours: {neighbours}')
        # print(f'{similar_cells} / {occupied_cells}')

        if occupied_cells == 0:
            return self.__no_neighbours_validation()  # returns 1

        return similar_cells / occupied_cells

    def _get_type(self, agent_index):
        return self.lattice[agent_index[0], agent_index[1]]

    def validate_happiness(self, agent_index: tuple[int, int], agent_type: int):
        return self.get_happiness(agent_index, agent_type) >= self.j_t[int(agent_type)-1]

    def play(self, iter_num):

        for i in range(iter_num):
            # image save
            if self.gif_tool:
                self.gif_tool.save_pic(data=self.lattice, title=f'iteration {i}')
            self.single_iteration()
            if not self.run:

                # print(f'{i} iterations')
                return i
        # print(self.lattice)
        return iter_num

    def single_iteration(self):
        self.run = False
        self._update_empty_spots()
        list_of_agents = copy(self._update_agents())

        # shuffle
        random.shuffle(list_of_agents)

        for agent in list_of_agents:
            agent_type = self._get_type(agent)

            # print(agent, agent_type, self.validate_happiness(agent, agent_type))

            if not self.validate_happiness(agent, agent_type):  # agent is unhappy
                self.move(agent, agent_type)
                self.run = True

    @staticmethod
    def __no_neighbours_validation():
        return 1

    def move(self, agent, agent_type):
        new_place = random.choice(self.empty_spots)

        self.lattice[agent[0], agent[1]] = 0
        self.lattice[new_place[0], new_place[1]] = agent_type

        self.agents.remove(agent)
        self.empty_spots.append(agent)

        self.agents.append(new_place)
        self.empty_spots.remove(new_place)

    def get_segregation_index(self):
        agents = self._update_agents()

        return np.mean([self.get_happiness(agent, self._get_type(agent)) for agent in agents])


class SchellingModelUnhappy(SchellingModel):
    """ Schelling Model where a cell w/o occupied cells around is considered as unhappy."""
    @staticmethod
    def __no_neighbours_validation():
        return 0


if __name__ == "__main__":
    gif_tool = GifTool()
    a = SchellingModel(size=100, neighbourhood_range=1, n_agents=(250, 250), j_t=(0.5, 0.5), gif_tool=gif_tool)
    # print(a.lattice)
    # print(a.get_neighbours((5,8)))
    # print(a._get_m_parameter())
    # print(a.lattice)
    # print())
    # n = from_list(a.get_neighbours((0, 0)))
    # a.lattice[n[0], n[1]] = 8

    # index = (2, 2)
    # print(a.lattice)
    # print(a.validate_happiness(index))
    a.play(10)
    gif_tool.save_gif('test.gif', duration=1)
    # TODO: test that in jupyter

    print(a.get_segregation_index())