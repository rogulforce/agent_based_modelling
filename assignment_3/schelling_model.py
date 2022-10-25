# Piotr Rogula, 249801
import itertools
import numpy as np
# from assignment_3.utils import to_list, from_list
# from .utils import to_list, from_list
def from_list(list_of_cords: list[list[int, int]]):
    return [[it[0] for it in list_of_cords], [it[1] for it in list_of_cords]]


def to_list(cords: list[list[int], list[int]]):
    return [it for it in zip(cords[0], cords[1])]

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
                 j_t: tuple[float, float] = (0.5, 0.5)):
        self.n_agents = n_agents
        self.size = size
        self.neighbourhood_range = neighbourhood_range
        self.j_t = j_t

        self.lattice = self._generate_lattice()
        # self.m = self._get_m_parameter()
        self.empty_spots = []

    # def _get_m_parameter(self):
    #     """ Get m (num of neighbours) parameter according to neighbourhood range. """
    #     return 4 *(self.neighbourhood_range+1) * self.neighbourhood_range

    def _generate_lattice(self):
        self.lattice = np.zeros(self.size ** 2 - np.sum(self.n_agents))
        for ind, it in enumerate(self.n_agents):
            self.lattice = np.append(self.lattice, np.ones(it) * (ind + 1))
        np.random.shuffle(self.lattice)
        self.lattice = self.lattice.reshape((self.size, self.size))
        return self.lattice

    def _update_empty_spots(self):
        self.empty_spots = np.where(self.lattice == 0)
        return self.empty_spots

    def get_neighbours(self, agent_index: tuple[int, int]):
        """ Get all the neighbours of given agent_index (row_num, col_num)
        args:
            agent_index: given agent_index
        Returns:
            neighbour list. """
        nb_row = [it for it in range(agent_index[0] - self.neighbourhood_range, agent_index[0] + self.neighbourhood_range + 1)
                  # if 0 <= it < self.size
                  ]
        nb_col = [it for it in range(agent_index[1] - self.neighbourhood_range, agent_index[1] + self.neighbourhood_range + 1)
                  # if 0 <= it < self.size
                  ]
        neighbours = itertools.product(nb_row, nb_col)

        return [it for it in neighbours if it != agent_index]

    def get_happiness(self, agent_index: tuple[int, int], agent_type: int):

        neighbours = from_list(self.get_neighbours(agent_index))
        occupied_cells = np.sum(self.lattice[neighbours[0], neighbours[1]] != 0)
        similar_cells = np.sum(self.lattice[neighbours[0], neighbours[1]] == agent_type)

        return similar_cells / occupied_cells

    def validate_happiness(self, agent_index: tuple[int, int]):
        agent_type = self.lattice[agent_index[0], agent_index[1]]
        return self.get_happiness(agent_index, agent_type) >= self.j_t[int(agent_type)-1]



if __name__ == "__main__":
    a = SchellingModel(size=10, neighbourhood_range=1, n_agents=(50, 30))

    # print(a.get_neighbours((5,8)))
    # print(a._get_m_parameter())
    # print(a.lattice)
    # print())
    # n = from_list(a.get_neighbours((0, 0)))
    # a.lattice[n[0], n[1]] = 8
    index = (2,2)
    print(a.lattice)
    print(a.validate_happiness(index))

