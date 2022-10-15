# Piotr Rogula, 249801
import itertools
import numpy as np
from utils import to_list, from_list


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

    def get_neighbours(self, index: tuple[int, int]):
        """ Get all the neighbours of given index (row_num, col_num)
        args:
            index: given index
        Returns:
            neighbour list. """
        nb_row = [it for it in range(index[0] - self.neighbourhood_range, index[0] + self.neighbourhood_range + 1)
                  if 0 <= it < self.size]
        nb_col = [it for it in range(index[1] - self.neighbourhood_range, index[1] + self.neighbourhood_range + 1)
                  if 0 <= it < self.size]
        neighbours = itertools.product(nb_row, nb_col)
        # m = len(neighbours)
        return [it for it in neighbours if it != index]


if __name__ == "__main__":
    a = SchellingModel(size=10, neighbourhood_range=4, n_agents=(50, 30))

    # print(a.get_neighbours((5,8)))
    # print(a._get_m_parameter())
    print(a.lattice)
