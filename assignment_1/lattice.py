# Piotr Rogula, 249801
import itertools

import numpy as np


class Lattice:
    def __init__(self, size: int, p: float):
        """ Lattice class. For each there is probability p for tree to be placed at initial state.
        States:
            0 = empty
            1 = tree placed
            -1 = burned tree

        Args:
            size (int): >0. size of the L x L lattice """
        self.size = size
        self.lattice = np.array(np.random.random((size, size)) < p, dtype=int)

    def play(self):
        self.burn_the_line('left')
        while self._update_state:
            self._update_state

    def _update_state(self):
        # if there is no update: return false
        burning_trees = np.argwhere(self.lattice == 5).tolist() # change 5 to -1

        trees_to_burn = []
        for tree in burning_trees:
            print(self.get_neighbours(tree))


        return True

    def get_neighbours(self, index: tuple[int,int]):
        nb_row = [it for it in [index[0]-1, index[0], index[0]+1] if 0 <= it < self.size]
        nb_col = [it for it in [index[0] - 1, index[1], index[1] + 1] if 0 <= it < self.size]
        neighbours = itertools.product(nb_row, nb_col)
        return neighbours

    def burn_the_line(self, side: str = 'left'):
        if side == 'left':
            self.lattice[:, 0] = -1
        # TODO: add other sides


if __name__ == "__main__":
    a = Lattice(3, 0.9)
    a.burn_the_line()
    print(a.lattice)
    a.lattice[2,1] = 5
    # l = np.argwhere(a.lattice == -1).tolist()
    # print(l)
    # print(a.lattice[l[1][0],l[1][1]])
    a._update_state()