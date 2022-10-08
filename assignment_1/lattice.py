# Piotr Rogula, 249801
import itertools

import numpy as np


class Lattice:
    def __init__(self, size: int, p: float):
        """ Lattice class. For each there is probability p for tree to be placed at initial state.
        States:
            0 = empty
            1 = tree placed
            2 = burning tree
            -1 = burned tree

        Args:
            size (int): >0. size of the L x L lattice """
        self.size = size
        self.lattice = np.array(np.random.random((size, size)) < p, dtype=int)

        self.burning_trees = []
        self.burned_trees = []
        self.living_trees = []

        self._update_living_trees()

        self.run = True

    def _update_burning_trees(self):
        self.burning_trees = np.where(self.lattice == 2)

    def _update_burned_trees(self):
        self.burned_trees = np.where(self.lattice == -1)

    def _update_living_trees(self):
        self.living_trees = np.where(self.lattice == 1)

    def play(self, gif = False):
        self._set_initial_fire('left')
        while self.run:
            print(self.lattice)
            self.run = self._update_state()

    def _update_state(self):
        """ """
        if not (self.burning_trees[0].size and self.burning_trees[1].size): # if there is nothing to burn
            print('end of data')
            return False

        # get trees to be burned
        trees_to_burn = []
        for tree in to_list(self.burning_trees):
            # print([it for it in self.get_neighbours(tree)])
            trees_to_burn.extend([it for it in self.get_neighbours(tree)])

        # print('living',to_list(self.living_trees))
        # print('toburn',trees_to_burn)

        trees_to_burn = [it for it in trees_to_burn if it in to_list(self.living_trees)]
        trees_to_burn = from_list(trees_to_burn)

        # update living trees.
        self._update_living_trees()

        # burning trees -> burned trees
        self._burn_trees()
        # living trees -> burning trees
        self.lattice[trees_to_burn[0], trees_to_burn[1]] = 2
        self._update_burning_trees()

        return True

    def get_neighbours(self, index: tuple[int, int]):
        """ Method getting all the neighbours of given index (row_num, col_num)"""
        nb_row = [it for it in [index[0] - 1, index[0], index[0] + 1] if 0 <= it < self.size]
        nb_col = [it for it in [index[1] - 1, index[1], index[1] + 1] if 0 <= it < self.size]
        neighbours = itertools.product(nb_row, nb_col)
        return neighbours

    def _burn_trees(self):
        """ Method burning trees."""
        self.lattice[self.burning_trees[0], self.burning_trees[1]] = -1

    def _set_initial_fire(self, side: str = 'left'):
        """ Method setting initial fire on the lattice. """
        if side == 'left':
            self.lattice[:, 0] = self.lattice[:, 0] * 2
        # TODO: add other sides

        self._update_burning_trees()


def from_list(list_of_cords: list[list[int, int]]):
    return [[it[0] for it in list_of_cords],[it[1] for it in list_of_cords]]


def to_list(cords: list[list[int], list[int]]):
    return [it for it in zip(cords[0], cords[1])]


if __name__ == "__main__":
    a = Lattice(5, 0.5)

    """ check 1"""
    # a._set_initial_fire()
    # print(a.lattice)
    # l = np.where(a.lattice == 2)[1]
    # print(l)

    a.play()

