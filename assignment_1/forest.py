# Piotr Rogula, 249801
import itertools
import random
from copy import copy

import numpy as np

from gif_tool import GifTool
from utils import to_list, from_list


class Forest:
    def __init__(self, size: int, p: float, gif_tool: None | GifTool = None):
        """ Forest class. For each there is probability p for tree to be placed at initial state.
        States:
            0 = empty
            1 = tree placed
            2 = burning tree
            -1 = burned tree

        Args:
            size (int): >0. size of the L x L lattice
            p (float): density of the forest
            gif_tool (optional). Visualisation tool. Defaults to None."""

        if not 0 <= p <= 1:
            raise "density shall be in range [0,1]"

        self.size = size
        self.density = p
        self.lattice = np.array(np.random.random((size, size)) < p, dtype=int)
        self.initial_lattice = copy(self.lattice)

        self.burning_trees = []
        self.burned_trees = []
        self.living_trees = []

        self._update_living_trees()

        self.run = True

        self.gif_tool = gif_tool

    def _update_burning_trees(self):
        self.burning_trees = np.where(self.lattice == 2)

    def _update_burned_trees(self):
        self.burned_trees = np.where(self.lattice == -1)

    def _update_living_trees(self):
        self.living_trees = np.where(self.lattice == 1)

    def play(self, gif=False):
        """ Execute the model

        Args:
            gif: argument defining visualisation of the play.
        """
        self._set_initial_fire('left')
        while self.run:

            # image save
            if self.gif_tool:
                self.gif_tool.save_pic(data=self.lattice)

            self.run = self._update_state()

    def _update_state(self):
        """ Update state of the play."""
        if not (self.burning_trees[0].size and self.burning_trees[1].size): # if there is nothing to burn
            # end of data
            return False

        # get trees to be burned
        trees_to_burn = []
        for tree in to_list(self.burning_trees):
            # print([it for it in self.get_neighbours(tree)])
            trees_to_burn.extend([it for it in self.get_neighbours(tree)])

        # print('living',to_list(self.living_trees))
        # print('toburn',trees_to_burn)

        trees_to_burn = self._audit_trees_to_burn(trees_to_burn)

        # update living trees.
        self._update_living_trees()

        # burning trees -> burned trees
        self._burn_trees()
        # living trees -> burning trees
        self.lattice[trees_to_burn[0], trees_to_burn[1]] = 2
        self._update_burning_trees()

        return True

    def _audit_trees_to_burn(self, tree_list: list[list[int, int]]):
        """ Remove trees from <tree_list> which are not trees or have state = burning.
        Args:
            tree_list: list of trees to audit.
        Returns:
             audited list"""
        trees_to_burn = [it for it in tree_list if it in to_list(self.living_trees) and
                         it not in to_list(self.burning_trees)]
        return from_list(trees_to_burn)

    def get_neighbours(self, index: tuple[int, int]):
        """ Get all the neighbours of given index (row_num, col_num)
        args:
            index: given index
        Returns:
            neighbour list. """
        nb_row = [it for it in [index[0] - 1, index[0], index[0] + 1] if 0 <= it < self.size]
        nb_col = [it for it in [index[1] - 1, index[1], index[1] + 1] if 0 <= it < self.size]
        neighbours = itertools.product(nb_row, nb_col)
        return neighbours

    def _burn_trees(self):
        """ Get trees from state 'burning' to 'burned'"""
        self.lattice[self.burning_trees[0], self.burning_trees[1]] = -1

    def _set_initial_fire(self, side: str = 'left'):
        """ Set initial fire on the lattice. """
        if side == 'left':
            self.lattice[:, 0] = self.lattice[:, 0] * 2
        # TODO: in future: add other sides.

        self._update_burning_trees()

    def fire_hit_edge(self, side: str = 'right'):
        """ Give boolean telling if fire hit given edge.
        args:
            side: Defaults to 'right'.
        returns:
            bool. True for -1 in the <side> side, otherwise, False."""
        if side == 'right':
            return -1 in self.lattice[:, -1]
        # TODO: in future: add other sides

    def get_max_cluster_size(self, side: str = 'left'):
        """ Get max cluster size.
        args:
            side: starting side
        returns:
            int. Number of trees in max cluster."""

        max_cluster = 0
        # points with initial fire
        self.lattice = copy(self.initial_lattice)
        self._update_living_trees()
        self.run = True

        self._set_initial_fire(side=side)
        initial_fire = np.argwhere(self.lattice == 2).tolist()

        for tree in initial_fire:
            if self.lattice[tree[0], tree[1]] == -1: # tree already burned
                continue

            # restoring burned trees.
            self.lattice = copy(self.initial_lattice)
            self._update_living_trees()
            # set fire in that one tree
            self.lattice[tree[0], tree[1]] = 2
            self._update_burning_trees()

            while self.run:
                self.run = self._update_state()

            max_cluster = np.max((max_cluster, np.count_nonzero(self.lattice == -1)))

            self.run = True

        return max_cluster

    def __add_tree(self):
        rnd_index = random.choice(np.argwhere(self.lattice == 0).tolist())
        self.lattice[rnd_index[0], rnd_index[1]] = 1

    def __restore_trees(self):
        self._update_burned_trees()
        self.lattice[self.burned_trees[0], self.burned_trees[1]] = 1

    def percolation_threshold_estimator(self):
        self.lattice = np.zeros(shape=(self.size, self.size))
        while not self.fire_hit_edge():
            self.__restore_trees()
            self.__add_tree()
            self.burning_trees = []
            self.burned_trees = []
            self.run = True
            self._update_living_trees()

            self.play()

        return (np.count_nonzero(self.lattice == -1) + np.count_nonzero(self.lattice == 1)) / (self.size ** 2)

    def hoshen_kopelman(self):
        """ hoshen kopelman algorithm. """
        largest_label = 4
        for y in range(0, self.size):  # row
            for x in range(0, self.size):  # column
                point = (y, x)
                if (y, x) not in to_list(self.living_trees):
                    continue

                left = self.get_n(point, 'left')
                above = self.get_n(point, 'above')
                left_above = self.get_n(point, 'left_above')
                right_above = self.get_n(point, 'right_above')

                # new label
                if left == 0 and above == 0 and left_above == 0 and right_above == 0:
                    largest_label += 1
                    self.lattice[y, x] = largest_label

                # 1 label
                elif left != 0 and above == 0 and left_above == 0 and right_above == 0:
                    self.lattice[y, x] = left
                elif left == 0 and above != 0 and left_above == 0 and right_above == 0:
                    self.lattice[y, x] = above
                elif left == 0 and above == 0 and left_above != 0 and right_above == 0:
                    self.lattice[y, x] = left_above
                elif left == 0 and above == 0 and left_above == 0 and right_above != 0:
                    self.lattice[y, x] = right_above

                # 2 labels
                elif left != 0 and above != 0 and left_above == 0 and right_above == 0:
                    self.lattice[self.lattice == above] = left
                    self.lattice[y, x] = left
                elif left != 0 and above == 0 and left_above != 0 and right_above == 0:
                    self.lattice[self.lattice == left_above] = left
                    self.lattice[y, x] = left
                elif left != 0 and above == 0 and left_above == 0 and right_above != 0:
                    self.lattice[self.lattice == right_above] = left
                    self.lattice[y, x] = left
                elif left == 0 and above != 0 and left_above != 0 and right_above == 0:
                    self.lattice[self.lattice == left_above] = above
                    self.lattice[y, x] = above
                elif left == 0 and above != 0 and left_above == 0 and right_above != 0:
                    self.lattice[self.lattice == right_above] = above
                    self.lattice[y, x] = above
                elif left == 0 and above == 0 and left_above != 0 and right_above != 0:
                    self.lattice[self.lattice == right_above] = left_above
                    self.lattice[y, x] = left_above

                # 3 labels
                elif left != 0 and above != 0 and left_above != 0 and right_above == 0:
                    self.lattice[self.lattice == left_above] = left
                    self.lattice[self.lattice == above] = left
                    self.lattice[y, x] = left
                elif left != 0 and above != 0 and left_above == 0 and right_above != 0:
                    self.lattice[self.lattice == right_above] = left
                    self.lattice[self.lattice == above] = left
                    self.lattice[y, x] = left
                elif left != 0 and above == 0 and left_above != 0 and right_above != 0:
                    self.lattice[self.lattice == left_above] = left
                    self.lattice[self.lattice == right_above] = left
                    self.lattice[y, x] = left
                elif left == 0 and above != 0 and left_above != 0 and right_above != 0:
                    self.lattice[self.lattice == left_above] = above
                    self.lattice[self.lattice == right_above] = above
                    self.lattice[y, x] = above

                # 4 labels
                elif left != 0 and above != 0 and left_above != 0 and right_above != 0:
                    self.lattice[self.lattice == above] = left
                    self.lattice[self.lattice == left_above] = left
                    self.lattice[self.lattice == right_above] = left
                    self.lattice[y, x] = left

        # get occurences of max label
        max_cnt = 0
        for item in np.unique(self.lattice):
            if item == 0:
                continue
            else:
                max_cnt = max(max_cnt, np.count_nonzero(self.lattice == item))
        return max_cnt

    def get_n(self, point: tuple[int, int], ttype: str):
        """ get value of the neighbour if exists, else: 0"""
        y, x = point
        if ttype == 'left':
            if x > 0:
                return self.lattice[y, x-1]
        elif ttype == 'above':
            if y > 0:
                return self.lattice[y-1, x]
        elif ttype == 'left_above':
            if y > 0 and x > 0:
                return self.lattice[y-1, x-1]
        elif ttype == 'right_above':
            if y > 0 and x < self.size - 1:
                return self.lattice[y-1, x+1]
        return 0


class WindyForest(Forest):
    """ Forest model with added wind. <wind_power> argument defines probability with which fire is being spread
    between the trees.  """
    def __init__(self, size: int, p: float, gif_tool=None, wind_power: float = 1):
        super(WindyForest, self).__init__(size=size, p=p, gif_tool=gif_tool)
        self.wind_power = wind_power

    def _audit_trees_to_burn(self, tree_list: list[list[int, int]]):
        trees_to_burn = [it for it in tree_list if it in to_list(self.living_trees) and
                         it not in to_list(self.burning_trees)]

        # wind power working
        for tree in trees_to_burn:
            if np.random.random() > self.wind_power:
                trees_to_burn.remove(tree)

        return from_list(trees_to_burn)


if __name__ == "__main__":
    a = Forest(size=5, p=.5)
    # print((0,0) in to_list(a.living_trees))
    print(a.hoshen_kopelman())
    print(a.lattice)
