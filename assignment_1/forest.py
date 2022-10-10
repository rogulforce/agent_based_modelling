# Piotr Rogula, 249801
import itertools
import numpy as np

# gif
from matplotlib import pyplot as plt
from matplotlib import colors
from natsort import natsorted
import os
import imageio.v2 as imageio


class Forest:
    def __init__(self, size: int, p: float, gif_tool=None):
        """ Forest class. For each there is probability p for tree to be placed at initial state.
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

        self.gif_tool = gif_tool

    def _update_burning_trees(self):
        self.burning_trees = np.where(self.lattice == 2)

    def _update_burned_trees(self):
        self.burned_trees = np.where(self.lattice == -1)

    def _update_living_trees(self):
        self.living_trees = np.where(self.lattice == 1)

    def play(self, gif = False):

        self._set_initial_fire('left')
        while self.run:

            # image save
            if self.gif_tool:
                self.gif_tool.save_pic(data=self.lattice)

            self.run = self._update_state()

    def _update_state(self):
        """ """
        if not (self.burning_trees[0].size and self.burning_trees[1].size): # if there is nothing to burn
            # print('end of data') # debug
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

    def _audit_trees_to_burn(self, tree_list: list[list[int,int]]):
        """ method removing neighbours which are not trees and which are already burning."""
        trees_to_burn = [it for it in tree_list if it in to_list(self.living_trees) and
                         it not in to_list(self.burning_trees)]
        return from_list(trees_to_burn)

    def get_neighbours(self, index: tuple[int, int]):
        """ Method getting all the neighbours of given index (row_num, col_num)"""
        nb_row = [it for it in [index[0] - 1, index[0], index[0] + 1] if 0 <= it < self.size]
        nb_col = [it for it in [index[1] - 1, index[1], index[1] + 1] if 0 <= it < self.size]
        neighbours = itertools.product(nb_row, nb_col)
        return neighbours

    def _burn_trees(self):
        """ Method getting trees from state 'burning' to 'burned'"""
        self.lattice[self.burning_trees[0], self.burning_trees[1]] = -1

    def _set_initial_fire(self, side: str = 'left'):
        """ Method setting initial fire on the lattice. """
        if side == 'left':
            self.lattice[:, 0] = self.lattice[:, 0] * 2
        # TODO: in future: add other sides.

        self._update_burning_trees()

    def fire_hit_edge(self, side: str = 'right'):
        if side == 'right':
            return -1 in self.lattice[:, -1]
        # TODO: in future: add other sides

    def save_gif(self):
        if self.gif_tool:
            self.gif_tool.save_gif()


class GifTool:
    def __init__(self, pic_dir: str = 'data/temp', gif_dir: str = 'data/gif'):
        self.pic_dir = pic_dir
        self.gif_dir = gif_dir

        self.clear_dir(self.pic_dir)

        self.cmap = colors.ListedColormap(['grey', 'white', 'green', 'orange'])
        bounds = [-1.5,-.5, .5, 1.5, 2.5]
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)
        self.img_num = 0

    def save_pic(self, data):
        self.visualize(data)
        plt.savefig(f'data/temp/fig{self.img_num}')
        plt.close()
        self.img_num += 1

    def visualize(self, data):
        plt.figure(figsize=(10, 10))
        img = plt.imshow(data, interpolation='nearest', origin='lower',
                         cmap=self.cmap, norm=self.norm)
        # return img

    def save_gif(self, name, **kwargs):
        images = []
        for file_name in natsorted(os.listdir(self.pic_dir)):
            if file_name.endswith('.png'):
                file_path = os.path.join(self.pic_dir, file_name)
                images.append(imageio.imread(file_path))

        imageio.mimsave(f'{self.gif_dir}/{name}', images, **kwargs)

    @staticmethod
    def clear_dir(directory):
        for file_name in os.listdir(directory):
            if file_name.endswith('.png'):
                os.remove(f'{directory}/{file_name}')


class WindyForest(Forest):
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


def from_list(list_of_cords: list[list[int, int]]):
    return [[it[0] for it in list_of_cords],[it[1] for it in list_of_cords]]


def to_list(cords: list[list[int], list[int]]):
    return [it for it in zip(cords[0], cords[1])]


if __name__ == "__main__":
    # a = Forest(10, 0.5, gif_tool=GifTool())
    #
    # a.play()
    #
    # a.gif_tool.save_gif('1.gif')

    a = WindyForest(10, 0.5, gif_tool=GifTool(), wind_power=0.5)

    a.play()

    a.gif_tool.save_gif('1.gif', duration=1)

