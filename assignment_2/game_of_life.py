# Piotr Rogula, 249801
import itertools

import numpy as np

from assignment_2.const import OCCUPIED_CELLS

from copy import copy
from gif_tool import GifTool


class GameOfLife:
    """ Schnelling Model implementation."""
    def __init__(self, size: int = 100, boundaries: str = 'open', gif_tool: None | GifTool = None):
        """
        Args:
            size: size of the square lattice
            boundaries: 'open' or 'closed'
            gif_tool: visualization tool
        """
        self.size = size
        self.neighbourhood_range = 1
        self.boundaries = boundaries
        self.lattice = None

        self.gif_tool = gif_tool

    def __generate_mess_lattice(self):
        i = []
        j = []
        g1 = OCCUPIED_CELLS['glider']
        g2 = [[el + 8 for el in g1[0]], [el for el in g1[1]]]
        g3 = [[self.size-1 - ((el-max(g1[0]))*(-1) + max(g1[0])) for el in g1[0]], [self.size-1 - ((el-max(g1[1]))*(-1) + max(g1[1])) for el in g1[1]]]
        for item in [g2, g3]:
            i += item[0]
            j += item[1]
        self.lattice[i, j] = 1
        return self.lattice

    def generate_lattice(self, init_type: str):
        """ generate initial lattice."""
        self.lattice = np.zeros((self.size, self.size))
        if not init_type == 'mess':
            occupied_points = OCCUPIED_CELLS[init_type]
            self.lattice[occupied_points[0], occupied_points[1]] = 1
        else:
            self.__generate_mess_lattice()
        return self.lattice

    def get_neighbours(self, ind: tuple[int, int]):
        """ Get all the neighbours of given ind (row_num, col_num)
        args:
            ind: given ind
        Returns:
            neighbour list. """
        if self.boundaries == 'open':
            nb_row = [it % self.size for it in range(ind[0] - 1, (ind[0] + 1 + 1))]
            nb_col = [it % self.size for it in range(ind[1] - 1, (ind[1] + 1 + 1))]
            neighbours = itertools.product(nb_row, nb_col)
            return [it for it in neighbours if tuple(it) != ind]
        elif self.boundaries == 'closed':
            nb_row = [it for it in [ind[0] - 1, ind[0], ind[0] + 1] if 0 <= it < self.size]
            nb_col = [it for it in [ind[1] - 1, ind[1], ind[1] + 1] if 0 <= it < self.size]
            neighbours = itertools.product(nb_row, nb_col)
            return [it for it in neighbours if tuple(it) != ind]
        else:
            raise NotImplementedError

    def _get_type(self, ind: tuple[int, int]):
        """ get agent type. """
        return self.lattice[ind[0], ind[1]]

    def change_cells_state(self):
        new_lattice = copy(self.lattice)
        for i in range(self.size):
            for j in range(self.size):
                cell_type = self._get_type((i, j))
                neigh_num = len([it for it in self.get_neighbours((i, j)) if self._get_type(it) == 1])
                if cell_type == 1:
                    # living cell
                    if neigh_num < 2 or neigh_num > 3:
                        new_lattice[i, j] = 0
                else:
                    # dead cell
                    if neigh_num == 3:
                        new_lattice[i, j] = 1
        return new_lattice

    def play(self, iter_num):
        """ process simulation. """
        for i in range(iter_num):
            # image save
            if self.gif_tool:
                self.gif_tool.save_pic(data=self.lattice, title=f'iteration {i}')
            self.single_iteration()
        self.gif_tool.save_pic(data=self.lattice, title=f'iteration {iter_num}')
        return iter_num

    def single_iteration(self):
        """ one iteration. """
        self.lattice = self.change_cells_state()


if __name__ == "__main__":
    # # 'gospel_glider_gun'
    # print('gospel closed')
    # gif_tool = GifTool()
    # lattice = GameOfLife(size=40, gif_tool=gif_tool, boundaries='closed')
    # lattice.generate_lattice(init_type='gospel_glider_gun')
    # lattice.play(200)
    # print('saving...')
    # gif_tool.save_gif('gospel_glider_gun_closed.gif')

    # # 'gospel_glider_gun'
    # print('gospel open')
    # gif_tool = GifTool()
    # lattice = GameOfLife(size=40, gif_tool=gif_tool, boundaries='open')
    # lattice.generate_lattice(init_type='gospel_glider_gun')
    # lattice.play(300)
    # print('saving...')
    # gif_tool.save_gif('gospel_glider_gun_open.gif')

    # # 'blinker'
    # print('blinker')
    # gif_tool = GifTool()
    # lattice = GameOfLife(size=10, gif_tool=gif_tool, boundaries='open')
    # lattice.generate_lattice(init_type='blinker')
    # lattice.play(30)
    # print('saving...')
    # gif_tool.save_gif('blinker.gif')

    # print('middle_weight_spaceship')
    # gif_tool = GifTool()
    # lattice = GameOfLife(size=20, gif_tool=gif_tool, boundaries='open')
    # lattice.generate_lattice(init_type='middle_weight_spaceship')
    # lattice.play(100)
    # print('saving...')
    # gif_tool.save_gif('middle_weight_spaceship_open.gif')
    #
    # print('middle_weight_spaceship')
    # gif_tool = GifTool()
    # lattice = GameOfLife(size=20, gif_tool=gif_tool, boundaries='closed')
    # lattice.generate_lattice(init_type='middle_weight_spaceship')
    # lattice.play(100)
    # print('saving...')
    # gif_tool.save_gif('middle_weight_spaceship_closed.gif')

    print('mess')
    gif_tool = GifTool()
    lattice = GameOfLife(size=30, gif_tool=gif_tool, boundaries='open')
    lattice.generate_lattice(init_type='mess')
    lattice.play(500)
    print('saving...')
    gif_tool.save_gif('mess.gif')
