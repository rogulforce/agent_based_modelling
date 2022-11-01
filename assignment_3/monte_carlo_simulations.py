# Piotr Rogula, 249801
from schelling_model import SchellingModel, SchellingModelUnhappy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from IPython.display import Image, display
from multiprocessing import Pool
from functools import partial


class Task:
    """ MC process class """
    @staticmethod
    def single_step(*args, **kwargs):
        raise NotImplementedError

    def simulation(self, model, mc_steps: int, model_params, max_iter):
        """ Function counting percolation of the lattice for <mc_steps> monte carlo steps, being given parameters p, size
        Args:
            p: density
            size:
            mc_steps:
        Returns:

        """
        mc_list = []
        with Pool() as pool:
            f = partial(self.single_step, model, model_params, max_iter)
            mc_list = pool.map(f, range(mc_steps))

        return np.mean(mc_list)


class Task3(Task):
    @staticmethod
    def single_step(model, model_params, max_iter, _):
        """ return number of iterations. """
        sim = model(**model_params)
        it = sim.play(max_iter)
        return it


class Task4(Task):
    @staticmethod
    def single_step(model, model_params, max_iter, _):
        """ return segregation index. """
        sim = model(**model_params)
        sim.play(max_iter)
        segr = sim.get_segregation_index()
        return segr


def task3_plot_data(model, model_params, max_iter, mc_steps, show: bool = True):
    param_change = np.arange(250, 4001, 50)

    val_dict = {it: 0 for it in param_change}

    for value in tqdm(param_change):
        model_params['n_agents'] = (value, value)
        mc_val = Task3().simulation(model=model, mc_steps=mc_steps, model_params=model_params, max_iter=max_iter)
        val_dict[value] = mc_val

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.scatter(val_dict.keys(), val_dict.values())
    plt.title(f'{mc_steps} monte carlo steps\nN from 250 to 4000, size={model_args["size"]}, '
              f'range={model_args["neighbourhood_range"]}, j_t={model_args["j_t"]}')
    plt.xlabel('N')
    plt.ylabel('avg number of iterations')

    if show:
        plt.show()

    return val_dict


def task4_plot_data(model, model_params, max_iter, mc_steps, show: bool = True):
    param_change = np.arange(0.125, 0.90, 0.125)
    # param_change = np.arange(0.04167, 0.9585, 0.04167)

    val_dict = {it: 0 for it in param_change}

    for value in tqdm(param_change):
        model_params['j_t'] = (value, value)
        mc_val = Task4().simulation(model=model, mc_steps=mc_steps, model_params=model_params, max_iter=max_iter)
        val_dict[value] = mc_val

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.scatter(val_dict.keys(), val_dict.values())
    plt.title(f'{mc_steps} monte carlo steps\nj_t from 1/8 to 7/8, size={model_args["size"]}, '
              f'range={model_args["neighbourhood_range"]}, N={model_args["n_agents"]}')
    plt.xlabel('j_t')
    plt.ylabel('segregation index')

    if show:
        plt.show()

    return val_dict


def task5_plot_data(model, model_params, max_iter, mc_steps, show: bool = True):
    param_change = np.arange(1, 5.1, 1)

    val_dict = {it: 0 for it in param_change}

    for value in tqdm(param_change):
        model_params['neighbourhood_range'] = int(value)
        mc_val = Task4().simulation(model=model, mc_steps=mc_steps, model_params=model_params, max_iter=max_iter)
        val_dict[value] = mc_val

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.scatter(val_dict.keys(), val_dict.values())
    plt.title(f'{mc_steps} monte carlo steps\nrange from 1 to 5, size={model_params["size"]}, '
              f'range={model_params["neighbourhood_range"]}, N={model_params["n_agents"]}, j_t={model_params["j_t"]}')
    plt.xlabel('neighbourhood range')
    plt.ylabel('segregation index')

    if show:
        plt.show()

    return val_dict


if __name__ == "__main__":
    # model_args = {'size': 100, 'neighbourhood_range': 1, 'n_agents': (4000, 4000), 'j_t': (0.5, 0.5)}
    # """task 3"""
    # vals = task3_plot_data(SchellingModel, model_args, max_iter=100, mc_steps=12*5, show=False)
    # plt.savefig('data/figs/task3.png')
    # print(vals)
    #
    # """task 3 v2"""
    # vals = task3_plot_data(SchellingModelUnhappy, model_args, max_iter=100, mc_steps=12 * 5, show=False)
    # plt.savefig('data/figs/task3_v2.png')
    # print(vals)
    #
    # model_args = {'size': 100, 'neighbourhood_range': 1, 'n_agents': (250, 250), 'j_t': (0.5, 0.5)}
    #
    # """task 3"""
    # vals = task3_plot_data(SchellingModel, model_args, max_iter=100, mc_steps=12 * 5, show=False)
    # plt.savefig('data/figs/task3_250.png')
    # print(vals)
    #
    # """task 3 v2"""
    # vals = task3_plot_data(SchellingModelUnhappy, model_args, max_iter=100, mc_steps=12 * 5, show=False)
    # plt.savefig('data/figs/task3_250_v2.png')
    # print(vals)


    # """task 4"""
    # vals = task4_plot_data(SchellingModel, model_args, max_iter=300, mc_steps=24, show=False)
    # plt.savefig('data/figs/task4_1500.png')
    # print(vals)
    #
    # """task 4 v2"""
    # vals = task4_plot_data(SchellingModelUnhappy, model_args, max_iter=300, mc_steps=24, show=False)
    # plt.savefig('data/figs/task4_1500_v2.png')
    # print(vals)
    """task 4 range = 2"""
    # vals = task4_plot_data(SchellingModel, model_args, max_iter=300, mc_steps=24, show=False)
    # plt.savefig('data/figs/task4_range_2.png')
    # print(vals)

    model_args = {'size': 100, 'neighbourhood_range': 1, 'n_agents': (250, 250), 'j_t': (0.5, 0.5)}
    """task 5"""
    vals = task5_plot_data(SchellingModel, model_args, max_iter=300, mc_steps=48, show=False)
    plt.savefig('data/figs/task5_250.png')
    print(vals)

    vals = task5_plot_data(SchellingModelUnhappy, model_args, max_iter=300, mc_steps=48, show=False)
    plt.savefig('data/figs/task5_250_v2.png')
    print(vals)

    model_args = {'size': 100, 'neighbourhood_range': 1, 'n_agents': (2500, 2500), 'j_t': (0.5, 0.5)}
    """task 5"""
    vals = task5_plot_data(SchellingModel, model_args, max_iter=300, mc_steps=48, show=False)
    plt.savefig('data/figs/task5_2500.png')
    print(vals)

    vals = task5_plot_data(SchellingModelUnhappy, model_args, max_iter=300, mc_steps=48, show=False)
    plt.savefig('data/figs/task5_2500_v2.png')
    print(vals)

    model_args = {'size': 100, 'neighbourhood_range': 1, 'n_agents': (250, 250), 'j_t': (0.625, 0.625)}
    """task 5"""
    vals = task5_plot_data(SchellingModel, model_args, max_iter=300, mc_steps=48, show=False)
    plt.savefig('data/figs/task5_j0625.png')
    print(vals)

    vals = task5_plot_data(SchellingModelUnhappy, model_args, max_iter=300, mc_steps=48, show=False)
    plt.savefig('data/figs/task5_j0625_v2.png')
    print(vals)
