# Piotr Rogula, 249801
from schelling_model import SchellingModel, SchellingModelUnhappy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from IPython.display import Image, display
from multiprocessing import Pool
from functools import partial


class Task:

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
        sim = model(**model_params)
        it = sim.play(max_iter)
        return it


class Task4(Task):
    @staticmethod
    def single_step(model, model_params, max_iter, _):
        sim = model(**model_params)
        sim.play(max_iter)
        segr = sim.get_segregation_index()
        return segr


model_args = {'size': 100, 'neighbourhood_range': 1, 'n_agents': (1500, 1500), 'j_t': (0.5, 0.5)}


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

    val_dict = {it: 0 for it in param_change}

    for value in tqdm(param_change):
        model_params['j_t'] = (value, value)
        mc_val = Task4().simulation(model=model, mc_steps=mc_steps, model_params=model_params, max_iter=max_iter)
        val_dict[value] = mc_val

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.scatter(val_dict.keys(), val_dict.values())
    plt.title(f'{mc_steps} monte carlo steps\nj_t from 0.1 to 0.9, size={model_args["size"]}, '
              f'range={model_args["neighbourhood_range"]}, N={model_args["n_agents"]}')
    plt.xlabel('j_t')
    plt.ylabel('segregation index')

    if show:
        plt.show()

    return val_dict


if __name__ == "__main__":

    """task 3"""
    # vals = task3_plot_data(SchellingModel, model_args, max_iter=100, mc_steps=12*5, show=False)
    # plt.savefig('data/figs/task3.png')
    # print(vals)

    """task 3 v2"""
    # vals = task3_plot_data(SchellingModelUnhappy, model_args, max_iter=100, mc_steps=12 * 5, show=False)
    # plt.savefig('data/figs/task3_v2.png')
    # print(vals)

    """task 4"""
    vals = task4_plot_data(SchellingModel, model_args, max_iter=300, mc_steps=24, show=False)
    plt.savefig('data/figs/task4_3000_diff.png')
    print(vals)

    """task 4 v2"""
    task4_plot_data(SchellingModelUnhappy, model_args, max_iter=300, mc_steps=24, show=False)
    plt.savefig('data/figs/task4_3000_diff_v2.png')
    print(vals)

