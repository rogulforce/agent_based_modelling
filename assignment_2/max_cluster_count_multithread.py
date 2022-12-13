# Piotr Rogula, 249801
from game_of_life import Forest, GifTool, WindyForest
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from IPython.display import Image, display
from multiprocessing import Pool
from functools import partial


def single_step(forest, _):
    forest.__init__(forest.size, forest.density)
    max_cluster = forest.get_max_cluster_size()
    return max_cluster


def max_cluster_size(p: float, size: int, mc_steps: int):
    """ Function counting percolation of the lattice for <mc_steps> monte carlo steps, being given parameters p, size
    Args:
        p: density
        size:
        mc_steps:
    Returns:

    """

    forest = Forest(size=size, p=p)
    size_list = []
    with Pool() as pool:
        f = partial(single_step, forest)
        size_list = pool.map(f, range(mc_steps))
    return np.mean(size_list)


def plot_max_cluster_size(size: int, mc_steps: int, p_step: float = 0.02, p_min: float = 0, p_max: float = 1,
                          show: bool = True):
    # percolation p*(p)
    p_list = np.arange(p_min, p_max + p_step, p_step)

    size_dict = {prb: 0 for prb in p_list}

    for prb in tqdm(p_list):
        size_p = max_cluster_size(p=prb, size=size, mc_steps=mc_steps)
        size_dict[prb] = size_p

    plt.figure(figsize=(10, 5))
    plt.scatter(size_dict.keys(), size_dict.values())
    plt.title(f'avg max cluster size of p, for {mc_steps} monte carlo steps, lattice {size}x{size}')
    plt.xlabel('p')
    plt.ylabel('avg max cluster size')
    plt.grid()
    if show:
        plt.show()

    return size_dict


if __name__ == "__main__":
    size = 100
    mc_steps = 36  # since I have 12 processes at most.
    p_step = 0.01
    plot_max_cluster_size(size=size, mc_steps=mc_steps, p_step=p_step, p_min=0.35, p_max=0.45, show=False)
    plt.savefig('data/figs/max_cluster_size_2.png')

