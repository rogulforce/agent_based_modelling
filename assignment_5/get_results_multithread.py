# Piotr Rogula, 249801
import matplotlib.pyplot as plt
from models import QVoter
import numpy as np
import networkx as nx
from tqdm import tqdm

import pandas as pd
import pickle
from multiprocessing import Pool
from functools import partial


class BaseMC:
    """ MC process class """
    @staticmethod
    def single_step(*args, **kwargs):
        raise NotImplementedError

    def simulation(self, num_of_steps, N, q, f, p, MC_runs):
        """ complete graph """
        network = nx.complete_graph(N)
        q_voter = QVoter(network)
        all_data = []

        for q_i in q:
            print(f'q={q_i}')
            for f_i in f:
                print(f'f={f_i}')
                for p_i in tqdm(p):
                    model_params = {'q': q_i, 'f': f_i, 'p': p_i, 'num_of_events': num_of_steps}
                    with Pool() as pool:
                        # partial - to give parameters of the function
                        f = partial(self.single_step, q_voter, model_params)
                        concentration_over_time = pool.map(f, range(MC_runs))

                    all_data.append((f'complete-graph({N})',
                        q_i,
                        p_i,
                        f_i,
                        np.mean(concentration_over_time, 0)
                    ))
        return all_data


class GetConcentrationMC(BaseMC):
    @staticmethod
    def single_step(model, model_params, _):
        """ return number of iterations. """
        concentration = model.simulate(**model_params)
        return concentration


if __name__ == "__main__":
    # parameters to run
    num_of_steps = 5
    N = 100
    q = [4]
    f = [0.2, 0.3, 0.4, 0.5]
    # f = [0.5]
    p = np.arange(0, 1.01, 0.025)

    MC_runs = 996

    # data collection
    """
    all the data is storaged in the list:
    [
    [name_of_initial_graph, q_0, p_0, f_0, [average concentration for each of <num_of_steps> steps]
    [name_of_initial_graph, q_1, p_1, f_1, [average concentration for each of <num_of_steps> steps]
    .
    .
    .
    [name_of_initial_graph, q_n, p_n, f_n, [average concentration for each of <num_of_steps> steps]
    ]
    """
    # proceed MC
    all_data = GetConcentrationMC().simulation(num_of_steps, N, q, f, p, MC_runs)

    # save data
    df = pd.DataFrame(all_data, columns=['graph_name', 'q', 'p', 'f', 'avg_concentration_over_time'])
    with open('data/all_data.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(df)
