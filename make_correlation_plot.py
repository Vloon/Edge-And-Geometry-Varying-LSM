from helper_functions import get_cmd_params, set_GPU, triu2mat, lorentz_to_poincare

arguments = [('-gpu', 'gpu', str, ''),
             ('-k', 'k', float, 1.5),
             ('-sigz', 'sigz_vals', float, [1.], '+'),
             ]

cmd_params = get_cmd_params(arguments)
gpu = cmd_params['gpu']
set_GPU(gpu)

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# gt_vs_learned_dist_correlations_hyp_k1.5_sig0.5
hyperbolic = True
k = cmd_params['k']
sigz_vals = cmd_params['sigz_vals']

con_fig = plt.figure(0)
bin_fig = plt.figure(1)
figs = [con_fig, bin_fig]

for sigz in sigz_vals:
    correlations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"gt_vs_learned_dist_correlations_{'hyp' if hyperbolic else 'euc'}_k{k}_sig{sigz}.pkl")
    with open(correlations_filename, 'rb') as f:
        correlations = pickle.load(f) # 2, n_tasks, n_subjects, num_particles

    _, n_tasks, n_subjects, num_particles = correlations.shape

    for et_i, edge_type in enumerate(['con', 'bin']):
        _correlations = correlations[et_i]
        plt.figure(et_i)
        ax = plt.gca()
        ax.scatter(np.repeat(sigz, n_tasks*n_subjects*num_particles), _correlations.ravel(), s=1, color='k', alpha=0.05)

for et_i, edge_type in enumerate(['con', 'bin']):
    ax = figs[et_i].get_axes()
    filename =  os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"gt_vs_learned_dist_correlations_{edge_type}_{'hyp' if hyperbolic else 'euc'}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()




