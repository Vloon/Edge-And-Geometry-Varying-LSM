import os

from helper_functions import get_cmd_params, set_GPU, read_seed, write_seed, triu2mat

arguments = [('-gpu', 'gpu', str, ''),
             ('--plot', 'make_plot', bool)
             ]

cmd_params = get_cmd_params(arguments)
gpu = cmd_params['gpu']
set_GPU(gpu)
print(f'Setting GPU to [{gpu}]')

make_plot = cmd_params['make_plot']

import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_default_device", jax.devices()[0])
jax.config.update("jax_enable_x64", True)
import jax.random as jrnd
import jax.numpy as jnp
import blackjax as bjx
import distrax as dx
from typing import Dict, Callable, Tuple
from jaxtyping import Array, Float

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from models import ClusterModel, LSM, GibbsState

###########################
###### Set variables ######
###########################

## General
seed = read_seed('seed.txt')
key = jrnd.PRNGKey(seed)

N = 100 # Total number of nodes
M = N*(N-1)//2 # Total number of edges
D = 2 # Latent dimensions

n_tasks = 1
n_observations = 1 # / subjects?

## Cluster prior
n_clusters =  3
probability_per_cluster = [0.5, 0.25, 0.25]
min_cluster_dist = 0. # Increase to make clusters more distant
sigmas =[1.]*n_clusters # Standard deviations for each cluster

k = [3.]*n_clusters # Shape parameter of the gamma distribution
theta = [1]*n_clusters # Scale parameter of the gamma distribution

mu_sigma = 0. # mean of the sigma distribution
sigma_sigma = 1. # standard deviation of the sigma distribution

hyperbolic = True

## Inference
mu_z = 0.
sigma_z = 1.

num_mcmc_steps = 100
num_particles = 1000

#####################################################################################
###### Sample a prior with a number of clusters in the position's ground truth ######
#####################################################################################

latpos = '_z' if hyperbolic else 'z'

all_observations = np.zeros((n_tasks, n_observations, M))
gt_distances = np.zeros((n_tasks, M))

for task in range(n_tasks):
    print(f"Sampling ground truth for task {task}")
    gt_priors = dict(r = dx.Gamma(k, theta),
                     phi = dx.Uniform(jnp.zeros(n_clusters), 2*jnp.pi*jnp.ones(n_clusters)),
                     sigma_beta =dx.Transformed(dx.Normal(mu_sigma, sigma_sigma), tfb.Sigmoid())
                    )

    cluster_model = ClusterModel(prior = gt_priors,
                                 N = N,
                                 hyperbolic = hyperbolic,
                                 prob_per_cluster = probability_per_cluster,
                                 min_cluster_dist = min_cluster_dist)
    key, subkey = jrnd.split(key)
    sampled_state = cluster_model.sample_from_prior(subkey, num_samples=1)

    ## Save ground truth distances
    gt_distances[task] = cluster_model.get_hyperbolic_distance(sampled_state.position) if hyperbolic else cluster_model.get_euclidean_distance(sampled_state.position)

    gt_positions = sampled_state.position[latpos]
    gt_noise_term = sampled_state.position['sigma_beta']

    filename = f"gt_{'hyp' if hyperbolic else 'euc'}_T{task}"

    if make_plot:
        cluster_colors = np.array([plt.get_cmap('cool')(i) for i in np.linspace(0, 1, n_clusters)])
        node_colors = cluster_colors[cluster_model.gt_cluster_index,:]

        cluster_means = cluster_model.cluster_means

        plt.figure()
        plt.scatter(cluster_means[:, 0], cluster_means[:, 1], c=cluster_colors, marker='*')
        plt.scatter(gt_positions[:, 0], gt_positions[:, 1], c=node_colors, s=5)
        savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"{filename}.png")
        plt.savefig(savetitle, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {savetitle}")

    ## Save latent positions
    ground_truth = {latpos:gt_positions,
                    'sigma_beta':gt_noise_term}

    ground_truth_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"{filename}.pkl")
    with open(ground_truth_filename, 'wb') as f:
        pickle.dump(ground_truth, f)

    ## Sample observations from the likelihood
    key, subkey = jrnd.split(key)
    observations = cluster_model.con_loglikelihood_fn(sampled_state).sample(seed=key, sample_shape=(n_observations))

    all_observations[task] = observations

## Save observations
observations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"observations_{'hyp' if hyperbolic else 'euc'}.pkl")
with open(observations_filename, 'wb') as f:
    pickle.dump(all_observations, f)

################################################################################################
###### Binarize the continuous observations by leaving at most 0.05% of nodes unconnected ######
################################################################################################

def cond(state:Tuple[int, Array], max_unconnected:int=0.05) -> bool:
    """
    Args:
        state: tuple containing th_idx and degree
            th_idx : index of the ordered threshold
            degree : (N,) degree distribution given the threshold
        max_unconnected: the maximum number of unconnected nodes

    Returns:
        Whether the number of unconnected nodes is less than max_unconnected
    """
    _, degree = state
    return jnp.sum(degree==0)/N < max_unconnected

def get_degree_dist(state:Tuple[int, Array], obs:Array, sorted_idc:Array) -> Tuple[int, Array]:
    """
    Args:
        state: tuple containing th_idx and degree
            th_idx : index of the ordered threshold
            degree : (N,) degree distribution given the threshold
        obs: (M,) continuous observation
        sorted_idc: (M,) indices of the sorted observation

    Returns:
        The threshold's index in the sorted list and the degree distribution of the binary network
    """
    th_idx, _ = state
    threshold = obs[sorted_idc[th_idx]]
    bin_obs = obs > threshold
    degree = jnp.sum(triu2mat(bin_obs), axis=1)
    return th_idx+1, degree

binary_observations = np.zeros((n_tasks, n_observations, M))
for task in range(n_tasks):
    for subject in range(n_observations):
        curr_obs = all_observations[task, subject]
        sorted_idc = jnp.argsort(all_observations[task, subject])
        get_degree_dist_fn = lambda state: get_degree_dist(state, obs=jnp.array(curr_obs), sorted_idc=sorted_idc)
        th_idx_plus_two, _ = jax.lax.while_loop(cond, get_degree_dist_fn, (0, jnp.ones(N)))

        threshold = curr_obs[sorted_idc[th_idx_plus_two - 2]]
        binary_observations[task, subject, :] = curr_obs > threshold

## Save observations
observations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"binary_observations_{'hyp' if hyperbolic else 'euc'}.pkl")
with open(observations_filename, 'wb') as f:
    pickle.dump(binary_observations, f)

######################################################################################################
###### Re-learn the latent positions, given the "wrong" (non-hierarchical) prior over positions ######
######################################################################################################

for task in range(n_tasks):
    for subject in range(n_observations):
        print(f"Performing inference on Task {task} for subject {subject}")
        ## Create new non-hierarchical LSM model
        nonh_prior = {latpos: dx.Normal(mu_z*jnp.ones((N-3, D)), sigma_z*jnp.ones((N-3, D))),
                      f"{latpos}b2x": dx.Normal(mu_z, sigma_z),
                      f"{latpos}b2y": dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()),
                      f"{latpos}b3x": dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()),
                      'sigma_beta': dx.Transformed(dx.Normal(mu_sigma, sigma_sigma), tfb.Sigmoid())}

        learned_model = LSM(nonh_prior, all_observations[task, subject])
        num_params = (N - 3) * D + 3 + 1  # regular nodes + Bookstein coordinates + sigma_beta_T
        rmh_parameters = dict(sigma=0.01 * jnp.eye(num_params))
        smc_parameters = dict(kernel=bjx.rmh,
                              kernel_parameters=rmh_parameters,
                              num_particles=num_particles,
                              num_mcmc_steps=num_mcmc_steps)
        key, smc_key = jrnd.split(key)
        start_time = time.time()
        posterior, n_iter, lml = learned_model.inference(smc_key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
        end_time = time.time()
        print(f"Embedded in {n_iter} iterations in {end_time-start_time} seconds to get a LML of {lml}.")

        flat_particles, _ = jax.tree_util.tree_flatten(posterior.particles)
        variable_names = list(posterior.particles.keys())
        reshaped_posterior = [{vname:flat_particles[i][j] for i, vname in enumerate(variable_names) if latpos in vname} for j in range(num_particles)]
        learned_distances = [learned_model.distance_func(p) for p in reshaped_posterior]
        print(f"Distances: {len(learned_distances)}x{learned_distances[0].shape}")

        if make_plot:
            p_idx = np.random.randint(num_particles)

            plt.figure()
            plt.scatter(gt_distances[task], learned_distances[p_idx], color='k', s=1)
            plt.xlabel('Ground truth distance')
            plt.ylabel('Learned distance')
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"gt_vs_learned_dist_{'hyp' if hyperbolic else 'euc'}_S{subject}_T{task}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

write_seed('seed.txt', key)