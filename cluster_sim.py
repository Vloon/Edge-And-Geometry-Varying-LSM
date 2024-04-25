import os

from helper_functions import get_cmd_params, set_GPU, read_seed, write_seed, triu2mat

arguments = [('-gpu', 'gpu', str, ''),
             ('--plot', 'make_plot', bool),
             ('-k', 'k_', float, 1.5),
             ('-sigz', 'sigma_z', float, 1.),
             ('-mcd', 'min_cluster_dist', float, 0.),
             ('-nclusters', 'n_clusters', int, 3),
             ('-nsubjects', 'n_subjects', int, 3)
             ]

cmd_params = get_cmd_params(arguments)
gpu = cmd_params['gpu']
set_GPU(gpu)
print(f'Setting GPU to [{gpu}]')

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
from blackjax.smc.tempered import TemperedSMCState

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

N = 164 # Total number of nodes
M = N*(N-1)//2 # Total number of edges
D = 2 # Latent dimensions

max_D = 3.5 # Maximum latent distance allowed

n_tasks = 1
n_subjects = cmd_params.get('n_subjects') # / subjects?

make_plot = cmd_params.get('make_plot')

## Cluster prior parameters
k_ = cmd_params.get('k_')
theta_ = 2/3
sigma_z = cmd_params.get('sigma_z')
min_cluster_dist = cmd_params.get('min_cluster_dist')
n_clusters = cmd_params.get('n_clusters')

probability_per_cluster = [1/n_clusters]*n_clusters

k = [k_]*n_clusters # Shape parameter of the gamma distribution
theta = [theta_]*n_clusters # Scale parameter of the gamma distribution

hyperbolic = True
latpos = '_z' if hyperbolic else 'z'

## Inference parameters
mu_z = 0.
l_sigma_z = 1.

num_mcmc_steps = 100
num_particles = 1000

#####################################################################################
###### Sample a prior with a number of clusters in the position's ground truth ######
#####################################################################################

continuous_observations = np.zeros((n_tasks, n_subjects, M))
gt_distances = np.zeros((n_tasks, M))

for task in range(n_tasks):
    print(f"Sampling ground truth for task {task}")
    gt_priors = dict(r = dx.Gamma(k, theta),
                     phi = dx.Uniform(jnp.zeros(n_clusters), 2*jnp.pi*jnp.ones(n_clusters)),
                     sigma_beta = dx.Uniform(0., 1.)
                    )
    hyperparams = dict(mu_z=0.,  # Mean of the _z Normal distribution
                      sigma_z=sigma_z,  # Std of the _z Normal distribution
                      eps=1e-5,  # Clipping value for p/mu & kappa.
                      bkst=False,  # Whether the position is in Bookstein coordinates
                      B=0.3,  # Bookstein distance
                      )

    cluster_model = ClusterModel(prior = gt_priors,
                                 N = N,
                                 hyperbolic = hyperbolic,
                                 prob_per_cluster = probability_per_cluster,
                                 min_cluster_dist = min_cluster_dist,
                                 hyperparams = hyperparams)
    key, subkey = jrnd.split(key)
    sampled_state = cluster_model.sample_from_prior(subkey, num_samples=1)

    ## Scale the positions so that the maximum distance is max_D
    dists = cluster_model.distance_func(sampled_state.position)
    scale = np.max(cluster_model.distance_func(sampled_state.position)) / max_D
    sampled_state.position[latpos] /= scale

    ## Save ground truth distances
    gt_distances[task] = cluster_model.distance_func(sampled_state.position)

    gt_positions = sampled_state.position[latpos]
    gt_noise_term = sampled_state.position['sigma_beta']

    gt_filename = f"gt_k{k_}_sig{sigma_z}_{'hyp' if hyperbolic else 'euc'}_T{task}"

    if make_plot:
        cluster_colors = np.array([plt.get_cmap('cool')(i) for i in np.linspace(0, 1, n_clusters)])
        node_colors = cluster_colors[cluster_model.gt_cluster_index,:]

        cluster_means = cluster_model.cluster_means/scale

        plt.figure()
        plt.scatter(cluster_means[:, 0], cluster_means[:, 1], c=cluster_colors, marker='*')
        plt.scatter(gt_positions[:, 0], gt_positions[:, 1], c=node_colors, s=5)
        savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"{gt_filename}.png")
        plt.savefig(savetitle, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {savetitle}")

    ## Save latent positions
    ground_truth = {latpos:gt_positions,
                    'sigma_beta':gt_noise_term}

    ground_truth_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"{gt_filename}.pkl")
    with open(ground_truth_filename, 'wb') as f:
        pickle.dump(ground_truth, f)

    ## Sample observations from the likelihood
    key, subkey = jrnd.split(key)
    observations = cluster_model.con_loglikelihood_fn(sampled_state).sample(seed=key, sample_shape=(n_subjects))

    continuous_observations[task] = observations

    if make_plot:
        for subject in range(n_subjects):
            obs = triu2mat(observations[subject])
            plt.figure()
            plt.imshow(obs, cmap='OrRd', vmin=0, vmax=1)
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"con_obs_from_{gt_filename}_S{subject}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

## Save observations
observations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"con_observations_k{k_}_sig{sigma_z}_{'hyp' if hyperbolic else 'euc'}.pkl")
with open(observations_filename, 'wb') as f:
    pickle.dump(continuous_observations, f)

################################################################################################
###### Binarize the continuous observations by leaving at most 0.05% of nodes unconnected ######
################################################################################################

#
def cond(state:Tuple[int, Array], max_unconnected:Float=0.05) -> bool:
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
    return jnp.sum(degree==0)/N <= max_unconnected

#
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

binary_observations = np.zeros((n_tasks, n_subjects, M))
for task in range(n_tasks):
    for subject in range(n_subjects):
        curr_obs = continuous_observations[task, subject]
        sorted_idc = jnp.argsort(continuous_observations[task, subject])
        get_degree_dist_fn = lambda state: get_degree_dist(state, obs=jnp.array(curr_obs), sorted_idc=sorted_idc)
        th_idx_plus_two, _ = jax.lax.while_loop(cond, get_degree_dist_fn, (0, jnp.ones(N)))

        threshold = curr_obs[sorted_idc[th_idx_plus_two - 2]]
        binary_observations[task, subject, :] = curr_obs > threshold

    if make_plot:
        for subject in range(n_subjects):
            obs = triu2mat(binary_observations[task, subject])
            plt.figure()
            plt.imshow(obs, cmap='OrRd', vmin=0, vmax=1)
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"bin_obs_from_{gt_filename}_S{subject}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

## Save observations
observations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"bin_observations_k{k_}_sig{sigma_z}_{'hyp' if hyperbolic else 'euc'}.pkl")
with open(observations_filename, 'wb') as f:
    pickle.dump(binary_observations, f)

######################################################################################################
###### Re-learn the latent positions, given the "wrong" (non-hierarchical) prior over positions ######
######################################################################################################

#
def extract_from_trace(posterior: TemperedSMCState, model: LSM, hyperbolic: bool = hyperbolic) -> Tuple[Array, Array]:
    """
    Extracts the positions with Bookstein anchors and distances from the posterior trace.
    Args:
        posterior: The learned posterior from the adaptive tempered SMC algorithm.
        model: The LSM, which should contain a .distance_func field.
        hyperbolic: Whether the model is in hyperbolic geometry

    Returns:
        learned_positions: (num_particles, N, D) position posterior. If hyperbolic, then these are the true hyperbolic positions, post-hyperbolic projection.
        learned_distances: (num_particles, M) upper triangles of the learned distance matrix per particle.
    """
    flat_particles, _ = jax.tree_util.tree_flatten(posterior.particles)
    variable_names = list(posterior.particles.keys())
    reshaped_posterior = [{vname: flat_particles[i][j] for i, vname in enumerate(variable_names) if latpos in vname} for j in range(num_particles)]
    learned_positions = jnp.array([model.get_latent_positions(p) for p in reshaped_posterior])
    if hyperbolic:
        learned_positions = jnp.array([model.get_hyperbolic_positions(_z) for _z in learned_positions])
    learned_distances = jnp.array([model.distance_func(p) for p in reshaped_posterior])
    return learned_positions, learned_distances

#
def distance_correlations(gt_distances: Array, learned_distances: list, num_particles: int):
    """
    Calculates the correlations between the ground truth and each particle
    Args:
        gt_distances: upper triangle of the ground truth distance matrix.
        learned_distances: list of upper triangles of the learned distance matrix per particle.

    Returns:
        corrs: matrix containing the correlations between the ground truth and each particle
    """
    get_corr = lambda p, corrs: corrs.at[p].set(jnp.corrcoef(gt_distances, learned_distances[p])[0, 1])
    corrs = jax.lax.fori_loop(0, num_particles,
                              get_corr,
                              jnp.zeros(num_particles))
    return corrs

correlations = np.zeros((2, n_tasks, n_subjects, num_particles))

for task in range(n_tasks):
    for subject in range(n_subjects):
        print(f"Performing inference on Task {task} for subject {subject}")
        ## Create new non-hierarchical continuous LSM
        nonh_con_prior = {latpos: dx.Normal(mu_z*jnp.ones((N-3, D)), l_sigma_z*jnp.ones((N-3, D))),
                          f"{latpos}b2x": dx.Normal(mu_z, l_sigma_z),
                          f"{latpos}b2y": dx.Transformed(dx.Normal(mu_z, l_sigma_z), tfb.Exp()),
                          f"{latpos}b3x": dx.Transformed(dx.Normal(mu_z, l_sigma_z), tfb.Exp()),
                          'sigma_beta': dx.Uniform(0., 1.)}

        learned_model = LSM(nonh_con_prior, continuous_observations[task, subject])
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

        ## Get positions and distances per particle
        learned_positions, learned_distances = extract_from_trace(posterior, learned_model)
        correlations[0, task, subject, :] = distance_correlations(gt_distances[task], learned_distances, num_particles)

        ## Save posterior positions
        posterior_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"posterior_con_{'hyp' if hyperbolic else 'euc'}_k{k_}_sig{sigma_z}_S{subject}_T{task}.pkl")
        with open(posterior_filename, 'wb') as f:
            pickle.dump(learned_positions, f)

        if make_plot:
            p_idx = np.random.randint(num_particles)

            plt.figure()
            plt.scatter(gt_distances[task], learned_distances[p_idx], color='k', s=1)
            plt.xlabel('Ground truth distance')
            plt.ylabel('Learned distance')
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"gt_vs_learned_dist_con_{'hyp' if hyperbolic else 'euc'}_k{k_}_sig{sigma_z}_S{subject}_T{task}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

        ## Create new non-hierarchical binary LSM
        nonh_bin_prior = {latpos: dx.Normal(mu_z * jnp.ones((N - 3, D)), l_sigma_z * jnp.ones((N - 3, D))),
                          f"{latpos}b2x": dx.Normal(mu_z, l_sigma_z),
                          f"{latpos}b2y": dx.Transformed(dx.Normal(mu_z, l_sigma_z), tfb.Exp()),
                          f"{latpos}b3x": dx.Transformed(dx.Normal(mu_z, l_sigma_z), tfb.Exp())}

        learned_model = LSM(nonh_bin_prior, binary_observations[task, subject])
        num_params = (N - 3) * D + 3 # regular nodes + Bookstein coordinates
        rmh_parameters = dict(sigma=0.01 * jnp.eye(num_params))
        smc_parameters = dict(kernel=bjx.rmh,
                              kernel_parameters=rmh_parameters,
                              num_particles=num_particles,
                              num_mcmc_steps=num_mcmc_steps)
        key, smc_key = jrnd.split(key)
        start_time = time.time()
        posterior, n_iter, lml = learned_model.inference(smc_key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
        end_time = time.time()
        print(f"Embedded in {n_iter} iterations in {end_time - start_time} seconds to get a LML of {lml}.")

        ## Get correlation per particle
        learned_positions, learned_distances = extract_from_trace(posterior, learned_model)
        correlations[1, task, subject, :] = distance_correlations(gt_distances[task], learned_distances, num_particles)

        ## Save posterior
        posterior_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"posterior_bin_{'hyp' if hyperbolic else 'euc'}_k{k_}_sig{sigma_z}_S{subject}_T{task}.pkl")
        with open(posterior_filename, 'wb') as f:
            pickle.dump(learned_positions, f)

        if make_plot:
            p_idx = np.random.randint(num_particles)

            plt.figure()
            plt.scatter(gt_distances[task], learned_distances[p_idx], color='k', s=1)
            plt.xlabel('Ground truth distance')
            plt.ylabel('Learned distance')
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"gt_vs_learned_dist_bin_{'hyp' if hyperbolic else 'euc'}_k{k_}_sig{sigma_z}_S{subject}_T{task}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

correlations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"gt_vs_learned_dist_correlations_{'hyp' if hyperbolic else 'euc'}_k{k_}_sig{sigma_z}.pkl")
with open(correlations_filename, 'wb') as f:
    pickle.dump(correlations, f)

write_seed('seed.txt', key)