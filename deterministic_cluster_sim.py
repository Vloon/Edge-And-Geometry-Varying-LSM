from helper_functions import get_cmd_params, set_GPU, triu2mat

arguments = [('-gpu', 'gpu', str, ''),
             ('-seed', 'seed', int, 1234),
             ('-sigz', 'sigma_z', float, 1.),
             ('-numgts', 'num_ground_truths', int, 1),
             ('-numsamples', 'num_samples', int, 1),
             ('-sigb', 'sigma_beta', float, 0.3),
             ('--plot', 'make_plot', bool),
             ]

cmd_params = get_cmd_params(arguments)
gpu = cmd_params.get('gpu')
set_GPU(gpu)
print(f'Setting GPU to [{gpu}]')

import jax
jax.config.update("jax_default_device", jax.devices()[0])
jax.config.update("jax_enable_x64", True)
import jax.random as jrnd
import jax.numpy as jnp
import blackjax as bjx
import distrax as dx

import matplotlib.pyplot as plt
import os, pickle, time

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from typing import Dict, Callable, Tuple
from jaxtyping import Array, Float
from blackjax.smc.tempered import TemperedSMCState

from models import LSM, GibbsState

make_plot = cmd_params.get('make_plot')
hyperbolic = True
latpos = '_z' if hyperbolic else 'z'

sigma_z = cmd_params.get('sigma_z')
max_D = 3.5

N = 162
M = N*(N-1)//2
D = 2

cluster_means = jnp.array([[0,0], [0,1], [1,0]])
n_clusters = len(cluster_means)
prob_per_cluster = [1/n_clusters]*n_clusters
sigma_beta = cmd_params.get('sigma_beta')

num_ground_truths = cmd_params.get('num_ground_truths')
num_samples = cmd_params.get('num_samples')

key = jrnd.PRNGKey(cmd_params.get('seed'))

#####################################################################################
###### Sample a prior with a number of clusters in the position's ground truth ######
#####################################################################################

continuous_observations = jnp.zeros((num_ground_truths, num_samples, M))
gt_distances = jnp.zeros((num_ground_truths, M))

for n_gt in range(num_ground_truths):
    key, div_key, prior_key, obs_key = jrnd.split(key, 4)
    ## Sample ground truth from the prior
    gt_cluster_index = jrnd.choice(div_key, n_clusters, shape=(N,), p=jnp.array(prob_per_cluster))
    cluster_means_per_node = cluster_means[gt_cluster_index]

    gt_priors = {latpos: dx.Normal(cluster_means_per_node, sigma_z*jnp.ones((N, D))),
                     'sigma_beta': dx.Uniform(0., 1.),
                    }
    hyperparams = dict(mu_z=0.,  # Mean of the z distribution in hyperbolic space (to-position in parellel transport).
                       eps=1e-5,  # Clipping value for p/mu & kappa.
                       bkst=False,  # Whether the position is in Bookstein coordinates
                       B=0.3,  # Bookstein distance
                       )
    gt_model = LSM(gt_priors, hyperparameters=hyperparams)
    gt_state = gt_model.sample_from_prior(prior_key, num_samples=1, max_distance=max_D)

    gt_positions = gt_state.position[latpos]
    gt_distances = gt_distances.at[n_gt].set(gt_model.distance_func(gt_state.position))

    gt_filename = f"gt_{n_gt}_possig{sigma_z}_edgesig{sigma_beta}_{'hyp' if hyperbolic else 'euc'}"

    if make_plot:
        cluster_colors = jnp.array([plt.get_cmap('cool')(i) for i in jnp.linspace(0, 1, n_clusters)])
        node_colors = cluster_colors[gt_cluster_index, :]

        cluster_means /= gt_model.scale
        filename = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"{gt_filename}.png")
        plt.figure()
        plt.scatter(cluster_means[:, 0], cluster_means[:, 1], c=cluster_colors, marker='*')
        plt.scatter(gt_positions[:, 0], gt_positions[:, 1], c=node_colors, s=5)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {filename}")

    ## Save ground truth latent variables and cluster information
    ground_truth = {latpos:gt_positions,
                    'sigma_beta':sigma_beta}
    ground_truth_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"{gt_filename}.pkl")
    with open(ground_truth_filename, 'wb') as f:
        pickle.dump(ground_truth, f)
    gt_cluster_info = dict(gt_cluster_index=gt_cluster_index,
                           gt_cluster_means=cluster_means)
    gt_cluster_info_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"{gt_filename}_cluster_info.pkl")
    with open(gt_cluster_info_filename, 'wb') as f:
        pickle.dump(gt_cluster_info, f)

    ## Sample observations from the likelihood
    observations = gt_model.con_loglikelihood_fn(gt_state).sample(seed=obs_key, sample_shape=(num_samples))
    continuous_observations = continuous_observations.at[n_gt].set(observations)

    if make_plot:
        for sample in range(num_samples):
            obs = triu2mat(observations[sample])
            plt.figure()
            plt.imshow(obs, cmap='OrRd', vmin=0, vmax=1)
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"con_obs_from_{gt_filename}_S{sample}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

## Save observations
observations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"con_observations_sig{sigma_z}_{'hyp' if hyperbolic else 'euc'}.pkl")
with open(observations_filename, 'wb') as f:
    pickle.dump(continuous_observations, f)

#################################################################################################
###### Binarize the continuous observations by leaving at most some % of nodes unconnected ######
#################################################################################################

#
def cond(state:Tuple[int, Array], max_unconnected:Float) -> bool:
    """
    Args:
        state:
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
        state:
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

binary_observations = jnp.zeros((num_ground_truths, num_samples, M))
for n_gt in range(num_ground_truths):
    for sample in range(num_samples):
        curr_obs = continuous_observations[n_gt, sample]
        sorted_idc = jnp.argsort(continuous_observations[n_gt, sample])
        cond_ = lambda state: cond(state, max_unconnected=0.02)
        get_degree_dist_ = lambda state: get_degree_dist(state, obs=jnp.array(curr_obs), sorted_idc=sorted_idc)
        th_idx_plus_two, _ = jax.lax.while_loop(cond_, get_degree_dist_, (0, jnp.ones(N)))

        threshold = curr_obs[sorted_idc[th_idx_plus_two - 2]]
        binary_observations = binary_observations.at[n_gt, sample, :].set(curr_obs > threshold)

    if make_plot:
        for sample in range(num_samples):
            obs = triu2mat(binary_observations[n_gt, sample])
            plt.figure()
            plt.imshow(obs, cmap='OrRd', vmin=0, vmax=1)
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"bin_obs_from_{gt_filename}_S{sample}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

## Save observations
observations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"bin_observations_sig{sigma_z}_{'hyp' if hyperbolic else 'euc'}.pkl")
with open(observations_filename, 'wb') as f:
    pickle.dump(binary_observations, f)

###################################################################################################
###### Re-learn the latent positions, given the "wrong" (non-clustered) prior over positions ######
###################################################################################################

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

num_mcmc_steps = 100
num_particles = 1_000

correlations = jnp.zeros((2, num_ground_truths, num_samples, num_particles))

for n_gt in range(num_ground_truths):
    for sample in range(num_samples):
        print(f"Performing inference on ground truth {n_gt}, sample {sample}")
        ## Create new non-hierarchical continuous LSM
        nonh_con_prior = {latpos: dx.Normal(jnp.zeros((N-3, D)), jnp.ones((N-3, D))),
                          f"{latpos}b2x": dx.Normal(0, 1),
                          f"{latpos}b2y": dx.Transformed(dx.Normal(0, 1), tfb.Exp()),
                          f"{latpos}b3x": dx.Transformed(dx.Normal(0, 1), tfb.Exp()),
                          'sigma_beta': dx.Uniform(0., 1.)}

        learned_model = LSM(nonh_con_prior, continuous_observations[n_gt, sample])
        num_params = (N - 3) * D + 3 + 1  # regular nodes + Bookstein coordinates + sigma_beta_T
        rmh_parameters = dict(sigma=0.01 * jnp.eye(num_params))
        smc_parameters = dict(kernel=bjx.rmh,
                              kernel_parameters=rmh_parameters,
                              num_particles=num_particles,
                              num_mcmc_steps=num_mcmc_steps)
        key, smc_key, plot_key = jrnd.split(key, 3)
        start_time = time.time()
        posterior, n_iter, lml = learned_model.inference(smc_key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
        end_time = time.time()
        print(f"Embedded continuous observation in {n_iter} iterations in {end_time-start_time} seconds to get a LML of {lml}.")

        ## Get positions and distances per particle
        learned_positions, learned_distances = extract_from_trace(posterior, learned_model)
        correlations = correlations.at[0, n_gt, sample, :].set(distance_correlations(gt_distances[n_gt], learned_distances, num_particles))

        ## Save posterior positions
        posterior_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"posterior_con_{'hyp' if hyperbolic else 'euc'}_possig{sigma_z}_edgesig{sigma_beta}_gt{n_gt}_S{sample}.pkl")
        with open(posterior_filename, 'wb') as f:
            pickle.dump(learned_positions, f)

        if make_plot:
            p_idx = jrnd.randint(plot_key, shape=(1,), minval=0, maxval=num_particles)

            plt.figure()
            plt.scatter(gt_distances[n_gt], learned_distances[p_idx], color='k', s=1)
            plt.xlabel('Ground truth distance')
            plt.ylabel('Learned distance')
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"gt_vs_learned_dist_con_{'hyp' if hyperbolic else 'euc'}_possig{sigma_z}_edgesig{sigma_beta}_gt{n_gt}_S{sample}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

        ## Create new non-hierarchical binary LSM
        nonh_bin_prior = {latpos: dx.Normal(jnp.zeros((N - 3, D)), jnp.ones((N - 3, D))),
                          f"{latpos}b2x": dx.Normal(0, 1),
                          f"{latpos}b2y": dx.Transformed(dx.Normal(0, 1), tfb.Exp()),
                          f"{latpos}b3x": dx.Transformed(dx.Normal(0, 1), tfb.Exp())}

        learned_model = LSM(nonh_bin_prior, binary_observations[n_gt, sample])
        num_params = (N - 3) * D + 3 # regular nodes + Bookstein coordinates
        rmh_parameters = dict(sigma=0.01 * jnp.eye(num_params))
        smc_parameters = dict(kernel=bjx.rmh,
                              kernel_parameters=rmh_parameters,
                              num_particles=num_particles,
                              num_mcmc_steps=num_mcmc_steps)
        key, smc_key, plot_key = jrnd.split(key, 3)
        start_time = time.time()
        posterior, n_iter, lml = learned_model.inference(smc_key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
        end_time = time.time()
        print(f"Embedded binary observation in {n_iter} iterations in {end_time - start_time} seconds to get a LML of {lml}.")

        ## Get correlation per particle
        learned_positions, learned_distances = extract_from_trace(posterior, learned_model)
        correlations = correlations.at[1, n_gt, sample, :].set(distance_correlations(gt_distances[n_gt], learned_distances, num_particles))

        ## Save posterior
        posterior_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"posterior_bin_{'hyp' if hyperbolic else 'euc'}_possig{sigma_z}_edgesig{sigma_beta}_gt{n_gt}_S{sample}.pkl")
        with open(posterior_filename, 'wb') as f:
            pickle.dump(learned_positions, f)

        if make_plot:
            p_idx = jrnd.randint(plot_key, shape=(1,), minval=0, maxval=num_particles)

            plt.figure()
            plt.scatter(gt_distances[n_gt], learned_distances[p_idx], color='k', s=1)
            plt.xlabel('Ground truth distance')
            plt.ylabel('Learned distance')
            savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"gt_vs_learned_dist_bin_{'hyp' if hyperbolic else 'euc'}_possig{sigma_z}_edgesig{sigma_beta}_gt{n_gt}_S{sample}.png")
            plt.savefig(savetitle, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {savetitle}")

correlations_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"gt_vs_learned_dist_correlations_{'hyp' if hyperbolic else 'euc'}_possig{sigma_z}_edgesig{sigma_beta}.pkl")
with open(correlations_filename, 'wb') as f:
    pickle.dump(correlations, f)
