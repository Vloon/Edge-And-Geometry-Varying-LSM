import os

from helper_functions import get_cmd_params, set_GPU, read_seed, write_seed

arguments = [('-gpu', 'gpu', str, ''),
             ]

cmd_params = get_cmd_params(arguments)
gpu = cmd_params['gpu']
set_GPU(gpu)
print(f'Setting GPU to [{gpu}]')

import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_default_device", jax.devices()[0])
jax.config.update("jax_enable_x64", True)
import jax.random as jrnd
import jax.numpy as jnp
import blackjax as bjx
import distrax as dx
from typing import Union, Dict, Callable
from jaxtyping import PRNGKeyArray, Array, Float

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from models import LSM, GibbsState

Numeric = Union[int, float]

#
class ClusterModel(LSM):

    #
    def __init__(self,
                 prior: Dict,
                 N: int,
                 hyperbolic: bool = False,
                 prob_per_cluster: list[Float] = None, # Leave None to divide nodes (roughly) evenly
                 min_cluster_dist: Numeric = 0.,       # Minimum cluster distance, added to cluster means radial distance
                 observations: Array = None,           # Leave None to allow sampling from prior
                 hyperparams: Dict = dict(mu_z=0.,     # Mean of the _z Normal distribution
                                          sigma_z=0.25,  # Std of the _z Normal distribution
                                          eps=1e-5,    # Clipping value for p/mu & kappa.
                                          B=0.3,       # Bookstein distance
                                          )
                 ):
        self.N = N
        self.latpos = '_z' if hyperbolic else 'z'
        if not prob_per_cluster:
            prob_per_cluster = 1/n_clusters
        assert sum(prob_per_cluster) == 1., f"Probabilities per cluster must sum to 1 but instead sums to {sum(prob_per_cluster)}"
        self.prob_per_cluster = jnp.array(prob_per_cluster)
        self.min_cluster_dist = min_cluster_dist
        super().__init__(prior, observations, hyperparams)

    #
    def get_cluster_means(self, r:Array, phi: Array) -> Array:
        """
        Calculates the cluster means. For each node, its cluster mean is put in that node's position in mu_z.
        Also saves the cluster means as a field, to be used in plotting later.
        Args:
            r: (n_clusters,) radial coordinates
            phi: (n_clusters,) angular coordinates

        Returns:
            mu_z: (N, 2) means in cartesian coordinates
        """
        r_ = r + self.min_cluster_dist
        mu_x = jnp.cos(phi) * r_
        mu_y = jnp.sin(phi) * r_
        self.cluster_means = jnp.vstack([mu_x, mu_y]).T
        cluster_means_per_node = self.cluster_means[self.gt_cluster_index]
        return cluster_means_per_node

    #
    def init_fn(self, key: PRNGKeyArray, num_particles:int=1):
        ## Initialize r, phi, and sigma_beta_T
        key_prior, key_div, key_z = jrnd.split(key, 3)
        initial_state = super().init_fn(key_prior, num_particles)
        initial_position = initial_state.position
        r = initial_position['r']
        phi = initial_position['phi']

        n_clusters = r.shape[0]

        ## Divide nodes over clusters
        self.gt_cluster_index = jrnd.choice(key_div, n_clusters, shape=(self.N,), p=self.prob_per_cluster)

        ## Calculate cluster means
        cluster_means_per_node = self.get_cluster_means(r, phi)

        def sample_fun(key, means, variance):
            z_prior = dx.Normal(loc=means, scale=variance)
            return z_prior.sample(seed=key)

        if num_particles > 1:
            keys = jrnd.split(key_z, num_particles)
            initial_position[self.latpos] = jax.vmap(sample_fun, in_axes=(0, 0))(keys, cluster_means_per_node, self.hyperparameters['sigma_z'])
        else:
            initial_position[self.latpos] = jnp.squeeze(sample_fun(key_z, cluster_means_per_node, self.hyperparameters['sigma_z']))
        return GibbsState(position=initial_position)

    #
    def logprior_fn(self) -> Callable:
        """
        p(r, phi, z, sigma) = p(z | r, phi) p(r) p(phi) p(sigma)
        """
        #
        def logprior_fn_(state: GibbsState) -> Float:
            position = getattr(state, 'position', state)
            r = position['r']
            phi = position['phi']
            sigma_beta = position['sigma_beta']
            latpos = position[self.latpos]

            logpdf = 0
            logpdf += jnp.sum(self.param_priors['r'].log_prob(r))
            logpdf += jnp.sum(self.param_priors['phi'].log_prob(phi))
            logpdf += jnp.sum(self.param_priors['sigma_beta'].log_prob(sigma_beta))

            cluster_means_per_node = self.get_cluster_means(r, phi)
            latpos_prior = dx.Normal(loc=cluster_means_per_node, scale=self.hyperparameters['sigma_z'])
            logpdf += jnp.sum(latpos_prior.log_prob(latpos))
            return logpdf

        return logprior_fn_

    #

## Set variables
seed = read_seed('seed.txt')
key = jrnd.PRNGKey(seed)

N = 100 # Total number of nodes
D = 2 # Latent dimensions

n_tasks = 4

n_clusters =  3
probability_per_cluster = [0.5, 0.25, 0.25]
min_cluster_dist = 0. # Increase to make clusters more distant
sigmas =[1.]*n_clusters # Standard deviations for each cluster

k = [3.]*n_clusters # Shape parameter of the gamma distribution
theta = [1]*n_clusters # Scale parameter of the gamma distribution

mu_sigma = 0. # mean of the sigma distribution
sigma_sigma = 1. # standard deviation of the sigma distribution

hyperbolic = True
latpos = '_z' if hyperbolic else 'z'

priors = dict(
    r = dx.Gamma(k, theta),
    phi = dx.Uniform(jnp.zeros(n_clusters), 2*jnp.pi*jnp.ones(n_clusters)),
    sigma_beta =dx.Transformed(dx.Normal(mu_sigma, sigma_sigma), tfb.Sigmoid())
)

cluster_model = ClusterModel(prior=priors,
                             N=N,
                             hyperbolic=hyperbolic,
                             prob_per_cluster=probability_per_cluster,
                             min_cluster_dist=min_cluster_dist)
key, subkey = jrnd.split(key)
sampled_state = cluster_model.sample_from_prior(subkey, num_samples=1)

make_plot = True

if make_plot:
    gt_positions = sampled_state.position[latpos]

    cluster_colors = np.array([plt.get_cmap('cool')(i) for i in np.linspace(0, 1, n_clusters)])
    node_colors = cluster_colors[cluster_model.gt_cluster_index]

    cluster_means = cluster_model.cluster_means

    plt.figure()
    plt.scatter(cluster_means[:, 0], cluster_means[:, 1], c=cluster_colors, marker='*')
    plt.scatter(gt_positions[:, 0], gt_positions[:, 1], c=node_colors, s=5)
    savetitle = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"gt_{'hyp' if hyperbolic else 'euc'}.png")
    print(f"Saved figure to {savetitle}")
    plt.savefig(savetitle, bbox_inches='tight')
    plt.close()

#### TODO: save latent positions
# ## Save latent positions
# ground_truth = dict(latpos_varname=latent_positions,
#                     sigma_beta_T=sigma_beta_T)
#
# ground_truth_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"cluster_sim_gt_T{task}.pkl")
# with open(ground_truth_filename, 'wb') as f:
#     pickle.dump(ground_truth, f)


#### TODO: Sample observations from the likelihood


write_seed('seed.txt', key)