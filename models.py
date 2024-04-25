import os, sys

import distrax as dx
import jax
import jax.numpy as jnp
import jax.random as jrnd

from typing import Callable, Dict, Union, Optional
from jaxtyping import Array, Float, PRNGKeyArray

sys.path.insert(0, os.path.expanduser('~/bayesianmodels')) ## Location on the cluster, not local machine
from uicsmodels.bayesianmodels import BayesianModel, GibbsState

Numeric = Union[int, Float]

class LSM(BayesianModel):
    """
    Definition of the latent space model with varying latent geometry and edge type.
    The distance and likelihood functions are chosen based on the values in the prior:
    If '_z' is in the prior, we use hyperbolic distances, else we take the Euclidean distances from 'z'. If neither is in, we use the 'hyperbolic' parameter (used for hierarchical models where position distribution is not in the prior).
    If 'sigma_beta' is given, the continuous likelihood function is used.
    """
    def __init__(self,
                 prior : Dict,
                 observation : Optional[Array] = None, # If left None, we can use it to sample observations from the model
                 hyperbolic : Optional[bool] = False, #  Not used unless neither 'z' nor '_z' is in the prior
                 hyperparameters : Dict = dict(mu_z=0.,         # Mean of the _z distribution
                                               sigma_z=1.,      # Std of the _z distribution
                                               eps=1e-5,        # Clipping value for p/mu & kappa.
                                               obs_eps=1e-7,    # Clipping value for the continuous observations
                                               bkst=True,       # Whether the position is in Bookstein coordinates
                                               B=0.3,           # Bookstein distance
                                              )
                 ):
        if observation is not None and 'sigma_beta' in prior:
            ## Clip the continuous observations to deal with rounding to zero/one.
            observation = jnp.clip(observation, hyperparameters.get('obs_eps'), 1 - hyperparameters.get('obs_eps'))

        self.observation = observation
        self.param_priors = prior
        self.hyperparameters = hyperparameters
        ## Define distance function based on prior parameters. If there is no position distribution in the prior (in hierarchical models), use the hyperbolic paramter.
        if '_z' in prior or 'z' in prior:
            self.distance_func = self.get_hyperbolic_distance if '_z' in prior else self.get_euclidean_distance
            self.latpos = '_z'  if '_z' in prior else 'z'
        else:
            self.distance_func = self.get_hyperbolic_distance if hyperbolic else self.get_euclidean_distance
            self.latpos = '_z' if hyperbolic else 'z'

    ## Define a number of functions used in the LSM
    def add_bookstein_anchors(self, z: Array, zb2x: Float, zb2y: Float, zb3x: Float, B: Float = 0.3) -> Array:
        """
        Returns the Bookstein anchor coordinates for the first positions, in 2 dimensions.
        These Bookstein coordinates are restricted, meaning only 1 node is set, and the others are restricted and learned.
        Args:
            z: (N-3, D) regular nodes
            zb2x: x-coordinate of the 2nd Bookstein anchor
            zb2y: y_coordinate of the 2nd Bookstein anchor
            zb3x: x-coordinate of the 3rd Bookstein anchor, of which the y-value must be 0
            B: offset on the x-axis that the Bookstein anchors are put

        Returns:
            z: (N, D) positions including the Bookstein anchors
        """
        bookstein_anchors = jnp.zeros(shape=(3, 2))
        ## First positions at (-B,0)
        bookstein_anchors = bookstein_anchors.at[0, 0].set(-B)
        ## Second positions at (x,y)
        bookstein_anchors = bookstein_anchors.at[1, 0].set(zb2x)
        bookstein_anchors = bookstein_anchors.at[1, 1].set(zb2y)
        ## Third position at (x,0)
        bookstein_anchors = bookstein_anchors.at[2, 0].set(zb3x - B)
        zc = jnp.concatenate([bookstein_anchors, z])
        return zc

    #
    def get_latent_positions(self, position: GibbsState) -> Array:
        """
        Args:
            position: current position

        Returns:
            the _z/z positions, with Bookstein anchors if applicable
        """
        z = position.get(self.latpos)
        if self.hyperparameters.get('bkst'):
            ## Add Bookstein anchors to the positions
            zb2x = position.get(f"{self.latpos}b2x")
            zb2y = position.get(f"{self.latpos}b2y")
            zb3x = position.get(f"{self.latpos}b3x")
            z = self.add_bookstein_anchors(z, zb2x, zb2y, zb3x, self.hyperparameters['B'])
        return z
    #
    def hyp_pnt(self, X: Array) -> jnp.ndarray:
        """
        Creates [z,x,y] positions in hyperbolic space by projecting [x,y] positions onto hyperbolic plane directly by solving the equation defining the hyperbolic plane.
        Args:
            X: (N,D) array containing 2D points to be projected up onto the hyperbolic plane

        Returns:
            x_hyp: (N,D+1) positions on the hyperbolic plane
        """
        N, D = X.shape
        z = jnp.sqrt(jnp.sum(X ** 2, axis=1) + 1)
        x_hyp = jnp.zeros((N, D + 1))
        x_hyp = x_hyp.at[:, 0].set(z)
        x_hyp = x_hyp.at[:, 1:].set(X)
        return x_hyp

    def distance_mapping(self, d: Array, eps: float) -> Array:
        """
        Calculates the Bernoulli probability p or beta mean mu given the latent distances d.
        Args:
            d: (M) latent distances.
            eps: offset for calculating {p,mu}, to insure 0 < {p,mu} < 1

        Returns:
            p or mu, depending on use case
        """
        return jnp.clip(jnp.exp(-d), eps, 1 - eps)

    def euclidean_distance(self, z: Array) -> Array:
        """
        Calculates the Euclidean distance between all elements in z, calculated via the Gram matrix.
        Args:
            z: (N,D) latent positions

        Returns:
            d: (N,N) distance matrix
        """
        G = jnp.dot(z, z.T)
        g = jnp.diag(G)
        ones = jnp.ones_like(g)
        inside = jnp.maximum(jnp.outer(ones, g) + jnp.outer(g, ones) - 2 * G, 0)
        return jnp.sqrt(inside)

    def lorentzian(self, v: Array, u: Array, keepdims: bool = False) -> Array:
        """
        Calculates the Lorentzian prodcut of v and u, defined as :math:`-v_0*u_0 + \sum{i=1}^N v_i*u_i`
        Args:
            v: (N,D) vector
            u: (N,D) vector
            keepdims: whether to keep the same number of dimensions, or flatten the new array.

        Returns:
            :math:`\langle v,u \rangle_L`
        """
        signs = jnp.ones_like(v)
        signs = signs.at[:, 0].set(-1)
        return jnp.sum(v * u * signs, axis=1, keepdims=keepdims)

    def lorentz_distance(self, z: Array) -> Array:
        """
        Calculates the hyperbolic distance between all N points in z as an N x N matrix.
        Args:
            z : (N,D) points in hyperbolic space
        Returns:
            d: (N,N) distance matrix in hyperbolic space.
        """
        def arccosh(x: Array) -> Array:
            """Definition of the arccosh function"""
            x_clip = jnp.maximum(x, jnp.ones_like(x, dtype=jnp.float32))
            return jnp.log(x_clip + jnp.sqrt(x_clip ** 2 - 1))

        signs = jnp.ones_like(z)
        signs = signs.at[:, 0].set(-1)
        lor = jnp.dot(signs * z, z.T)
        ## Due to numerical instability, we can get NaN's on the diagonal, hence we force the diagonal to be zero.
        dist = arccosh(-lor)
        dist = dist.at[jnp.diag_indices_from(dist)].set(0)
        return dist

    def parallel_transport(self, v: Array, nu: Array, mu: Array) -> Array:
        """
        Parallel transports the points v sampled around nu to the tangent space of mu.
        Args:
            v: (N,D) points on tangent space of nu [points on distribution around nu]
            nu: (N,D) point in hyperbolic space [center to move from] (mu_0 in Fig 2 of Nagano et al. (2019))
            mu: (N,D) point in hyperbolic space [center to move to]
        Returns:
            u: (N,D) points in the tangent space of mu
        """
        alpha = -self.lorentzian(nu, mu, keepdims=True)
        u = v + self.lorentzian(mu - alpha * nu, v, keepdims=True) / (alpha + 1) * (nu + mu)
        return u

    def exponential_map(self, mu: Array, u: Array, eps: float = 1e-6) -> Array:
        """
        Maps the points u on the tangent space of mu onto the hyperolic plane
        Args:
            mu: (N,D)  Transported middle points
            u: (N,D) Points to be mapped onto hyperbolic space (after parallel transport)
            eps: minimum value for the norm of u

        Returns:
            z: (N,D) Positions on the hyperbolic plane
        """
        # Euclidean norm from mu_0 to v is the same as from mu to u is the same as the hyperbolic norm from mu to exp_mu(u), hence we can use the Euclidean norm of v.
        lor = self.lorentzian(u, u, keepdims=True)
        u_norm = jnp.sqrt(jnp.clip(lor, eps, lor))  # If eps is too small, u_norm gets rounded right back to zero and then we divide by zero
        return jnp.cosh(u_norm) * mu + jnp.sinh(u_norm) * u / u_norm

    def get_euclidean_distance(self, position: Dict) -> Array:
        """
        Args:
            position: dictionary containing the latent Euclidean positions

        Returns:
            The Euclidean distances
        """
        z = self.get_latent_positions(position)

        N, D = z.shape
        triu_indices = jnp.triu_indices(N, k=1)
        d = self.euclidean_distance(z)[triu_indices]
        return d

    #
    def get_hyperbolic_positions(self, _z: Array):
        """
        Maps the positions pre-hyperbolic projection onto the hyperbolic plane.
        Args:
            _z: (N,D) positions pre-hyperbolic projection

        Returns:
            z: (N,D+1) positions on the hyperbolic plane
        """
        N, D = _z.shape
        mu_0 = jnp.zeros((N, D + 1))
        mu_0 = mu_0.at[:, 0].set(1)

        mu_tilde = self.hyperparameters.get('mu_z') * jnp.ones_like(_z)
        mu = self.hyp_pnt(mu_tilde)
        v = jnp.concatenate([jnp.zeros((N, 1)), _z], axis=1)
        u = self.parallel_transport(v, mu_0, mu)
        z = self.exponential_map(mu, u)
        return z

    #
    def get_hyperbolic_distance(self, position: GibbsState) -> Array:
        ## Get distances on hyperbolic plane
        _z = self.get_latent_positions(position)
        N, D = _z.shape
        triu_indices = jnp.triu_indices(N, k=1)
        z = self.get_hyperbolic_positions(_z)
        d = self.lorentz_distance(z)[triu_indices]
        return d

    #
    def con_loglikelihood_fn(self, state: GibbsState) -> Float:
        position = getattr(state, 'position', state)
        d = self.distance_func(position)

        beta_noise = position.get('sigma_beta')
        eps = self.hyperparameters.get('eps')
        mu_beta = self.distance_mapping(d, eps)
        bound = jnp.sqrt(mu_beta * (1 - mu_beta))
        sigma_beta = beta_noise * bound

        kappa = jnp.maximum(mu_beta * (1 - mu_beta) / jnp.maximum(sigma_beta ** 2, eps) - 1, eps)
        a = mu_beta * kappa
        b = (1 - mu_beta) * kappa
        loglikelihood = dx.Beta(alpha=a, beta=b)
        return loglikelihood

    #
    def bin_loglikelihood_fn(self, state: GibbsState) -> Float:
        position = getattr(state, 'position', state)
        d = self.distance_func(position)
        p = self.distance_mapping(d, self.hyperparameters.get('eps'))
        loglikelihood = dx.Bernoulli(probs=p)
        return loglikelihood

    #
    def loglikelihood_fn(self) -> Callable:
        """
        Checks whether the edges are binary or continuous, as well as whether the geometry is Euclidean or hyperbolic based on the prior parameters.

        Returns:
            log_likelihood_fn: The appropriate log-likelihood function of the LSM
        """
        loglikelihood_distr = self.con_loglikelihood_fn if 'sigma_beta'in self.param_priors else self.bin_loglikelihood_fn
        loglikelihood_fn_ = lambda s: loglikelihood_distr(s).log_prob(self.observation).sum()
        return loglikelihood_fn_
    #

#
class ClusterModel(LSM):

    #
    def __init__(self,
                 prior: Dict,
                 N: int,
                 prob_per_cluster: list[Float],         # Leave None to divide nodes (roughly) evenly
                 hyperbolic: bool = False,
                 min_cluster_dist: Numeric = 0.,        # Minimum cluster distance, added to cluster means' radial distance
                 observations: Array = None,            # Leave None to allow sampling from prior
                 hyperparams: Dict = dict(mu_z=0.,      # Mean of the _z Normal distribution
                                          sigma_z=1.,   # Std of the _z Normal distribution
                                          eps=1e-5,     # Clipping value for p/mu & kappa.
                                          bkst=False,   # Whether the position is in Bookstein coordinates
                                          B=0.3,        # Bookstein distance
                                          )
                 ):
        self.N = N
        assert sum(prob_per_cluster) == 1., f"Probabilities per cluster must sum to 1 but instead sums to {sum(prob_per_cluster)}"
        self.prob_per_cluster = jnp.array(prob_per_cluster)
        self.min_cluster_dist = min_cluster_dist
        super().__init__(prior, observations, hyperbolic, hyperparams)

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
        r = initial_position.get('r')
        phi = initial_position.get('phi')

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
            initial_position[self.latpos] = jax.vmap(sample_fun, in_axes=(0, 0))(keys, cluster_means_per_node, self.hyperparameters.get('sigma_z'))
        else:
            initial_position[self.latpos] = jnp.squeeze(sample_fun(key_z, cluster_means_per_node, self.hyperparameters.get('sigma_z')))
        return GibbsState(position=initial_position)

    #
    def logprior_fn(self) -> Callable:
        """
        p(r, phi, z, sigma) = p(z | r, phi) p(r) p(phi) p(sigma)
        """
        #
        def logprior_fn_(state: GibbsState) -> Float:
            position = getattr(state, 'position', state)
            r = position.get('r')
            phi = position.get('phi')
            sigma_beta = position.get('sigma_beta')
            latpos = position.get(self.latpos)

            logpdf = 0
            logpdf += jnp.sum(self.param_priors.get('r').log_prob(r))
            logpdf += jnp.sum(self.param_priors.get('phi').log_prob(phi))
            logpdf += jnp.sum(self.param_priors.get('sigma_beta').log_prob(sigma_beta))

            cluster_means_per_node = self.get_cluster_means(r, phi)
            latpos_prior = dx.Normal(loc=cluster_means_per_node, scale=self.hyperparameters.get('sigma_z'))
            logpdf += jnp.sum(latpos_prior.log_prob(latpos))
            return logpdf

        return logprior_fn_
    #

#