import os, sys

import distrax as dx
import jax
import jax.numpy as jnp

from typing import Callable, Dict
from jaxtyping import Array, Float

sys.path.insert(0, os.path.expanduser('~/bayesianmodels')) ## Location on the cluster, not local machine
from uicsmodels.bayesianmodels import BayesianModel, GibbsState

class LSM(BayesianModel):
    """
    Definition of the latent space model with varying latent geometry and edge type.
    The distance and likelihood functions are chosen based on the values in the prior:
    If '_z' is in the prior, we use hyperbolic distances, else we take the Euclidean distances from 'z'. 
    If 'sigma_beta_T' is given, the continuous likelihood function is used.
    """
    def __init__(self,
                 prior : Dict,
                 observation : Array = None, # If left None, we can use it to sample observations from the model
                 hyperparameters : Dict = dict(mu_z=0., # Mean of the z/_z distribution
                                             sigma_z=1., # Std of the z/_z distribution
                                             eps=1e-5, # Clipping value for p/mu & kappa.
                                             B=0.3, # Bookstein distance
                                             )
                 ):
        self.observation = observation
        self.param_priors = prior
        self.hyperparameters = hyperparameters

    ### Define a number of functions used in the LSM
    def add_bookstein_anchors(self, _z, _zb2x: Float, _zb2y: Float, _zb3x: Float, B: Float = 0.3) -> Array:
        """
        Returns the Bookstein anchor coordinates for the first positions, in 2 dimensions.
        These Bookstein coordinates are restricted, meaning only 1 node is set, and the others are restricted and learned.
        Args:
            _z: (N-3, D) regular nodes
            _zb2x: x-coordinate of the 2nd Bookstein anchor
            _zb2y: y_coordinate of the 2nd Bookstein anchor
            _zb3x: x-coordinate of the 3rd Bookstein anchor, of which the y-value must be 0
            B: offset on the x-axis that the Bookstein anchors are put

        Returns:
            _z: (N, D) positions including the Bookstein anchors
        """
        bookstein_anchors = jnp.zeros(shape=(3, 2))
        ## First positions at (-B,0)
        bookstein_anchors = bookstein_anchors.at[0, 0].set(-B)
        ## Second positions at (x,y)
        bookstein_anchors = bookstein_anchors.at[1, 0].set(_zb2x)
        bookstein_anchors = bookstein_anchors.at[1, 1].set(_zb2y)
        ## Third position at (x,0)
        bookstein_anchors = bookstein_anchors.at[2, 0].set(_zb3x - B)
        _zc = jnp.concatenate([bookstein_anchors, _z])
        return _zc

    def hyp_pnt(self, X: Array) -> jnp.ndarray:
        """
        Creates [z,x,y] positions in hyperbolic space by projecting [x,y] positions onto hyperbolic plane directly by solving the equation defining the hyperbolic plane.
        Args:
            X: (N,D) array containing 2D points to be projected up onto the hyperbolic plane

        Returns:
            _X: (N,D+1) positions on the hyperbolic plane
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
        return jnp.clip(jnp.exp(-d ** 2), eps, 1 - eps)

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
            :math:`\langle v,u \rangle`
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

    #
    def loglikelihood_fn(self) -> Callable:
        """
        Checks whether the edges are binary or continuous, as well as whether the geometry is Euclidean or hyperbolic based on the prior parameters.

        Returns:
            log_likelihood_fn: The appropriate log-likelihood function of the LSM
        """

        #
        def get_euclidean_distance(position: Dict) -> Array:
            """
            Args:
                position: dictionary containing the latent Euclidean positions

            Returns:
                The Euclidean distances
            """
            z = position['z']
            zb2x = position['zb2x']
            zb2y = position['zb2y']
            zb3x = position['zb3x']

            ## Add Bookstein anchors to the positions
            z = self.add_bookstein_anchors(z, zb2x, zb2y, zb3x, self.hyperparameters['B'])

            N, D = z.shape
            triu_indices = jnp.triu_indices(N, k=1)
            d = self.euclidean_distance(z)[triu_indices]
            return d

        #
        def get_hyperbolic_distance(position: Dict) -> Array:
            _z = position['_z']
            _zb2x = position['_zb2x']
            _zb2y = position['_zb2y']
            _zb3x = position['_zb3x']

            ## Add Bookstein anchors to the positions
            _z = self.add_bookstein_anchors(_z, _zb2x, _zb2y, _zb3x, self.hyperparameters['B'])

            ## Get distances on hyperbolic plane
            N, D = _z.shape
            triu_indices = jnp.triu_indices(N, k=1)
            mu_0 = jnp.zeros((N, D + 1))
            mu_0 = mu_0.at[:, 0].set(1)

            mu_tilde = self.hyperparameters['mu_z'] * jnp.ones_like(_z)
            mu = self.hyp_pnt(mu_tilde)
            v = jnp.concatenate([jnp.zeros((N, 1)), _z], axis=1)
            u = self.parallel_transport(v, mu_0, mu)
            z = self.exponential_map(mu, u)

            d = self.lorentz_distance(z)[triu_indices]
            return d

        ## Define distance function based on prior parameters
        distance_func = get_hyperbolic_distance if '_z' in self.param_priors else get_euclidean_distance

        #
        def con_loglikelihood_fn(state: GibbsState) -> Float:
            position = getattr(state, 'position', state)
            d = distance_func(position)

            beta_noise = position['sigma_beta']
            eps = self.hyperparameters['eps']
            mu_beta = self.distance_mapping(d, eps)
            bound = jnp.sqrt(mu_beta * (1 - mu_beta))
            sigma_beta = beta_noise * bound

            kappa = jnp.maximum(mu_beta * (1 - mu_beta) / jnp.maximum(sigma_beta ** 2, eps) - 1, eps)
            a = mu_beta * kappa
            b = (1 - mu_beta) * kappa
            loglikelihood = dx.Beta(alpha = a, beta = b).log_prob(self.observation).sum()
            return loglikelihood

        #
        def bin_loglikelihood_fn(state:GibbsState) -> Float:
            position = getattr(state, 'position', state)
            d = distance_func(position)
            p = self.distance_mapping(d, self.hyperparameters['eps'])
            loglikelihood = dx.Bernoulli(probs=p).log_prob(self.observation).sum()
            return loglikelihood

        return con_loglikelihood_fn if 'sigma_beta'in self.param_priors else bin_loglikelihood_fn
    #
