import os, sys

import distrax as dx
import jax
import jax.numpy as jnp

from typing import Callable, Dict
from jaxtyping import Array, Float

sys.path.insert(0, os.path.expanduser('~/bayesianmodels')) ## Location on the cluster, not local machine
from uicsmodels.bayesianmodels import BayesianModel, GibbsState

from helper_functions import parallel_transport, exponential_map, lorentz_distance, euclidean_distance,\
    distance_mapping, add_bookstein_anchors, hyp_pnt

class LSM(BayesianModel):
    """
    Definition of the latent space model with varying latent geometry and edge type.
    The distance and likelihood functions are chosen based on the values in the prior:
    If '_z' is in the prior, we use hyperbolic distances, else we take the Euclidean distances from 'z'. 
    If 'sigma_beta_T' is given, the continuous likelihood function is used.
    """
    def __init__(self,
                 prior: Dict,
                 observation : Array = None, # If left None, we can use it to sample observations from the model
                 hyperparameters:Dict = dict(mu_z=0., # Mean of the z/_z distribution
                                             sigma_z=1., # Std of the z/_z distribution
                                             eps=1e-5, # Clipping value for p/mu & kappa.
                                             B=0.3, # Bookstein distance
                                             )
                 ):
        self.observation = observation
        self.param_priors = prior
        self.hyperparameters = hyperparameters

    #
    def loglikelihood_fn(self) -> Callable:
        """
        Log-likelihood of the LSM. Checks whether the edges are binary or continuous, as well as whether the geometry is Euclidean or hyperbolic based on the prior parameters.
        """

        #
        def get_euclidean_distance(position: Dict) -> Array:
            z = position['z']
            zb2x = position['zb2x']
            zb2y = position['zb2y']
            zb3x = position['zb3x']

            ## Add Bookstein anchors to the positions
            z = add_bookstein_anchors(z, zb2x, zb2y, zb3x, self.hyperparameters['B'])

            N, D = z.shape
            triu_indices = jnp.triu_indices(N, k=1)
            d = euclidean_distance(z)[triu_indices]
            return d

        #
        def get_hyperbolic_distance(position: Dict) -> Array:
            _z = position['_z']
            _zb2x = position['_zb2x']
            _zb2y = position['_zb2y']
            _zb3x = position['_zb3x']

            ## Add Bookstein anchors to the positions
            _z = add_bookstein_anchors(_z, _zb2x, _zb2y, _zb3x, self.hyperparameters['B'])

            ## Get distances on hyperbolic plane
            N, D = _z.shape
            triu_indices = jnp.triu_indices(N, k=1)
            mu_0 = jnp.zeros((N, D + 1))
            mu_0 = mu_0.at[:, 0].set(1)

            mu_tilde = self.hyperparameters['mu_z'] * jnp.ones_like(_z)
            mu = hyp_pnt(mu_tilde)
            v = jnp.concatenate([jnp.zeros((N, 1)), _z], axis=1)
            u = parallel_transport(v, mu_0, mu)
            z = exponential_map(mu, u)

            d = lorentz_distance(z)[triu_indices]
            return d

        ## Define distance function based on prior parameters
        distance_func = get_hyperbolic_distance if '_z' in self.param_priors else get_euclidean_distance

        #
        def con_loglikelihood_fn(state: GibbsState) -> Float:
            position = getattr(state, 'position', state)
            d = distance_func(position)

            beta_noise = position['sigma_beta']
            eps = self.hyperparameters['eps']
            mu_beta = distance_mapping(d, eps)
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
            p = distance_mapping(d, self.hyperparameters['eps'])
            loglikelihood = dx.Bernoulli(probs=p).log_prob(self.observation).sum()
            return loglikelihood

        return con_loglikelihood_fn if 'sigma_beta'in self.param_priors else bin_loglikelihood_fn
    #
