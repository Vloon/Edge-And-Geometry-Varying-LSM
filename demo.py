import os, sys
from helper_functions import get_cmd_params, set_GPU

arguments = [('-gpu', 'gpu', str, ''),
             ]

cmd_params = get_cmd_params(arguments)
gpu = cmd_params['gpu']
set_GPU(gpu)
print(f'Setting GPU to [{gpu}]')

import numpy as np
import jax
jax.config.update("jax_default_device", jax.devices()[0])
jax.config.update("jax_enable_x64", True)
import jax.random as jrnd
import jax.numpy as jnp
import blackjax as bjx
import distrax as dx

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from jax.random import PRNGKey

from models import LSM
from helper_functions import read_seed, write_seed

mu_z = 0.
sigma_z = 1.
mu_sigma = 0.
sigma_sigma = 1.
B = 0.3 # Bookstein distance

N = 10
M = N*(N-1)//2
D = 2

## Read key
seed = read_seed('seed.txt')
key = jrnd.PRNGKey(seed)

## Generate fake observations
key, bin_key, con_key = jrnd.split(key, 3)

bin_obs = jrnd.bernoulli(bin_key, 0.2, shape=(M,))
con_obs = jrnd.beta(con_key, 0.5, 0.5, shape=(M,))

## binary euclidean model
bineuc_priors = dict(z=dx.Normal(mu_z*jnp.ones((N-3, D)), sigma_z*jnp.ones((N-3, D)) ), # Non-anchor nodes
                  zb2x=dx.Normal(mu_z, sigma_z), # 2nd Bookstein anchor x-coordinate
                  zb2y=dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()), # 2nd Bookstein anchor y-coordinate
                  zb3x=dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()) # 3rd Bookstein anchor (Bookstein dist not yet taken into account)
                  )

bineuc_LSM = LSM(bineuc_priors, bin_obs)
num_params = (N-3) * D + 3 # regular nodes + Bookstein coordinates
rmh_parameters = dict(sigma=0.01 * jnp.eye(num_params))
smc_parameters = dict(kernel=bjx.rmh,
                      kernel_parameters=rmh_parameters,
                      num_particles=1_000,
                      num_mcmc_steps=100)
key, smc_key = jrnd.split(key)
particles, n_iter, lml = bineuc_LSM.inference(smc_key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
print(f"Embedded binary euclidean in {n_iter} iterations to get a LML of {lml}.")

## binary hyperbolic model
binhyp_priors = dict(_z=dx.Normal(mu_z*jnp.ones((N-3, D)), sigma_z*jnp.ones((N-3, D)) ), # Non-anchor nodes
                  _zb2x=dx.Normal(mu_z, sigma_z), # 2nd Bookstein anchor x-coordinate
                  _zb2y=dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()), # 2nd Bookstein anchor y-coordinate
                  _zb3x=dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()) # 3rd Bookstein anchor (Bookstein dist not yet taken into account)
                  )

binhyp_LSM = LSM(binhyp_priors, bin_obs)
smc_parameters = dict(kernel=bjx.rmh,
                      kernel_parameters=rmh_parameters,
                      num_particles=1_000,
                      num_mcmc_steps=100)
key, smc_key = jrnd.split(key)
particles, n_iter, lml = bineuc_LSM.inference(smc_key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
print(f"Embedded binary hyperbolic in {n_iter} iterations to get a LML of {lml}.")

## Create continuous euclidean model
coneuc_priors = dict(z=dx.Normal(mu_z*jnp.ones((N-3, D)), sigma_z*jnp.ones((N-3, D)) ), # Non-anchor nodes
                  zb2x=dx.Normal(mu_z, sigma_z), # 2nd Bookstein anchor x-coordinate
                  zb2y=dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()), # 2nd Bookstein anchor y-coordinate
                  zb3x=dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()), # 3rd Bookstein anchor (Bookstein dist not yet taken into account)
                  sigma_beta=dx.Transformed(dx.Normal(mu_sigma, sigma_sigma), tfb.Sigmoid())
                  )

coneuc_LSM = LSM(coneuc_priors, con_obs)
num_params = (N-3) * D + 3 + 1 # regular nodes + Bookstein coordinates + sigma_beta_T
rmh_parameters = dict(sigma=0.01 * jnp.eye(num_params))
smc_parameters = dict(kernel=bjx.rmh,
                      kernel_parameters=rmh_parameters,
                      num_particles=1_000,
                      num_mcmc_steps=100)
key, smc_key = jrnd.split(key)
particles, n_iter, lml = coneuc_LSM.inference(smc_key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
print(f"Embedded continuous euclidean in {n_iter} iterations to get a LML of {lml}.")

## Create continuous hyperbolic model
conhyp_priors = dict(_z=dx.Normal(mu_z*jnp.ones((N-3, D)), sigma_z*jnp.ones((N-3, D)) ), # Non-anchor nodes
                  _zb2x=dx.Normal(mu_z, sigma_z), # 2nd Bookstein anchor x-coordinate
                  _zb2y=dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()), # 2nd Bookstein anchor y-coordinate
                  _zb3x=dx.Transformed(dx.Normal(mu_z, sigma_z), tfb.Exp()), # 3rd Bookstein anchor (Bookstein dist not yet taken into account)
                  sigma_beta=dx.Transformed(dx.Normal(mu_sigma, sigma_sigma), tfb.Sigmoid())
                  )

conhyp_LSM = LSM(conhyp_priors, con_obs)
smc_parameters = dict(kernel=bjx.rmh,
                      kernel_parameters=rmh_parameters,
                      num_particles=1_000,
                      num_mcmc_steps=100)
key, smc_key = jrnd.split(key)
particles, n_iter, lml = conhyp_LSM.inference(smc_key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
print(f"Embedded continuous euclidean in {n_iter} iterations to get a LML of {lml}.")

## Write key 
write_seed('seed.txt', key)