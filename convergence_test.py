import os

from helper_functions import get_cmd_params, set_GPU, read_seed, write_seed, triu2mat

arguments = [('-gpu', 'gpu', str, ''),
             ('-seed', 'seed', int, 1234),
             ('-algo', 'algorithm', str, 'smc'),
             ]

cmd_params = get_cmd_params(arguments)
gpu = cmd_params['gpu']
set_GPU(gpu)
print(f'Setting GPU to [{gpu}]')

import jax
jax.config.update("jax_default_device", jax.devices()[0])
jax.config.update("jax_enable_x64", True)
import jax.random as jrnd
from jax.random import PRNGKey
import jax.numpy as jnp

import time
import matplotlib.pyplot as plt

import distrax as dx
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from blackjax import rmh, nuts, window_adaptation
from blackjax.diagnostics import potential_scale_reduction

from typing import Dict, Callable, Tuple
from jaxtyping import Array, Float
from blackjax.smc.tempered import TemperedSMCState

from models import LSM, GibbsState, BayesianModel

## Global parameters
VERBOSE = True
CONVERGENCE_FILENAME = 'convergence_log.txt'
ALGO = cmd_params.get('algorithm')

#
def has_converged(samples, threshold:Float = 1.1, verbose:bool = VERBOSE) -> bool:
    """
    Check whether the samples have converged, i.e. the potential scale reduction for all samples is smaller than the threshold
    Args:
        samples: smc/mcmc samples
        threshold: potential scale reduction factor threshold
        verbose: whether to print

    Returns:
        Whether all samples have converged
    """
    R = jax.tree_map(potential_scale_reduction, samples)
    R_scores = jnp.array([jnp.all(var < threshold) for var in jax.tree_util.tree_leaves(R)])
    # this is not watertight - depending on the MCMC/SMC approach this might contain a logdensity term as well
    # however, those tend to be 'converged' anyway
    if verbose:
        print([var for var in jax.tree_util.tree_leaves(R)])
        print([var.shape for var in jax.tree_util.tree_leaves(R)])
        log_txt = f"{ALGO}: Maximum PSRF: {jnp.max(jnp.array([jnp.max(r) for r in jax.tree_util.tree_leaves(R)])):0.3f}"
        print(log_txt)
        with open(CONVERGENCE_FILENAME, 'a') as f:
            f.write(log_txt)
            f.write('\n')
    return jnp.all(R_scores)

#
def run_smc_batch(key: PRNGKey, model: BayesianModel, num_params: int, num_particles:int, num_mcmc: int, rmh_stepsize:Float=0.01) -> Tuple[TemperedSMCState, int, Float]:
    """
    Runs one batch of SMC
    Args:
        key: jax random key
        model: uics Bayesian model
        num_params: number of parameters in the model
        num_particles: number of particles
        num_mcmc: number of mcmc steps per smc iteration
        rmh_stepsize: random-walk step size

    Returns:
        The SMC posterior, number of iterations, and log-marginal likelihood
    """
    rmh_parameters = dict(sigma=rmh_stepsize*jnp.eye(num_params))
    smc_parameters = dict(kernel=rmh,
                          kernel_parameters=rmh_parameters,
                          num_particles=num_particles,
                          num_mcmc_steps=num_mcmc)

    samples, num_adapt, lml = model.inference(key, mode='mcmc-in-smc', sampling_parameters=smc_parameters)
    return samples, num_adapt, lml

#
def run_smc_to_convergence(key:PRNGKey, model:BayesianModel, num_mcmc:int, increment:int, num_params:int, num_particles:int, num_chains:int, psrf_threshold:Float = 1.1) -> Tuple[TemperedSMCState, int, list]:
    """
    Runs smc with increasingly many mcmc steps per iteration, until for all parameters the psrf < threshold
    Args:
        key: jax random key
        model: uics Bayesian model
        num_mcmc: initial number of mcmc steps
        increment: multiplication factor for the number of mcmc steps between each run.
        num_params: number of total parameters per particle
        num_particles: number of particles
        num_chains: number of chains (= re-initializations) run per check
        psrf_threshold: potential scale reduction factor threshold, to define convergence

    Returns:
        The smc samples, final number of mcmc steps, and times it took per chain
    """
    key, *subkeys = jrnd.split(key, num_chains + 1)
    time_in_attempt = time.time()
    samples, _, _ = jax.vmap(run_smc_batch, in_axes=(0, None, None, None, None))(jnp.array(subkeys), model, num_params, num_particles, num_mcmc)
    elapsed_in_attempt = time.time() - time_in_attempt
    times = [elapsed_in_attempt]

    while not has_converged(samples.particles, threshold=psrf_threshold, verbose=VERBOSE):
        key, *subkeys = jrnd.split(key, num_chains + 1)
        num_mcmc *= increment
        time_in_attempt = time.time()
        samples, _, _ = jax.vmap(run_smc_batch, in_axes=(0, None, None, None, None))(jnp.array(subkeys), model, num_params, num_particles, num_mcmc)
        elapsed_in_attempt = time.time() - time_in_attempt
        times.append(elapsed_in_attempt)

        if VERBOSE:
            log_txt = f"{ALGO}: iteration with {num_mcmc} MCMC steps took {elapsed_in_attempt} seconds to run."
            print(log_txt)
            with open(CONVERGENCE_FILENAME, 'a') as f:
                f.write(log_txt)
                f.write('\n')

    return samples, num_mcmc, times

#
def run_nuts_batch(key: PRNGKey, model: BayesianModel, num_mcmc: int, initial_state: GibbsState = None, nuts_parameters:Dict=None):
    mcmc_parameters = dict(kernel=nuts,
                           kernel_parameters=nuts_parameters,
                           num_samples=num_mcmc,
                           num_burn=0)
    if initial_state is not None:
        mcmc_parameters['initial_state'] = initial_state

    samples = model.inference(key, mode='mcmc', sampling_parameters=mcmc_parameters)
    return samples

#
def run_nuts_to_convergence(key, model, num_warmup, num_mcmc, increment, num_chains, burnin_factor=0.5, downsample_to=1000, psrf_threshold=1.1):
    def init_fn_body(key, model):
        return model.init_fn(key, num_particles=1).position

    logdensity = lambda state: model.loglikelihood_fn()(state) + model.logprior_fn()(state)
    warmup = window_adaptation(nuts, logdensity)

    key, *subkeys = jrnd.split(key, num_chains + 1)
    initial_states = jax.vmap(init_fn_body, in_axes=(0, None))(jnp.array(subkeys), model)

    key, *subkeys = jrnd.split(key, num_chains + 1)
    init_states_in_axes = jax.tree_map(lambda l: 0, initial_states)
    (warm_states, warm_parameters), _ = jax.vmap(warmup.run, in_axes=(0,
                                                                      init_states_in_axes,
                                                                      None))(jnp.array(subkeys),
                                                                             initial_states,
                                                                             num_warmup)
    print('WARM STATES:', warm_states)
    print('WARM PARAMS:', warm_parameters)
    warm_states_in_axes = jax.tree_map(lambda l: 0, warm_states)
    warm_parameters_in_axes = jax.tree_map(lambda l: 0, warm_parameters)
    key, *subkeys = jrnd.split(key, num_chains + 1)
    samples = jax.vmap(run_nuts_batch, in_axes=(0,
                                                None,
                                                None,
                                                warm_states_in_axes,
                                                warm_parameters_in_axes))(jnp.array(subkeys),
                                                                          model,
                                                                          num_mcmc,
                                                                          warm_states,
                                                                          warm_parameters)

    while not has_converged(samples, threshold=psrf_threshold, verbose=VERBOSE):
        key, *subkeys = jrnd.split(key, num_chains + 1)
        if VERBOSE:
            log_txt = f"{ALGO}: Some parameters have not converged, now sampling at {num_mcmc+increment} samples"
            print(log_txt)
            with open(CONVERGENCE_FILENAME, 'a') as f:
                f.write(log_txt)
                f.write('\n')
        final_states = jax.tree_map(lambda l: l[:, -1, ...], samples)
        final_states_in_axes = jax.tree_map(lambda l: 0, final_states)
        new_samples = jax.vmap(run_nuts_batch, in_axes=(0,
                                                        None,
                                                        None,
                                                        final_states_in_axes,
                                                        warm_parameters_in_axes))(jnp.array(subkeys),
                                                                                  model,
                                                                                  increment,
                                                                                  final_states,
                                                                                  warm_parameters)
        samples = jax.tree_map(lambda *v: jnp.hstack(v), samples, new_samples)
        num_mcmc += increment

    samples = samples.position
    samples = downsample(samples, burnin_factor=burnin_factor, downsample_to=downsample_to)
    return samples, num_mcmc

#
def boxplot_coefficients(filename, samples, ylabel, labels=None):
    mean = jnp.mean(samples, axis=0)
    ix = jnp.argsort(jnp.abs(mean))[::-1]

    if labels == None:
        labels = [r'$x_{{{:d}}}$'.format(i) for i in jnp.arange(p)[ix]]
    else:
        labels = [labels[i] for i in ix]

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.axhline(y=0.0, lw=0.5, color='k', ls='--')
    bp = ax.boxplot(samples[:, ix].T,
                    patch_artist=True,
                    labels=labels,
                    boxprops=dict(facecolor='#FDB97D',
                                  linewidth=0.5),
                    capprops=dict(linewidth=0.5),
                    medianprops=dict(color='k',
                                     linestyle='-',
                                     linewidth=0.5),
                    flierprops=dict(markeredgewidth=0.5))

    ax.tick_params(labelrotation=90)
    ax.set_xlabel('Predictors')
    ax.set_ylabel(ylabel)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
    plt.close()

key = jrnd.PRNGKey(cmd_params.get('seed'))

N = 162 # Total number of nodes
M = N*(N-1)//2 # Total number of edges
D = 2 # Latent dimensions

max_D = 11.5 # Maximum latent distance allowed

## Prior parameters
cluster_means = jnp.array([[0,0], [0,1], [1,0]])
n_clusters = len(cluster_means)
prob_per_cluster = [1/n_clusters]*n_clusters
sigma_z = 1
sigma_beta = 0.2

hyperbolic = False
latpos = '_z' if hyperbolic else 'z'

## Initialize model to sample a prior and from that prior sample an observation. We will learn this observation back
key, subkey = jrnd.split(key)
gt_cluster_index = jrnd.choice(subkey, n_clusters, shape=(N,), p=jnp.array(prob_per_cluster))
cluster_means_per_node = cluster_means[gt_cluster_index]

gt_priors = {latpos:dx.Normal(cluster_means_per_node, sigma_z*jnp.ones((N, D))),
            'sigma_beta':dx.Uniform(0., 1.)
             }
hyperparams = dict(mu_z=0.,  # Mean of the _z Normal distribution
                   eps=1e-5,  # Clipping value for p/mu & kappa.
                   bkst=False,  # Whether the position is in Bookstein coordinates
                   B=0.3,  # Bookstein distance
                   )
gt_model = LSM(gt_priors, hyperparameters=hyperparams)

key, prior_key, obs_key, test_key = jrnd.split(key, 4)
sampled_state = gt_model.sample_from_prior(prior_key, num_samples=1, max_distance=max_D)
sampled_state.position['sigma_beta'] = sigma_beta

## Also scale cluster means
scale = gt_model.scale
cluster_means /= scale

## Sample observation
observation = gt_model.con_loglikelihood_fn(sampled_state).sample(seed=obs_key, sample_shape=(1))

## Create new non-hierarchical continuous LSM. This is the one we test convergence for
nonh_con_prior = {latpos: dx.Normal(jnp.zeros((N-3, D)), jnp.ones((N-3, D))),
                  f"{latpos}b2x": dx.Normal(0, 1),
                  f"{latpos}b2y": dx.Transformed(dx.Normal(0, 1), tfb.Exp()),
                  f"{latpos}b3x": dx.Transformed(dx.Normal(0, 1), tfb.Exp()),
                  'sigma_beta': dx.Uniform(0., 1.)}

learned_model = LSM(nonh_con_prior, observation)

num_params = (N - 3) * D + 3 + 1
num_particles = 1_000
num_chains = 4
psrf_threshold = 1.1
burnin_factor = 0.5

## Convergence test
if ALGO == 'smc':
    print('Starting SMC')

    labels = None #??

    samples_smc, num_mcmc, times_smc = run_smc_to_convergence(test_key,
                                                              learned_model,
                                                              num_mcmc=1_000,
                                                              increment=2,
                                                              num_params=num_params,
                                                              num_particles=num_particles,
                                                              num_chains=num_chains,
                                                              psrf_threshold=psrf_threshold
                                                              )

    if VERBOSE:
        log_txt = f"Adaptive-tempered SMC done in {num_mcmc} steps per iteration ({times_smc[-1]:0.2f} seconds; {jnp.sum(jnp.array(times_smc)):0.2f} cumulative)"
        print(log_txt)
        with open(CONVERGENCE_FILENAME, 'a') as f:
            f.write(log_txt)
            f.write('\n')

    # boxplot_coefficients(f'figures/cluster_sim/convergence/smc_variable_selection_{latpos}.pdf', samples_smc.particles[latpos], latpos, labels)
    boxplot_coefficients(f'figures/cluster_sim/convergence/smc_variable_selection_lambda.pdf', samples_smc.particles['lam'], r'$\lambda$', labels)
elif ALGO == 'nuts':
    print('Starting NUTS')

    labels = None  # ??

    start_nuts = time.time()
    samples_nuts, num_mcmc = run_nuts_to_convergence(test_key,
                                                     learned_model,
                                                     num_warmup=1_000,
                                                     num_mcmc=1_000,
                                                     increment=1_000,
                                                     num_chains=num_chains,
                                                     burnin_factor=burnin_factor,
                                                     downsample_to=num_particles,
                                                     psrf_threshold=psrf_threshold)

    end_nuts = time.time()
    elapsed_nuts = end_nuts - start_nuts
    print(f'Adaptive-tuned NUTS MCMC done in {num_mcmc} steps ({elapsed_nuts:0.2f} seconds)')

    # boxplot_coefficients(f'figures/cluster_sim/convergence/smc_variable_selection_{latpos}.pdf', samples_smc.particles[latpos], latpos, labels)
    boxplot_coefficients(f'figures/cluster_sim/convergence/smc_variable_selection_lambda.pdf', samples_smc.particles['lam'], r'$\lambda$', labels)

