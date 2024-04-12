import os, argparse, re

import numpy as np
import jax
import jax.numpy as jnp

from typing import Dict
from jaxtyping import Array, Float, PRNGKeyArray

#
def set_GPU(gpu:str = '') -> None:
    """
    Sets the GPU safely in os.environ
    Args:
        gpu: string format of the GPU used. Multiple GPUs can be seperated with commas, e.g. '0,1,2'.
    """
    if gpu is None: # If gpu is None, then all GPUs are used.
        gpu = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

#
def get_cmd_params(parameter_list:list) -> Dict:
    """
    Gets the parameters described in parameter_list from the command line.
    Args:
        parameter_list: list of tuples containing (arg_name <str>, dest <str>, type <type>, default <any> [*OPTIONAL], nargs <str> [OPTIONAL])
            - arg_name is the name of the argument in the command line.
            - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
            - type is the data-type of the variable.
            - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
            - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.

            Example of a valid parameter list:
                [('-m', 'mu', float, [1.,0.] '+'),
                ('-s', 'sigma', float, 1.)]

    Returns:
        cmd_params: dictionary containing the variables with their value, either the value defined in the command line or the default value.
    """
    ## Create parser
    parser = argparse.ArgumentParser()
    ## Add parameters to parser
    for parameter in parameter_list:
        assert len(parameter) in [3, 4, 5], f'Parameter tuple must be length 3 (bool only), 4 or 5 but is length {len(parameter)}.'
        if len(parameter) == 3:
            arg_name, dest, arg_type = parameter
            assert arg_type == bool, f'Three parameters were passed, so arg_type should be bool but is {arg_type}'
            nargs = '?'
        elif len(parameter) == 4:
            arg_name, dest, arg_type, default = parameter
            nargs = '?' # If no nargs is given we default to single value
        elif len(parameter) == 5:
            arg_name, dest, arg_type, default, nargs = parameter
        if arg_type != bool:
            parser.add_argument(arg_name, dest=dest, nargs=nargs, type=arg_type, default=default)
        else:
            parser.add_argument(arg_name, dest=dest, action='store_true', default=False)
    ## Parse arguments from CMD
    args = parser.parse_args()
    ## Create global parameters dictionary
    cmd_params = {arg:getattr(args,arg) for arg in vars(args)}
    return cmd_params

#
def write_seed(seed_file:str, key:PRNGKeyArray) -> None:
    """
    Writes the seed to the seed file
    Args:
        seed_file: file location of the seed file
        key: JAX random key
    """
    old_seed = read_seed(seed_file)
    with open(seed_file, 'w') as f:
        keystr = re.findall(r"[0-9]+", str(key))[-1]
        f.write(f"{keystr}\n{old_seed}")

#
def read_seed(seed_file:str) -> int:
    """
    Reads the seed from the seed file. The seed file consists of a line with the new seed and then a line with the old seed.
    Args:
        seed_file: file location of the seed file

    Returns:
        seed: the seed to be used for PRNGs
    """
    with open(seed_file, 'r') as f:
        next_seed = int(f.read().split('\n')[0])
    return next_seed

#
def triu2mat(v:Array) -> Array:
    """
    Fills a matrix from the upper triangle vector
    Args:
        v: (M,) upper triangle vector

    Returns:
        mat: (N,N) matrix
    """
    m = len(v)
    n = int((1 + np.sqrt(1 + 8 * m))/2)
    mat = jnp.zeros((n, n))
    triu_indices = jnp.triu_indices(n, k=1)
    mat = mat.at[triu_indices].set(v)
    return mat + mat.T

#
def lorentz_to_poincare(networks:Array) -> Array:
    """
    Convert Lorentz coordinates to Poincaré coordinates, eq. 11 from Nickel & Kiela (2018).
    Args:
        networks: (S,N,D) or (N,D) positions in Lorentzian coordinates

    Returns:
        networks: (S,N,D-1) or (N,D-1) positions in Poincaré coordinates
    """
    ## Assure networks has the correct shape
    one_nw = len(networks.shape) == 2
    if one_nw:
        networks = np.array([networks])

    ## Convert Lorentz positions to Poincaré positions
    S, N, D_L = networks.shape
    # calc_z_P = lambda i, nw_tpl: nw_tpl[1].at[i,:,:].set(nw_tpl[0][i,:,1:]/jnp.reshape(jnp.repeat(nw_tpl[0][i,:,0]+1, D_L-1), newshape=(N, D_L-1)))
    #
    # _, z_P = jax.lax.fori_loop(0, S,
    #                            calc_z_P,
    #                            (networks, jnp.zeros((S, N, D_L-1)))
    #                            )
    calc_z_P = lambda c, nw: (None, nw[:,1:]/jnp.reshape(jnp.repeat(nw[:,0]+1, D_L-1), newshape=(N, D_L-1)))
    _, z_P = jax.lax.scan(calc_z_P, None, networks)

    return z_P[0] if one_nw else z_P