import os, argparse

import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, PRNGKeyArray, Dict

def set_GPU(gpu:str = '') -> None:
    """
    Sets the GPU safely in os.environ, and initializes JAX
    PARAMS:
    gpu : string format of the GPU used. Multiple GPUs can be seperated with commas, e.g. '0,1,2'.
    """
    if gpu is None: # If gpu is None, then all GPUs are used.
        gpu = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
def get_cmd_params(parameter_list:list) -> Dict:
    """
    Gets the parameters described in parameter_list from the command line.
    Returns a dictionary containing the variables with their value, either the value defined in the command line or the default value.
    PARAMS:
    parameter_list : list of tuples containing (arg_name <str>, dest <str>, type <type>, default <any> [*OPTIONAL], nargs <str> [OPTIONAL])
        - arg_name is the name of the argument in the command line.
        - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
        - type is the data-type of the variable.
        - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
        - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.

        Example of a valid parameter list:
            [('-m', 'mu', float, [1.,0.] '+'),
            ('-s', 'sigma', float, 1.)]
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
    global_params = {arg:getattr(args,arg) for arg in vars(args)}
    return global_params

def write_seed(seed_file:str, key:PRNGKeyArray) -> None:
    """
    Writes the seed to the seed file
    PARAMS:
    seedfile : file location of the seed file
    key : JAX random key
    """
    old_seed = read_seed(seed_file)
    with open(seed_file, 'w') as f:
        keystr = re.findall(r"[0-9]+", str(key))[-1]
        f.write(f"{keystr}\n{old_seed}")

def read_seed(seed_file:str) -> int:
    """
    Reads the seed from the seed file. The seed file consists of a line with the new seed and then a line with the old seed.
    PARAMS:
    seedfile : file location of the seed file
    key : JAX random key
    """
    with open(seed_file, 'r') as f:
        next_seed = int(f.read().split('\n')[0])
    return next_seed

def add_bookstein_anchors(_z, _zb2x:Float, _zb2y:Float, _zb3x:Float, B:Float = 0.3) -> Array:
    """
    Returns the Bookstein anchor coordinates for the first positions, in 2 dimensions.
    These Bookstein coordinates are restricted, meaning only 1 node is set, and the others are restricted and not dealt with in this function.
    PARAMS:
    _z (N-3, D) : regular nodes
    _zb2x : x-coordinate of the 2nd Bookstein anchor
    _zb2y : y_coordinate of the 2nd Bookstein anchor
    _zb3x : x-coordinate of the 3rd Bookstein anchor, of which the y-value must be 0.
    B : offset on the x-axis that the Bookstein anchors are put.
    """
    bookstein_anchors = jnp.zeros(shape=(3, 2))
    ## First positions at (-B,0)
    bookstein_anchors = bookstein_anchors.at[0,0].set(-B) 
    ## Second positions at (x,y)
    bookstein_anchors = bookstein_anchors.at[1,0].set(_zb2x)
    bookstein_anchors = bookstein_anchors.at[1,1].set(_zb2y)
    ## Third position at (x,0)
    bookstein_anchors = bookstein_anchors.at[2,0].set(_zb3x-B)
    _zc = jnp.concatenate([bookstein_anchors, _z])
    return _zc

def hyp_pnt(X:Array) -> jnp.ndarray:
    """
    Creates [z,x,y] positions in hyperbolic space by projecting [x,y] positions onto hyperbolic plane directly by solving the equation defining the hyperbolic plane.
    PARAMS
    X : array containing 2D points to be projected up onto the hyperbolic plane
    """
    N, D = X.shape
    z = jnp.sqrt(jnp.sum(X**2, axis=1)+1)
    x_hyp = jnp.zeros((N,D+1))
    x_hyp = x_hyp.at[:,0].set(z)
    x_hyp = x_hyp.at[:,1:].set(X)
    return x_hyp

def distance_mapping(d:Array, eps:float) -> Array:
    """
    Returns the Bernoulli probability p or beta mean mu given the latent distances d.
    PARAMS:
    d : (M,) latent distances.
    eps : offset for calculating p, to insure 0 < p < 1.
    """
    return jnp.clip(jnp.exp(-d**2), eps, 1-eps)

def euclidean_distance(z:Array) -> Array:
    """
    Returns the Euclidean distance between all elements in z, calculated via the Gram matrix.
    PARAMS:
    z : (N,D) latent positions
    """
    G = jnp.dot(z, z.T)
    g = jnp.diag(G)
    ones = jnp.ones_like(g)
    inside = jnp.maximum(jnp.outer(ones, g) + jnp.outer(g, ones) - 2 * G, 0)
    return jnp.sqrt(inside)

def lorentzian(v:Array, u:Array, keepdims:bool=False) -> Array:
    """
    Returns the Lorentzian prodcut of v and u, defined as -v_0*u_0 + SUM_{i=1}^N v_i*u_i
    PARAMS:
    v : (N,D) vector
    u : (N,D) vector
    keepdims : whether to keep the same number of dimensions (True), or flatten the new array.
    """
    signs = jnp.ones_like(v)
    signs = signs.at[:,0].set(-1)
    return jnp.sum(v*u*signs, axis=1, keepdims=keepdims)

def lorentz_distance(z:Array) -> Array:
    """
    Returns the hyperbolic distance between all N points in z as an N x N matrix.
    PARAMS:
    z : (N,D) points in hyperbolic space
    """
    def arccosh(x:Array) -> Array:
        """
        Definition of the arccosh function
        PARAMS:
        x : input
        """
        x_clip = jnp.maximum(x, jnp.ones_like(x, dtype=jnp.float32))
        return jnp.log(x_clip + jnp.sqrt(x_clip**2 - 1))
    signs = jnp.ones_like(z)
    signs = signs.at[:,0].set(-1)
    lor = jnp.dot(signs*z, z.T)
    ## Due to numerical instability, we can get NaN's on the diagonal, hence we force the diagonal to be zero.
    dist = arccosh(-lor)
    dist = dist.at[jnp.diag_indices_from(dist)].set(0)
    return dist

def parallel_transport(v:Array, nu:Array, mu:Array) -> Array:
    """
    Parallel transports the points v sampled around nu to the tangent space of mu.
    PARAMS:
    v  (N,D) : points on tangent space of nu [points on distribution around nu]
    nu (N,D) : point in hyperbolic space [center to move from] (mu_0 in Fig 2 of Nagano et al. (2019))
    mu (N,D) : point in hyperbolic space [center to move to]
    """
    alpha = -lorentzian(nu, mu, keepdims=True)
    u = v + lorentzian(mu - alpha*nu, v, keepdims=True)/(alpha+1) * (nu + mu)
    return u

def exponential_map(mu:Array, u:Array, eps:float=1e-6) -> Array:
    """
    Maps the points v on the tangent space of mu onto the hyperolic plane
    PARAMS:
    mu (N,D) : Transported middle points
    u (N,D) : Points to be mapped onto hyperbolic space (after parallel transport)
    eps : minimum value for the norm of u
    """
    ## Euclidean norm from mu_0 to v is the same as from mu to u is the same as the hyperbolic norm from mu to exp_mu(u), hence we can use the Euclidean norm of v.
    lor = lorentzian(u,u,keepdims=True)
    u_norm = jnp.sqrt(jnp.clip(lor, eps, lor))  ## If eps is too small, u_norm gets rounded right back to zero and then we divide by zero
    return jnp.cosh(u_norm) * mu + jnp.sinh(u_norm) * u / u_norm