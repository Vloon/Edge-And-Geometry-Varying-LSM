from helper_functions import get_cmd_params, set_GPU, triu2mat, lorentz_to_poincare

arguments = [('-gpu', 'gpu', str, ''),
             ('-k', 'k', float, 1.5),
             ('-sigz', 'sigma_z', float, 1.),
             ('-et', 'edge_type', str, 'bin'),
             ('-task', 'task', int, 0),
             ('-subj', 'subject', int, 0),
             ]

cmd_params = get_cmd_params(arguments)
gpu = cmd_params['gpu']
set_GPU(gpu)

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes as Axes
from matplotlib.patches import Arc
from typing import Tuple
from jaxtyping import Array, Float

def plot_hyperbolic_edges(p:Array,
                          A:Array,
                          ax:Axes=None,
                          R:Float=1,
                          linewidth:Float=0.5,
                          threshold:Float=0.4,
                          zorder:Float=0,
                          overwrt_alpha:Float=None) -> Axes:
    """
    Plots the edges on the Poincaré disk, meaning these will look curved.
    PARAMS
    p (N,2) : points on the Poincaré disk
    A (N,N) or (M) : (upper triangle of the) adjacency matrix.
    ax : axis to be plotted on
    R : disk radius
    linewidth : linewidth of the edges
    threshold : minimum edge weight for the edge to be plotted
    overwrt_alpha : overwrite the alpha value (can be used for binary edges to decrease the intensity)
    """
    def mirror(p:Array, R:Float=1) -> Array:
        """
        Mirrors point p in circle with radius R
        Based on: https://math.stackexchange.com/questions/1322444/how-to-construct-a-line-on-a-poincare-disk
        PARAMS:
            p (N,2) : N points on a 2-dimensional Poincaré disk
            R : disk radius
        RETURNS:
            p_inv (N,2) : Mirror of p
        """
        N, D = p.shape
        p_norm = np.sum(p**2,1)
        p_inv = R**2*p/np.reshape(np.repeat(p_norm,D), newshape=(N,D))
        return p_inv

    def bisectors(p:Array, R:Float=1) -> Tuple[Array, Array, Array, Array]:
        """
        Returns the function for the perpendicular bisector as ax + b
        Based on: https://www.allmath.com/perpendicularbisector.php
        PARAMS:
            p (N,2) : List of points
            R : disk radius
        RETURNS:
            a_self (N*(N+1)/2) : Slopes of the bisectors for each combination of points in p
            b_self (N*(N+1)/2) : Intersects of the bisectors for each combination of points in p
            a_inv (N) : Slopes of the bisectors for each point with its mirror
            b_inv (N) : Intersects of the bisectors for each point with its mirror
        """
        N, D = p.shape
        assert D == 2, 'Cannot visualize a Poincaré disk with anything other than 2 dimensional points'
        triu_indices = np.triu_indices(N, k=1)

        ## Get mirror of points
        p_inv = mirror(p, R)

        ## Tile the points to get all combinations
        x_rep = np.tile(p[:,0], N).reshape((N,N))
        y_rep = np.tile(p[:,1], N).reshape((N,N))

        ## Get midpoints
        mid_x_self = ((x_rep.T+x_rep)/2)[triu_indices]
        mid_y_self = ((y_rep.T+y_rep)/2)[triu_indices]
        mid_x_inv  = (p[:,0]+p_inv[:,0])/2
        mid_y_inv  = (p[:,1]+p_inv[:,1])/2

        ## Get slopes
        dx_self = (x_rep - x_rep.T)[triu_indices]
        dy_self = (y_rep- y_rep.T)[triu_indices]
        dx_inv  = p[:,0] - p_inv[:,0]
        dy_inv  = p[:,1] - p_inv[:,1]
        a_self = -1/(dy_self/dx_self)
        a_inv  = -1/(dy_inv/dx_inv)

        ## Get intersects
        b_self = -a_self*mid_x_self + mid_y_self
        b_inv  = -a_inv*mid_x_inv + mid_y_inv

        return a_self, b_self, a_inv, b_inv

    if ax is None:
        plt.figure()
        ax = plt.gca()
    N, D = p.shape
    M = N*(N-1)//2
    assert D == 2, 'Cannot visualize a Poincaré disk with anything other than 2 dimensional points'
    if len(A.shape) == 2:
        A = A[np.triu_indices(N, k=1)]

    ## Calculate perpendicular bisectors for points in p with each other, and with their mirrors.
    a_self, b_self, a_inv, b_inv = bisectors(p,R)

    ## Repeat elements according to the upper triangle indices
    first_triu_idc = np.triu_indices(N,k=1)[0]
    a_inv_rep = np.array([a_inv[i] for i in first_triu_idc])
    b_inv_rep = np.array([b_inv[i] for i in first_triu_idc])
    px_rep = np.array([p[i,0] for i in first_triu_idc])
    py_rep = np.array([p[i,1] for i in first_triu_idc])

    ## Get coordinates and radius of midpoint of the circle
    cx = (b_self-b_inv_rep)/(a_inv_rep-a_self)
    cy = a_self*cx + b_self
    cR = np.sqrt((px_rep-cx)**2 + (py_rep-cy)**2)

    second_triu_idc = np.triu_indices(N,k=1)[1]
    qx_rep = np.array([p[i,0] for i in second_triu_idc])
    qy_rep = np.array([p[i,1] for i in second_triu_idc])

    ## Get starting and ending angles of the arcs
    theta_p = np.degrees(np.arctan2(py_rep-cy, px_rep-cx))
    theta_q = np.degrees(np.arctan2(qy_rep-cy, qx_rep-cx))

    for m in range(M):
        if A[m] >= threshold:
            ## Correct the angles for quadrant wraparound
            if cx[m] > 0:
                theta_p[m] = theta_p[m]%360
                theta_q[m] = theta_q[m]%360

            ## Draw the arc and add it to the axis
            alpha = A[m] if overwrt_alpha is None else overwrt_alpha
            arc = Arc(xy=(cx[m], cy[m]), width=2*cR[m], height=2*cR[m], angle=0, theta1=min(theta_p[m],theta_q[m]), theta2=max(theta_p[m],theta_q[m]), linewidth=linewidth, alpha=alpha, zorder=zorder)
            ax.add_patch(arc)

    return ax

def plot_position_means(z_mn: Array,
                        ax: Axes = None,
                        **kwargs) -> Axes:
    """
    Args:
        z: (N, D) node positions posterior
        ax: plot axis
        size: marker size
        **kwargs: any plt.scatter parameters

    Returns:
        The axis
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.scatter(z_mn[:,0], z_mn[:,1], **kwargs)
    return ax

def plot_position_posterior(z: Array,
                            ax: Axes = None,
                            colors: Array = None,
                            size: Float = 1.,
                            alpha: Float = 0.05,
                            **kwargs) -> Axes:
    """
    Args:
        z: (num_particles, N, D) node positions
        ax: plot axis
        colors: (N,) list of colors for each node
        alpha: opacity
        **kwargs: any plt.scatter parameters

    Returns:
        The axis
    """
    num_particles, N, D = z.shape

    for n in range(N):
        ax.scatter(z[:,n,0], z[:,n,1], color=colors[n], s=size, alpha=alpha, **kwargs)
    return ax

hyperbolic = True
bookstein = True
add_disk = False
latpos = '_z' if hyperbolic else 'z'
edge_type = cmd_params['edge_type']
subject = cmd_params['subject']
task = cmd_params['task']
k = cmd_params['k'] # Between-cluster distance
sig_z = cmd_params['sigma_z'] # Within-cluster distance

## Load posterior positions
posterior_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"posterior_{edge_type}_{'hyp' if hyperbolic else 'euc'}_k{k}_sig{sig_z}_S{subject}_T{task}.pkl")
with open(posterior_filename, 'rb') as f:
    posterior = pickle.load(f)

if hyperbolic:
    posterior = lorentz_to_poincare(posterior)
num_particles, N, D = posterior.shape
posterior = np.array(posterior) # Convert from jax.numpy to numpy array

if bookstein:
    posterior[:,:3,:] += np.random.normal(loc=0, scale=1e-12, size=(num_particles, 3, D)) # Wiggle the bookstein nodes to deal with zero-division.

## Load observation
observation_filename = os.path.join(os.getcwd(), 'Data', 'cluster_sim', f"{edge_type}_observations_k{k}_sig{sig_z}_{'hyp' if hyperbolic else 'euc'}.pkl")
with open(observation_filename, 'rb') as f:
    observations = np.array(pickle.load(f)) # Convert from jax.numpy to numpy array
observation = observations[task, subject]

## Plotting parameters
cmap = 'nipy_spectral'
node_colors = np.array([plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, N)])
fsz = 10
alpha = 0.3

## Make plot
plt.figure(figsize=(fsz, fsz))
ax = plt.gca()
ax.axis('off')

z_mn = np.mean(posterior, axis=0)
ax = plot_hyperbolic_edges(z_mn, observation, ax)
ax = plot_position_means(z_mn, ax, s=3, color='k', marker='x')
ax = plot_position_posterior(posterior, ax, node_colors, alpha=alpha)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

if add_disk:
    poincare_disk = plt.Circle((0, 0), 1, color='k', fill=False, clip_on=False)
    ax.add_patch(poincare_disk)
    if zoom:
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

figure_filename = os.path.join(os.getcwd(), 'Figures', 'cluster_sim', f"posterior_{edge_type}_{'hyp' if hyperbolic else 'euc'}_k{k}_sig{sig_z}_S{subject}_T{task}.png")
bbox = None if add_disk else 'tight'
plt.savefig(figure_filename, bbox_inches=bbox)
plt.close()