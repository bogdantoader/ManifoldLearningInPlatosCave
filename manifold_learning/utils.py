"""Functions used in the plotting and displaying of the results."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_two_matrices(A1, A2, colorbar=True, size=None):
    """Function to plot two images side by side."""

    size0 = plt.rcParams["figure.figsize"]

    if size is None:
        plt.rcParams["figure.figsize"] = size0[0] * 2, size0[1]
    else:
        plt.rcParams["figure.figsize"] = size[0], size[1]

    plt.subplot(1, 2, 1)
    plt.imshow(A1)
    if colorbar:
        plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(A2)
    if colorbar:
        plt.colorbar()

    plt.tight_layout()
    plt.rcParams["figure.figsize"] = size0


@jax.jit
def compute_density_pt(coords, r, i):
    """Count the number of points in an l2 ball of radius r centred at the
    i-th point in a list of points given by coords."""

    distances = jax.vmap(
        lambda pt0, pti: jnp.linalg.norm(pt0 - pti), in_axes=(None, 0)
    )(coords[i], coords)
    return jnp.sum(distances < r)


@jax.jit
def compute_density(coords, r):
    """Compute the density at each point in a list of points given by coords,
    using an l2 ball of radius r."""

    densities = jax.vmap(compute_density_pt, in_axes=(None, None, 0))(
        coords, r, jnp.arange(coords.shape[0])
    )
    return densities / jnp.sum(densities)


@jax.jit
def compute_density_Y_img(img_dists, r):
    """Count the number of elements of the img_dists array smaller than r."""

    return jnp.sum(img_dists < r)


@jax.jit
def compute_density_Y(A, r):
    """Given the pair-wise distance matrix A, compute the density at each point
    using an l2 ball of radius r."""

    densities = jax.vmap(compute_density_Y_img, in_axes=(0, None))(A, r)
    return densities / jnp.sum(densities)
