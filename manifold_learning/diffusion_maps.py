"""Implementation of the Diffusion Maps algorithm, as described in [1] and
adapted in [2].

[1] S. Lafon. “Diffusion maps and geometric harmonics”.
    PhD Thesis. Yale University, 2004.
[2] R. R. Lederman and R. Talmon. “Learning the geometry of common latent
    variables using alternating-diffusion”.
    Applied and Computational Harmonic Analysis 44.3 (2018), pp. 509–536.
"""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def compute_distance_matrix(X):
    """Compute the distance matrix given the data matrix X.

    Parameters:
    -----------
    X : float64[data_dim, N_pts]
        The data matrix, where data_dim is the dimension of the data
        and N_pts is the number of data points.

    Returns:
    --------
    A : float64[N_pts, N_pts]
        The distance matrix, where A[i, j] is the l2 distance
        between X[i] and X[j].
    """

    N = X.shape[1]

    # Vectorized distance function between one data point and an array of data points.
    def A_row(x1, v):
        return jax.vmap(lambda x2: jnp.linalg.norm(x1 - x2, 2), in_axes=(1))(v)

    # Iteratively better than vmap (memory-wise).
    A = np.zeros([N, N])
    for i in tqdm(range(N)):
        A[i, :] = A_row(X[:, i], X)

    return jnp.array(A)


def compute_embedding(A, eps, t=1):
    """Apply the diffusion maps algorithm to the distance matrix A
    to obtain the manifold embedding of the data X.


    Parameters:
    -----------
    A : float64[N_pts, N_pts]
        The distance matrix, where A[i, j] is the l2 distance
        between X[i] and X[j].
    eps : float64
        Gaussian kernel width.
    t : int
        Number of diffusion time steps.

    Returns:
    --------
    coords : float64[N_pts, N_pts]
        The diffusion coordinates
    w : float[N_pts]
        Eigenvalues of the Laplacian L
    v : float[N_pts, N_pts]
        The eigenvectors of the Laplacian L, normalized by v[:,0]
    W : float[N_pts, N_pts]
        The similarity matrix.
    L : float[N_pts, N_pts]
        The Laplacian matrix.

    """

    N = A.shape[0]

    W = jnp.exp(-(A**2) / eps)
    Q = jnp.diag(1 / jnp.sum(W, axis=1))
    Kt = Q @ W @ Q
    Qt = jnp.diag(1 / jnp.sqrt(jnp.sum(Kt, axis=1)))
    K = Qt @ Kt @ Qt

    L = jnp.eye(N) - K

    # Ensure that L is symmetric so we can apply jax.numpy.linalg.eigh
    assert jnp.max(jnp.abs(L - L.transpose())) < 1e-15
    w, v = jnp.linalg.eigh(L)

    # We are interested in the eigenvalues of K, in descending order
    w = 1 - w

    v = jax.vmap(lambda vi: vi / v[:, 0], in_axes=1, out_axes=1)(v)

    # And finally the embedding coordinates
    coords = jax.vmap(lambda wi, vi: wi**t * vi, in_axes=(0, 1), out_axes=1)(w, v)

    return coords, w, v, W, L
