import numpy as np
from scipy.linalg import eigh


def embed_points_isometric(points: np.ndarray) -> np.ndarray:
    """
    Embeds N points into an (N-1)-dimensional space, preserving distances,
    with options to handle the zero vector.

    Args:
        points: np.ndarray of shape (N, n)

    Returns:
        embedded: np.ndarray of shape (N, N-1) with distances preserved
    """
    N, n = points.shape

    zero_mask = np.all(points == 0, axis=1)
    zero_idx = np.where(zero_mask)[0]

    if len(zero_idx) == 0:
        raise ValueError(
            "Input must contain the zero vector when ensure_zero_exists=True"
        )
    else:
        # Move 0 to position 0 if not already there
        if zero_idx[0] != 0:
            points[[0, zero_idx[0]]] = points[[zero_idx[0], 0]]

    # Compute pairwise squared distances
    points_sq = np.sum(points**2, axis=1)
    D = points_sq[:, np.newaxis] + points_sq[np.newaxis, :] - 2 * points @ points.T

    D = np.maximum(D, 0)

    # Extract submatrix for non-zero points (indices 1 to N-1)
    D_sub = D[1:, 1:]

    # Compute the Gram matrix (G_ij = x_i · x_j) from distances
    # using: G_ij = (D_{0i} + D_{0j} - D_{ij}) / 2
    D0 = D[0, 1:]
    G = (D0[:, np.newaxis] + D0[np.newaxis, :] - D_sub) / 2

    # Eigendecomposition of G = U Λ U^T
    eigvals, eigvecs = eigh(G)

    # Ensure eigenvalues are non-negative - for to numerical precision
    eigvals = np.maximum(eigvals, 0)

    # Construct the embedding Y = U sqrt(Λ)
    Y = eigvecs * np.sqrt(eigvals)

    # Pad with 0 for the first point
    embedded = np.zeros((N, N-1))
    embedded[1:] = Y

    return embedded