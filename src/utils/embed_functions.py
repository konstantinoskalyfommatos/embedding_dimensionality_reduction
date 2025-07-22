import torch


def embed_points_isometric(points: torch.Tensor) -> torch.Tensor:
    """
    Embeds N points into an (N-1)-dimensional space, preserving distances,
    with options to handle the zero vector.

    # NOTE: Check out this jupyter notebook for more details:
    # https://drive.google.com/file/d/16mqMwqXe_JDxvxHDw2qrfQY7JHsU7Rrx/view?usp=sharing

    Args:
        points: torch.Tensor of shape (N, n) on the correct device

    Returns:
        embedded: torch.Tensor of shape (N, N-1) with distances preserved
    """
    device = points.device
    N, n = points.shape

    zero_mask = torch.all(points == 0, dim=1)
    zero_idx = torch.where(zero_mask)[0]

    if len(zero_idx) == 0:
        raise ValueError(
            "Input must contain the zero vector when ensure_zero_exists=True"
        )
    else:
        # Move 0 to position 0 if not already there
        if zero_idx[0] != 0:
            idx0 = zero_idx[0].item()
            points = points.clone()
            points[[0, idx0]] = points[[idx0, 0]]

    # Compute pairwise squared distances
    points_sq = torch.sum(points ** 2, dim=1)
    D = points_sq[:, None] + points_sq[None, :] - 2 * torch.matmul(points, points.T)
    D = torch.clamp(D, min=0)

    # Extract submatrix for non-zero points (indices 1 to N-1)
    D_sub = D[1:, 1:]

    # Compute the Gram matrix
    D0 = D[0, 1:]
    G = (D0[:, None] + D0[None, :] - D_sub) / 2

    # Eigendecomposition of G
    eigvals, eigvecs = torch.linalg.eigh(G)

    # Ensure eigenvalues are non-negative
    eigvals = torch.clamp(eigvals, min=0)

    # Construct the embedding Y = U sqrt(Λ)
    Y = eigvecs * torch.sqrt(eigvals)

    # Pad with 0 for the first point
    embedded = torch.zeros((N, N-1), device=device, dtype=points.dtype)
    embedded[1:] = Y

    return embedded