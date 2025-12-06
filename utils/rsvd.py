"""
Randomized SVD (RSVD) implementation for efficient low-rank approximation.

Based on "Finding Structure with Randomness: Probabilistic Algorithms for
Constructing Approximate Matrix Decompositions" by Halko et al. (2011)
"""

import torch


def randomized_svd(
    M: torch.Tensor,
    rank: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    power_iteration_normalizer: str = 'QR',
    random_state: int = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a truncated randomized SVD of matrix M.
    
    Args:
        M: Input matrix of shape (m, n)
        rank: Target rank for the approximation
        n_oversamples: Additional samples to improve accuracy (default: 10)
        n_iter: Number of power iterations (default: 2)
        power_iteration_normalizer: 'QR' or 'LU' normalization (default: 'QR')
        random_state: Random seed for reproducibility
        
    Returns:
        U: Left singular vectors of shape (m, rank)
        S: Singular values of shape (rank,)
        VT: Right singular vectors of shape (rank, n)
    """
    m, n = M.shape
    device = M.device
    dtype = M.dtype
    
    # Set random seed if provided
    if random_state is not None:
        generator = torch.Generator(device=device).manual_seed(random_state)
    else:
        generator = None
    
    # Determine the number of random vectors
    n_random = min(rank + n_oversamples, min(m, n))
    
    # Generate random matrix
    if generator is not None:
        Omega = torch.randn(n, n_random, dtype=dtype, device=device, generator=generator)
    else:
        Omega = torch.randn(n, n_random, dtype=dtype, device=device)
    
    # Stage A: Compute an approximate range
    Y = M @ Omega
    
    # Power iterations to improve accuracy
    for _ in range(n_iter):
        if power_iteration_normalizer == 'QR':
            Y, _ = torch.linalg.qr(Y)
            Z, _ = torch.linalg.qr(M.T @ Y)
            Y = M @ Z
        elif power_iteration_normalizer == 'LU':
            Y, _ = torch.linalg.lu(Y, pivot=False)
            Z, _ = torch.linalg.lu(M.T @ Y, pivot=False)
            Y = M @ Z
        else:
            # No normalization (not recommended)
            Y = M @ (M.T @ Y)
    
    # Orthonormalize Y
    Q, _ = torch.linalg.qr(Y)
    
    # Stage B: Compute the SVD on the projected matrix
    B = Q.T @ M
    U_tilde, S, VT = torch.linalg.svd(B, full_matrices=False)
    
    # Compute the left singular vectors of M
    U = Q @ U_tilde
    
    # Truncate to the desired rank
    U = U[:, :rank]
    S = S[:rank]
    VT = VT[:rank, :]
    
    return U, S, VT


def randomized_svd_adaptive(
    M: torch.Tensor,
    ratio: float = 0.8,
    n_oversamples: int = 10,
    n_iter: int = 2,
    power_iteration_normalizer: str = 'QR',
    random_state: int = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a truncated randomized SVD with adaptive rank based on compression ratio.
    
    Args:
        M: Input matrix of shape (m, n)
        ratio: Compression ratio to determine target rank
        n_oversamples: Additional samples to improve accuracy (default: 10)
        n_iter: Number of power iterations (default: 2)
        power_iteration_normalizer: 'QR' or 'LU' normalization (default: 'QR')
        random_state: Random seed for reproducibility
        
    Returns:
        U: Left singular vectors
        S: Singular values
        VT: Right singular vectors
    """
    m, n = M.shape
    # Calculate target rank based on compression ratio
    rank = int(m * n * ratio / (m + n))
    rank = max(1, min(rank, min(m, n)))  # Ensure rank is valid
    
    return randomized_svd(M, rank, n_oversamples, n_iter, power_iteration_normalizer, random_state)
