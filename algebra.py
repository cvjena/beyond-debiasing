import numpy as np
from numpy.typing import NDArray

def has_leftinverse(matrix: NDArray) -> bool:
    """Returns whether the matrix is left-invertible. That is, it returns whether
    a matrix A' exists to the given matrix A such that A'Ax=x for all x.

    Args:
        matrix (NDArray): Matrix A as numpy array.

    Returns:
        bool: Whether the given matrix A has a left inverse.
    """

    # A matrix can only have a left-inverse if it is of full column rank.
    m, n = matrix.shape  # rows, columns
    _, s, _ = np.linalg.svd(matrix)
    rank = np.sum(s > np.finfo(matrix.dtype).eps)

    return rank == n and n <= m


def random_orthogonal_matrix(
    n: int, complex: bool = False, seed: int = None
) -> NDArray:
    """Random orthogonal matrix distributed with Haar measure.

    Returns a random orthogonal matrix. To ensure randomness, we have to choose
    from the distribution created by the Haar measure. The calculation follows
    the results of:
        Mezzadri, Francesco: How to generate random matrices from the classical
        compact groups. In: NOTICES of the AMS, Vol. 54 (54 (2007).
        URL: https://arxiv.org/pdf/math-ph/0609050.

    Args:
        n (int): Matrix returned has dimensions nxn.
        complex (bool, optional): Whether or not the returned matrix contains complex numbers. Defaults to False.
        seed (int, optional): If int the seed to generate reproducible results. Defaults to None.

    Returns:
        NDArray: Random orthogonal matrix distributed with Haar measure.
    """

    if not seed is None:
        np.random.seed(seed)

    # The original algorithm provided by Mezzari's is only defined for complex
    # initialization.
    if complex:
        z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.lib.scimath.sqrt(
            2.0
        )
    else:
        z = np.random.randn(n, n)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph, q)

    return q
