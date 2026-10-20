"""Surrogate time series that keep a window's amplitude spectrum and value distribution.

Iterative amplitude-adjusted Fourier transform (IAAFT) surrogates (Schreiber & Schmitz, 1996)
preserve every window's marginal distribution exactly and its power spectrum approximately, and
randomise everything else (phase relations, the position and shape of local events). A probe that
scores as well on surrogates as on the original windows uses only spectral and distributional
information, which spectral and order statistics also carry.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def iaaft(x: NDArray, n_iter: int = 50, seed: int = 0) -> NDArray[np.float64]:
    """Return one IAAFT surrogate per row of ``x``.

    Args:
        x: Array of shape (N, L); each row is surrogated independently.
        n_iter: Number of alternating spectrum / rank-order projections.
        seed: Seed of the initial random shuffle.

    Returns:
        Array of shape (N, L). Each row is a permutation of the corresponding row of ``x`` (the
        marginal is exact) whose amplitude spectrum approximates that of ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (N, L), got {x.shape}")
    rng = np.random.default_rng(seed)
    length = x.shape[1]
    sorted_values = np.sort(x, axis=1)
    amplitude = np.abs(np.fft.rfft(x, axis=1))
    s = rng.permuted(x, axis=1)
    for _ in range(n_iter):
        spectrum = np.fft.rfft(s, axis=1)
        s = np.fft.irfft(amplitude * np.exp(1j * np.angle(spectrum)), n=length, axis=1)
        ranks = np.argsort(np.argsort(s, axis=1), axis=1)
        s = np.take_along_axis(sorted_values, ranks, axis=1)
    return s


def spectral_error(x: NDArray, s: NDArray) -> NDArray[np.float64]:
    """Relative L2 error between the amplitude spectra of ``x`` and ``s``, per row."""
    a = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64), axis=1))
    b = np.abs(np.fft.rfft(np.asarray(s, dtype=np.float64), axis=1))
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-12)
