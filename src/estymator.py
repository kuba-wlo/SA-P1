"""Estymator z oknem przesuwnym liczony przez skumulowaną sumę."""

import numpy as np


def okno_przesuwne(theta, H):
    """Średnia z ostatnich H pomiarów: theta^_k = (1/H) * sum_{i=0}^{H-1} theta_{k-i}.

    Działa wzdłuż ostatniej osi (wektor (N,) lub macierz (M, N)).
    Dla k < H-1 okno nie jest pełne, więc wynik to NaN.
    """
    theta = np.asarray(theta, dtype=float)
    N = theta.shape[-1]
    if not 1 <= H <= N:
        raise ValueError(f"H musi należeć do [1, {N}], podano {H}")
    zera = np.zeros(theta.shape[:-1] + (1,))
    s = np.concatenate([zera, np.cumsum(theta, axis=-1)], axis=-1)
    wynik = np.full(theta.shape, np.nan)
    wynik[..., H - 1:] = (s[..., H:] - s[..., :-H]) / H
    return wynik
