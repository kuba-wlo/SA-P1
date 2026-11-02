"""Wyznaczanie IMSE(H) metodą Monte Carlo."""

import numpy as np

from src import config, dane, estymator


def bledy(theta, theta_gw, H, h_max):
    """Błędy e_k = theta^_k(H) - theta*_k dla k = h_max-1, ..., N-1."""
    k0 = h_max - 1
    return estymator.okno_przesuwne(theta, H)[..., k0:] - theta_gw[k0:]


def imse_realizacji(theta, theta_gw, H_values, h_max):
    """IMSE każdej realizacji osobno (średnia po k), kształt (M, len(H_values))."""
    theta = np.atleast_2d(theta)
    return np.column_stack(
        [np.mean(bledy(theta, theta_gw, H, h_max) ** 2, axis=-1) for H in H_values]
    )


def imse_mc(rng, theta_0, alpha, v, N=config.N, M=config.M,
            H_values=config.H_VALUES, h_max=config.H_MAX):
    """IMSE(H) uśrednione po M realizacjach wraz ze składowymi bias^2 i var.

    bias_k i var_k (ddof=0) liczone po realizacjach, następnie uśredniane po k,
    więc IMSE = bias2 + var z dokładnością numeryczną.
    """
    theta_gw, theta = dane.pomiary(rng, theta_0, alpha, v, N, M=M)
    H_values = np.asarray(H_values)
    wynik = {name: np.empty(len(H_values)) for name in ("imse", "bias2", "var")}
    for j, H in enumerate(H_values):
        e = bledy(theta, theta_gw, int(H), h_max)
        wynik["imse"][j] = np.mean(e**2)
        wynik["bias2"][j] = np.mean(np.mean(e, axis=0) ** 2)
        wynik["var"][j] = np.mean(np.var(e, axis=0, ddof=0))
    wynik["H"] = H_values
    return wynik
