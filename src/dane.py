"""Generowanie trendu liniowego, zakłóceń trójkątnych i pomiarów."""

import numpy as np


def trend(theta_0, alpha, N):
    """Prawdziwy parametr theta*_k = theta_0 + alpha*k, k = 0, ..., N-1."""
    return theta_0 + alpha * np.arange(N, dtype=float)


def zaklocenia(rng, v, rozmiar):
    """Zakłócenia trójkątne o średniej 0 i wariancji v.

    z = sqrt(v)*sqrt(6)*(U1 + U2 - 1), U1, U2 ~ U[0, 1); nośnik [-sqrt(6v), sqrt(6v)].
    """
    u1 = rng.random(rozmiar)
    u2 = rng.random(rozmiar)
    return np.sqrt(v) * np.sqrt(6.0) * (u1 + u2 - 1.0)


def gestosc(z, v):
    """Gęstość teoretyczna f(z) = (a-|z|)/a^2 na [-a, a], a = sqrt(6v)."""
    a = np.sqrt(6.0 * v)
    z = np.asarray(z, dtype=float)
    return np.where(np.abs(z) <= a, (a - np.abs(z)) / a**2, 0.0)


def pomiary(rng, theta_0, alpha, v, N, M=None):
    """Pomiary theta_k = theta*_k + z_k.

    Dla M = None zwraca (theta*, theta) o kształcie (N,), w przeciwnym razie
    theta ma kształt (M, N) - każdy wiersz to jedna realizacja.
    """
    theta_gw = trend(theta_0, alpha, N)
    rozmiar = N if M is None else (M, N)
    return theta_gw, theta_gw + zaklocenia(rng, v, rozmiar)
