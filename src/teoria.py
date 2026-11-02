"""Wzory teoretyczne dla estymatora z oknem przesuwnym przy trendzie liniowym."""

import numpy as np


def bias(H, alpha):
    """Obciążenie estymaty: -alpha*(H-1)/2."""
    return -alpha * (np.asarray(H, dtype=float) - 1.0) / 2.0


def bias2(H, alpha):
    """Kwadrat obciążenia: alpha^2*(H-1)^2/4."""
    return bias(H, alpha) ** 2


def wariancja(H, v):
    """Wariancja estymaty: v/H."""
    return v / np.asarray(H, dtype=float)


def imse(H, alpha, v):
    """Teoretyczne IMSE(H) = bias^2(H) + Var(H)."""
    return bias2(H, alpha) + wariancja(H, v)


def h_opt(alpha, v, h_max):
    """Całkowite H_opt jako argmin IMSE po H w [1, h_max]; zwraca (H_opt, IMSE_min)."""
    H = np.arange(1, h_max + 1)
    wartosci = imse(H, alpha, v)
    i = int(np.argmin(wartosci))
    return int(H[i]), float(wartosci[i])


def h_opt_przyblizenie(alpha, v):
    """Przybliżenie ciągłe H_opt ≈ (2v/alpha^2)^(1/3); dla alpha = 0 zwraca inf."""
    if alpha == 0:
        return float("inf")
    return float((2.0 * v / alpha**2) ** (1.0 / 3.0))
