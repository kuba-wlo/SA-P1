import time

import numpy as np
import pytest

from src import config, dane, estymator


def wolna_petla(theta, H):
    wynik = np.full(len(theta), np.nan)
    for k in range(H - 1, len(theta)):
        wynik[k] = sum(theta[k - i] for i in range(H)) / H
    return wynik


@pytest.mark.parametrize("H", [1, 2, 3, 7, 20])
def test_zgodnosc_z_wolna_petla(H):
    theta = np.random.default_rng(5).normal(size=20) + np.arange(20)
    np.testing.assert_allclose(
        estymator.okno_przesuwne(theta, H), wolna_petla(theta, H), rtol=1e-12, equal_nan=True
    )


def test_nan_dla_niepelnego_okna():
    wynik = estymator.okno_przesuwne(np.arange(10.0), 4)
    assert np.all(np.isnan(wynik[:3]))
    assert not np.any(np.isnan(wynik[3:]))


def test_H1_zwraca_pomiary():
    theta = np.random.default_rng(6).normal(size=(3, 15))
    np.testing.assert_allclose(estymator.okno_przesuwne(theta, 1), theta, rtol=1e-12)


def test_macierz_wiersz_po_wierszu():
    theta = np.random.default_rng(7).normal(size=(4, 30))
    wynik = estymator.okno_przesuwne(theta, 5)
    for m in range(4):
        np.testing.assert_allclose(wynik[m], wolna_petla(theta[m], 5), rtol=1e-12, equal_nan=True)


def test_niepoprawne_H():
    with pytest.raises(ValueError):
        estymator.okno_przesuwne(np.arange(5.0), 0)
    with pytest.raises(ValueError):
        estymator.okno_przesuwne(np.arange(5.0), 6)


def test_czas_dla_macierzy_M_N():
    _, theta = dane.pomiary(np.random.default_rng(8), 1.0, 0.1, 10.0, config.N, M=config.M)
    start = time.perf_counter()
    for H in config.H_VALUES:
        estymator.okno_przesuwne(theta, H)
    assert time.perf_counter() - start < 5.0
