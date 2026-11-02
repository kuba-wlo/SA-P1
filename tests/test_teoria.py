import numpy as np
import pytest

from src import config, teoria

# Wartości z sekcji 3 instrukcji: (alpha, v, H_opt, IMSE_min)
TABELA = {
    "W1": (0.01, 0.5, 22, 0.0338),
    "W2": (0.01, 10.0, 59, 0.2536),
    "W3": (0.1, 0.5, 5, 0.1400),
    "W4": (0.1, 10.0, 13, 1.1292),
    "W0": (0.0, 10.0, 100, 0.1000),
}


@pytest.mark.parametrize("nazwa", TABELA)
def test_hopt_i_imse_min(nazwa):
    alpha, v, h_ref, imse_ref = TABELA[nazwa]
    h, imse_min = teoria.h_opt(alpha, v, config.H_MAX)
    assert h == h_ref
    assert round(imse_min, 4) == pytest.approx(imse_ref, abs=1e-12)


def test_parametry_zgodne_z_config():
    for nazwa, (alpha, v) in {**config.WARIANTY, **config.WARIANT_STALY}.items():
        assert TABELA[nazwa][:2] == (alpha, v)


def test_skladowe():
    H = np.array([1, 2, 11])
    np.testing.assert_allclose(teoria.bias(H, 0.1), [0.0, -0.05, -0.5])
    np.testing.assert_allclose(teoria.wariancja(H, 10.0), [10.0, 5.0, 10.0 / 11])
    np.testing.assert_allclose(
        teoria.imse(H, 0.1, 10.0), teoria.bias2(H, 0.1) + teoria.wariancja(H, 10.0)
    )


def test_staly_parametr_bez_obciazenia():
    H = np.arange(1, 101)
    np.testing.assert_array_equal(teoria.bias(H, 0.0), 0.0)
    np.testing.assert_allclose(teoria.imse(H, 0.0, 10.0), 10.0 / H)


@pytest.mark.parametrize("alpha, v", [(0.01, 10.0), (0.1, 0.5), (0.05, 2.0)])
def test_punkt_przeciecia(alpha, v):
    h = teoria.h_przeciecia(alpha, v)
    assert teoria.bias2(h, alpha) == pytest.approx(teoria.wariancja(h, v), rel=1e-9)


def test_przyblizenie_ciagle():
    assert teoria.h_opt_przyblizenie(0.1, 10.0) == pytest.approx(2000 ** (1 / 3))
    assert teoria.h_opt_przyblizenie(0.0, 10.0) == float("inf")
