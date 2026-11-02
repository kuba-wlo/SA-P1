import numpy as np
import pytest

from src import config, dane, imse, teoria


@pytest.fixture(scope="module")
def wyniki_mc():
    """Wyniki dla W1-W4 na tych samych strumieniach co w eksperymencie E3."""
    return {
        nazwa: imse.imse_mc(dane.generator("E3", i), config.THETA_0, alpha, v)
        for i, (nazwa, (alpha, v)) in enumerate(config.WARIANTY.items())
    }


def test_imse_rowne_sumie_skladowych(wyniki_mc):
    for w in wyniki_mc.values():
        np.testing.assert_allclose(w["imse"], w["bias2"] + w["var"], rtol=1e-10)


@pytest.mark.parametrize("nazwa", list(config.WARIANTY))
def test_zgodnosc_z_teoria(wyniki_mc, nazwa):
    alpha, v = config.WARIANTY[nazwa]
    w = wyniki_mc[nazwa]
    teor = teoria.imse(w["H"], alpha, v)
    assert np.max(np.abs(w["imse"] - teor) / teor) < 0.03


def test_theta0_nie_wplywa_na_imse():
    a = imse.imse_mc(np.random.default_rng(11), 1.0, 0.1, 10.0, N=500, M=20)
    b = imse.imse_mc(np.random.default_rng(11), -37.5, 0.1, 10.0, N=500, M=20)
    for klucz in ("imse", "bias2", "var"):
        np.testing.assert_allclose(a[klucz], b[klucz], rtol=1e-9)


def test_powtarzalnosc():
    a = imse.imse_mc(dane.generator("E3", 0), 1.0, 0.01, 0.5, N=500, M=20)
    b = imse.imse_mc(dane.generator("E3", 0), 1.0, 0.01, 0.5, N=500, M=20)
    for klucz in ("imse", "bias2", "var"):
        np.testing.assert_array_equal(a[klucz], b[klucz])


def test_strumienie_niezalezne():
    a = dane.generator("E3", 0).random(5)
    b = dane.generator("E3", 1).random(5)
    c = dane.generator("E4").random(5)
    assert not np.array_equal(a, b) and not np.array_equal(a, c)


def test_imse_realizacji_srednia_rowna_imse_mc():
    rng = np.random.default_rng(12)
    theta_gw, theta = dane.pomiary(rng, 1.0, 0.1, 10.0, 300, M=15)
    H = [1, 5, 20]
    per = imse.imse_realizacji(theta, theta_gw, H, 50)
    w = imse.imse_mc(np.random.default_rng(12), 1.0, 0.1, 10.0, N=300, M=15, H_values=H, h_max=50)
    assert per.shape == (15, 3)
    np.testing.assert_allclose(per.mean(axis=0), w["imse"], rtol=1e-12)
