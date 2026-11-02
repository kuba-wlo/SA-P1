import numpy as np
import pytest

from src import config, dane


@pytest.mark.parametrize("v", [0.5, 10.0])
def test_generator_momenty_i_nosnik(v):
    rng = np.random.default_rng(np.random.SeedSequence(config.SEED))
    z = dane.zaklocenia(rng, v, 10**6)
    # średnia w tolerancji 1% odchylenia standardowego, wariancja 1% względnie
    assert abs(z.mean()) < 0.01 * np.sqrt(v)
    assert z.var() == pytest.approx(v, rel=0.01)
    assert np.max(np.abs(z)) <= np.sqrt(6 * v)


def test_gestosc_calkuje_sie_do_jedynki():
    v = 10.0
    a = np.sqrt(6 * v)
    z = np.linspace(-a, a, 200001)
    f = dane.gestosc(z, v)
    assert np.sum((f[1:] + f[:-1]) / 2 * np.diff(z)) == pytest.approx(1.0, abs=1e-6)
    assert dane.gestosc(0.0, v) == pytest.approx(1 / a)
    assert dane.gestosc(a + 1, v) == 0.0


def test_trend():
    np.testing.assert_allclose(dane.trend(1.0, 0.1, 4), [1.0, 1.1, 1.2, 1.3])


def test_pomiary_ksztalty():
    rng = np.random.default_rng(1)
    theta_gw, theta = dane.pomiary(rng, 1.0, 0.1, 10.0, 50)
    assert theta_gw.shape == theta.shape == (50,)
    theta_gw, theta = dane.pomiary(rng, 1.0, 0.1, 10.0, 50, M=7)
    assert theta_gw.shape == (50,) and theta.shape == (7, 50)


def test_powtarzalnosc():
    def generuj():
        rng = np.random.default_rng(np.random.SeedSequence(config.SEED).spawn(1)[0])
        return dane.pomiary(rng, 1.0, 0.1, 10.0, 100, M=5)[1]

    np.testing.assert_array_equal(generuj(), generuj())
