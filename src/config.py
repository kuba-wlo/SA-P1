"""Parametry symulacji (jedyne miejsce ich definicji)."""

from pathlib import Path

THETA_0 = 1.0
N = 2000
M = 200
H_MAX = 100
H_VALUES = tuple(range(1, H_MAX + 1))
SEED = 2026

# Numery strumieni SeedSequence(SEED).spawn(...) dla eksperymentów
STRUMIENIE = {"E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4, "E6": 5, "E7": 6}

# Warianty: nazwa -> (alpha, v)
WARIANTY = {
    "W1": (0.01, 0.5),
    "W2": (0.01, 10.0),
    "W3": (0.1, 0.5),
    "W4": (0.1, 10.0),
}
WARIANT_STALY = {"W0": (0.0, 10.0)}

# E1: weryfikacja generatora
E1_V = 10.0
E1_PROBKI = 10**6
E1_KOSZE = 80

# E2: przebiegi czasowe
E2_WARIANT = "W4"
E2_POCZATEK_WYCINKA = 1000
E2_DLUGOSC_WYCINKA = 300

# E4: dekompozycja
E4_WARIANT = "W2"

# E5: pojedyncza realizacja wobec uśrednienia
E5_WARIANT = "W4"
E5_N = 500
E5_REALIZACJE = 500

# E6: stały parametr
E6_N_LISTA = (10, 30, 100, 300, 1000, 3000, 10000)
E6_M = 2000

# E7: wrażliwość H_opt
E7_ALPHA_LISTA = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
E7_V_PRZY_ALPHA = 2.0
E7_V_LISTA = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
E7_ALPHA_PRZY_V = 0.05

# Ścieżki wyników
KATALOG_GLOWNY = Path(__file__).resolve().parent.parent
KATALOG_WYNIKOW = KATALOG_GLOWNY / "results"
KATALOG_WYKRESOW = KATALOG_WYNIKOW / "figures"
KATALOG_TABEL = KATALOG_WYNIKOW / "tables"
DPI = 300
