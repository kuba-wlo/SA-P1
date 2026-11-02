"""Uruchomienie wszystkich eksperymentów i wykresów: python -m src.run_all"""

import time

from src import config, eksperymenty, wykresy

KROKI = [
    ("E1", eksperymenty.e1_generator, [wykresy.e1_generator]),
    ("E2", eksperymenty.e2_przebiegi, [wykresy.e2_przebiegi]),
    ("E3", eksperymenty.e3_warianty, [wykresy.e3_warianty]),
    ("E4", eksperymenty.e4_dekompozycja, [wykresy.e4_dekompozycja]),
    ("E5", eksperymenty.e5_realizacja, [wykresy.e5_realizacja, wykresy.e5_histogram_hopt]),
    ("E6", eksperymenty.e6_staly_parametr, [wykresy.e6_staly_parametr, wykresy.e6_wariancja_N]),
    ("E7", eksperymenty.e7_wrazliwosc, [wykresy.e7_wrazliwosc]),
]


def main():
    podsumowanie = config.KATALOG_WYNIKOW / "podsumowanie.json"
    if podsumowanie.exists():
        podsumowanie.unlink()
    start = time.perf_counter()
    for nazwa, oblicz, rysuj in KROKI:
        t = time.perf_counter()
        oblicz()
        for f in rysuj:
            f()
        print(f"{nazwa}: {time.perf_counter() - t:.1f} s")
    print(f"Razem: {time.perf_counter() - start:.1f} s, wyniki w {config.KATALOG_WYNIKOW}")


if __name__ == "__main__":
    main()
