# Estymacja parametru zmiennego liniowo: estymator z oknem przesuwnym

Projekt z przedmiotu *Sterowanie adaptacyjne* (projekt 1, grupa C).

## Zadanie

Prawdziwy parametr zmienia się liniowo, a pomiary są zakłócone szumem o symetrycznym
rozkładzie trójkątnym:

```
θ*_k = θ_0 + α·k
θ_k  = θ*_k + z_k,     E[z_k] = 0,  Var[z_k] = v,  k = 0, ..., N-1
z_k  = sqrt(v)·sqrt(6)·(U1 + U2 - 1),   U1, U2 ~ U[0, 1)
```

Parametr jest estymowany średnią z okna przesuwnego o szerokości `H`:

```
θ̂_k(H) = (1/H) · Σ_{i=0}^{H-1} θ_{k-i}
```

Dla trendu liniowego `bias(H) = -α(H-1)/2`, `Var(H) = v/H`, więc

```
IMSE(H) = α²(H-1)²/4 + v/H,      H_opt ≈ (2v/α²)^(1/3)
```

Duże okno tłumi szum, ale estymata opóźnia się względem trendu. Optymalne `H` jest
kompromisem między tymi efektami. Program porównuje `IMSE(H)` wyznaczone metodą Monte Carlo
z wartościami teoretycznymi. Bada też rozrzut `H_opt` wyznaczonego z jednej realizacji
i zależność `H_opt` od `α` oraz `v`.

Parametry symulacji: `θ_0 = 1`, `N = 2000`, `M = 200` realizacji, `H = 1, ..., 100`,
ziarno `2026`. Wszystkie znajdują się w [src/config.py](src/config.py).

| Wariant | α | v |
|---|---|---|
| W1 | 0,01 | 0,5 |
| W2 | 0,01 | 10 |
| W3 | 0,1 | 0,5 |
| W4 | 0,1 | 10 |
| W0 | 0 | 10 |

## Instalacja

Wymagany Python ≥ 3.11.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Uruchomienie

```
python -m src.run_all     # wszystkie obliczenia i wykresy (ok. 1 min)
python -m pytest          # testy
```

Wyniki są powtarzalne. Każdy eksperyment dostaje własny strumień liczb losowych
z `numpy.random.SeedSequence(2026).spawn(...)`, więc kolejne uruchomienia dają identyczne pliki.

## Struktura

```
src/
  config.py        parametry
  dane.py          trend, zakłócenia trójkątne, pomiary, strumienie losowe
  estymator.py     okno przesuwne (skumulowana suma, wektorowo)
  teoria.py        bias², Var, IMSE, H_opt
  imse.py          IMSE metodą Monte Carlo
  eksperymenty.py  obliczenia i zapis tabel
  wykresy.py       wykresy na podstawie zapisanych tabel
  run_all.py       uruchomienie całości
tests/             testy pytest
results/           wyniki (tabele CSV, wykresy PNG 300 dpi i PDF, podsumowanie.json)
```

## Wyniki

| Wykres (`results/figures/`) | Tabele (`results/tables/`) | Zawartość |
|---|---|---|
| `E1_generator` | `E1_generator`, `E1_histogram` | histogram zakłóceń (10⁶ próbek, `v = 10`) na tle gęstości trójkątnej |
| `E2_przebiegi` | `E2_przebiegi` | W4, jedna realizacja: pomiary, `θ*_k` i estymaty dla `H = 1`, `H_opt`, `100` |
| `E3_imse_warianty` | `E3_imse_tabela`, `E3_imse_krzywe` | `IMSE(H)` empiryczne i teoretyczne dla W1–W4, minima i płaskość minimum |
| `E4_dekompozycja` | `E4_dekompozycja` | W2: `bias²`, `Var` i `IMSE` oraz punkt `bias² = Var` |
| `E5_realizacja`, `E5_histogram_hopt` | `E5_realizacja`, `E5_hopt_realizacji`, `E5_statystyki_hopt` | W4, `N = 500`: `IMSE(H)` jednej realizacji i rozkład `H_opt` z 500 realizacji |
| `E6_staly_parametr`, `E6_wariancja_N` | `E6_staly_parametr`, `E6_wariancja_N` | W0 (`α = 0`): `IMSE = v/H`, wariancja estymaty dla `H = N` wobec `v/N` |
| `E7_wrazliwosc` | `E7_wrazliwosc_alfa`, `E7_wrazliwosc_v`, `E7_nachylenia` | `H_opt` w funkcji `α` i `v` (log-log), nachylenia wobec `-2/3` i `1/3` |

Najważniejsze liczby ze wszystkich eksperymentów zebrano w `results/podsumowanie.json`.
