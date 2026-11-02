# Estymacja parametru zmiennego liniowo za pomocą okna przesuwnego

Projekt nr 1 z przedmiotu *Sterowanie adaptacyjne*, grupa C.

Program bada, jak szerokość okna $H$ wpływa na błąd estymacji parametru, który zmienia się liniowo
i jest mierzony z zakłóceniem o rozkładzie trójkątnym. Wyniki symulacji Monte Carlo są porównywane
z zależnościami teoretycznymi.

## Model

Prawdziwy parametr i jego pomiar, dla $k = 0, 1, \dots, N-1$:

$$
\theta_k^* = \theta_0 + \alpha k, \qquad \theta_k = \theta_k^* + z_k
$$

Zakłócenia $z_k$ są niezależne, a ich rozkład jest trójkątny i symetryczny:

$$
\mathbb{E}[z_k] = 0, \qquad \mathrm{Var}[z_k] = v, \qquad
f(z) = \frac{a - |z|}{a^2} \quad \text{dla } |z| \le a, \qquad a = \sqrt{6v}
$$

Program generuje je z sumy dwóch rozkładów jednostajnych $U_1, U_2 \sim \mathcal{U}[0, 1)$:

$$
z_k = \sqrt{v} \cdot \sqrt{6} \cdot (U_1 + U_2 - 1)
$$

Estymata to średnia z ostatnich $H$ pomiarów:

$$
\hat{\theta}_k(H) = \frac{1}{H} \sum_{i=0}^{H-1} \theta_{k-i}, \qquad k \ge H - 1
$$

## Zależności teoretyczne

Duże okno tłumi szum, ale estymata opóźnia się względem trendu. Optymalne $H$ jest kompromisem
między tymi efektami:

$$
\mathrm{bias}(H) = -\frac{\alpha (H-1)}{2}, \qquad \mathrm{Var}(H) = \frac{v}{H}
$$

$$
\mathrm{MSE}(H) = \frac{\alpha^2 (H-1)^2}{4} + \frac{v}{H}, \qquad
H_{\mathrm{opt}} \approx \left( \frac{2v}{\alpha^2} \right)^{1/3}
$$

Dokładne $H_{\mathrm{opt}}$ program wyznacza jako minimum $\mathrm{MSE}(H)$ po całkowitych $H$.
Dla stałego parametru ($\alpha = 0$) obciążenie znika, a dla $H = N$ wariancja wynosi $v/N$
i dąży do zera, gdy $N \to \infty$.

Empiryczne $\mathrm{MSE}$ jest liczone dla $M$ realizacji, na tym samym zbiorze chwil $k$ dla każdego $H$:

$$
\widehat{\mathrm{MSE}}(H) = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{N - H_{\max} + 1}
\sum_{k=H_{\max}-1}^{N-1} \left( \hat{\theta}_k^{(m)}(H) - \theta_k^* \right)^2
$$

## Parametry

| Parametr | Symbol | Wartość |
|:---|:---:|:---:|
| Wartość początkowa | $\theta_0$ | $1$ |
| Liczba próbek | $N$ | $2000$ |
| Liczba realizacji | $M$ | $200$ |
| Szerokość okna | $H$ | $1, 2, \dots, 100$ |
| Największe okno | $H_{\max}$ | $100$ |
| Ziarno generatora | | $2026$ |

Wszystkie parametry znajdują się w [src/config.py](src/config.py).

## Wyniki główne

| Wariant | $\alpha$ | $v$ | $H_{\mathrm{opt}}$ teor. | $H_{\mathrm{opt}}$ emp. | $\mathrm{MSE}_{\min}$ teor. | $\mathrm{MSE}_{\min}$ emp. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| W1 | $0{,}01$ | $0{,}5$ | $22$ | $22$ | $0{,}0338$ | $0{,}0340$ |
| W2 | $0{,}01$ | $10$ | $59$ | $59$ | $0{,}2536$ | $0{,}2565$ |
| W3 | $0{,}1$ | $0{,}5$ | $5$ | $5$ | $0{,}1400$ | $0{,}1403$ |
| W4 | $0{,}1$ | $10$ | $13$ | $13$ | $1{,}1292$ | $1{,}1196$ |

Dodatkowo badany jest wariant W0 ze stałym parametrem: $\alpha = 0$, $v = 10$.
Pełne wyniki, tabele i wykresy znajdują się w [results/](results/).

## Instalacja

Wymagany Python 3.11 lub nowszy.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Uruchomienie

```
python -m src.run_all     # obliczenia, tabele i wykresy
python -m pytest          # testy
```

Wyniki są powtarzalne, bo każdy eksperyment dostaje własny strumień liczb losowych
z `numpy.random.SeedSequence(2026).spawn(...)`.

## Struktura

```
src/
  config.py        parametry
  dane.py          trend, zakłócenia trójkątne, pomiary, strumienie losowe
  estymator.py     okno przesuwne (skumulowana suma, wektorowo)
  teoria.py        bias², Var, MSE, H_opt
  imse.py          MSE metodą Monte Carlo
  eksperymenty.py  obliczenia i zapis tabel
  wykresy.py       wykresy na podstawie zapisanych tabel
  run_all.py       uruchomienie całości
tests/             testy pytest
results/           tabele CSV, wykresy PNG i PDF, podsumowanie.json
```