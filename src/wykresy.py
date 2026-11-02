"""Rysowanie wykresów na podstawie zapisanych tabel (bez obliczeń Monte Carlo)."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from src import config, dane  # noqa: E402

KOLOR_EMP = "tab:blue"
KOLOR_TEOR = "tab:orange"


# --- pomocnicze --------------------------------------------------------------

def liczba(x, miejsca=None, mat=False):
    """Liczba z przecinkiem dziesiętnym; `mat` - zapis do trybu matematycznego."""
    if abs(x) < 1e-12:
        x = 0.0
    s = f"{x:.{miejsca}f}" if miejsca is not None else f"{x:g}"
    if mat:
        return s.replace(".", "{,}")
    return s.replace(".", ",").replace("-", "−")


def _przecinek(ax):
    """Przecinek dziesiętny na osiach; na osiach log zwykłe liczby (x: 1-2-5, y: 1-2-3-5)."""
    for os_, subs in ((ax.xaxis, (1.0, 2.0, 5.0)), (ax.yaxis, (1.0, 2.0, 3.0, 5.0))):
        if os_.get_scale() == "log":
            os_.set_major_locator(LogLocator(subs=subs))
            os_.set_minor_formatter(NullFormatter())
        os_.set_major_formatter(FuncFormatter(lambda x, _: liczba(x)))


def _osie(ax, opis_x, opis_y):
    """Opisy osi, siatka i przecinek dziesiętny (bez tytułu, podpis jest w sprawozdaniu)."""
    ax.set_xlabel(opis_x)
    ax.set_ylabel(opis_y)
    ax.grid(True, linestyle=":", alpha=0.6)
    _przecinek(ax)


def _legenda(ax, **kwargs):
    """Legenda w prawym górnym rogu; jeśli zasłania dane, podnosi górną granicę osi Y."""
    legenda = ax.legend(loc="upper right", fontsize=9, title_fontsize=9, framealpha=0.9,
                        **kwargs)
    ax.figure.canvas.draw()
    do_osi = ax.transAxes.inverted()
    ramka = legenda.get_window_extent().transformed(do_osi)

    # punkty danych (linie i wykresy punktowe) we współrzędnych osi; pionowe linie pomocnicze
    # mają inną transformację, więc są pomijane
    punkty = [l.get_xydata() for l in ax.get_lines() if l.get_transform() == ax.transData]
    punkty += [k.get_offsets() for k in ax.collections if len(k.get_offsets())]
    if not punkty:
        return
    xy = np.concatenate([np.asarray(p, dtype=float) for p in punkty])
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    frac = do_osi.transform(ax.transData.transform(xy))
    pod_legenda = (frac[:, 0] >= ramka.x0 - 0.02) & (frac[:, 0] <= 1.0)
    granica = ramka.y0 - 0.03
    if not pod_legenda.any() or frac[pod_legenda, 1].max() <= granica:
        return

    skala = ax.yaxis.get_transform()
    dol, gora = skala.transform(np.array(ax.get_ylim()))
    szczyt = dol + frac[pod_legenda, 1].max() * (gora - dol)
    nowa_gora = dol + (szczyt - dol) / granica
    ax.set_ylim(top=skala.inverted().transform(np.array([nowa_gora]))[0])


def _adnotacja(ax, x, y, tekst, polozenie, **kwargs):
    """Opis w ramce ze strzałką do punktu (x, y); `polozenie` we współrzędnych osi."""
    ax.annotate(tekst, xy=(x, y), xytext=polozenie, textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
                bbox={"boxstyle": "round", "fc": "white", "ec": "gray", "alpha": 0.9},
                **kwargs)


def _minima(ax, h_emp, imse_emp, h_teor, imse_teor):
    ax.plot(h_emp, imse_emp, "o", color="red", markersize=8, label="minimum empiryczne")
    ax.plot(h_teor, imse_teor, "x", color="green", markersize=10, markeredgewidth=2,
            label="minimum teoretyczne")


def _opis_minimum(h, imse_min, przedrostek=""):
    return rf"{przedrostek}$H_{{\mathrm{{opt}}}}$ = {h}, $\mathrm{{MSE}}_{{\mathrm{{min}}}}$ = {liczba(imse_min, 4)}"


def wczytaj_csv(nazwa):
    """Wczytuje tabelę jako słownik kolumn (liczbowych, jeśli się da)."""
    with open(config.KATALOG_TABEL / f"{nazwa}.csv", encoding="utf-8") as f:
        wiersze = list(csv.reader(f))
    kolumny = {}
    for i, klucz in enumerate(wiersze[0]):
        wartosci = [w[i] for w in wiersze[1:]]
        try:
            kolumny[klucz] = np.array(wartosci, dtype=float)
        except ValueError:
            kolumny[klucz] = np.array(wartosci)
    return kolumny


def wczytaj_podsumowanie():
    return json.loads((config.KATALOG_WYNIKOW / "podsumowanie.json").read_text(encoding="utf-8"))


def zapisz(fig, nazwa):
    config.KATALOG_WYKRESOW.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # bez daty utworzenia w PDF, żeby kolejne uruchomienia dawały identyczne pliki
    fig.savefig(config.KATALOG_WYKRESOW / f"{nazwa}.png", dpi=config.DPI, bbox_inches="tight")
    fig.savefig(config.KATALOG_WYKRESOW / f"{nazwa}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    plt.close(fig)


# --- E1 ----------------------------------------------------------------------

def e1_generator():
    d = wczytaj_csv("E1_histogram")
    v = config.E1_V
    a = np.sqrt(6 * v)
    krawedzie = np.append(d["lewa_krawedz"], d["prawa_krawedz"][-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stairs(d["gestosc_emp"], krawedzie, fill=True, color=KOLOR_EMP, alpha=0.5,
              label=f"histogram ({config.E1_PROBKI:,} próbek)".replace(",", " "))
    z = np.linspace(-1.08 * a, 1.08 * a, 801)
    ax.plot(z, dane.gestosc(z, v), "--", color=KOLOR_TEOR, linewidth=1.6,
            label=r"gęstość teoretyczna $f(z) = (a-|z|)/a^2$")
    ax.axvline(-a, color="gray", linestyle=":", linewidth=1.2)
    ax.axvline(a, color="gray", linestyle=":", linewidth=1.2,
               label=rf"$\pm a = \pm\sqrt{{6v}} \approx {{\pm}}{liczba(a, 2, mat=True)}$")
    ax.set_ylim(0, 1.3 / a)
    _osie(ax, r"Wartość zakłócenia $z_k$", "Gęstość prawdopodobieństwa")
    _legenda(ax)
    zapisz(fig, "E1_generator")


# --- E2 ----------------------------------------------------------------------

def e2_przebiegi():
    d = wczytaj_csv("E2_przebiegi")
    p = wczytaj_podsumowanie()["E2"]
    h_1, h_opt, h_max = p["H"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(d["k"], d["theta"], color="gray", alpha=0.6, s=18, zorder=1, label=r"Pomiary $\theta_k$")
    ax.plot(d["k"], d["theta_gw"], "--", color="black", linewidth=2, zorder=4,
            label=r"Prawdziwy parametr $\theta^*_k = \theta_0 + \alpha k$")
    ax.plot(d["k"], d[f"est_H{h_1}"], color="tab:green", linewidth=1.3, alpha=0.75, zorder=2,
            label=rf"$\hat{{\theta}}_k$, $H = {h_1}$: dominacja szumu (niedofiltracja)")
    ax.plot(d["k"], d[f"est_H{h_opt}"], color="tab:blue", linewidth=3, zorder=5,
            label=rf"$\hat{{\theta}}_k$, $H = H_{{\mathrm{{opt}}}} = {h_opt}$: kompromis")
    ax.plot(d["k"], d[f"est_H{h_max}"], color="tab:red", linewidth=2.6, zorder=3,
            label=rf"$\hat{{\theta}}_k$, $H = {h_max}$: dominacja opóźnienia (nadfiltracja)")
    ax.set_xlim(d["k"][0], d["k"][-1])
    _osie(ax, r"Chwila $k$", "Wartość parametru")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    zapisz(fig, "E2_przebiegi")


# --- E3 ----------------------------------------------------------------------

def _panele_e3(z_teoria):
    d = wczytaj_csv("E3_imse_krzywe")
    tab = wczytaj_csv("E3_imse_tabela")
    H = d["H"]

    fig, osie = plt.subplots(2, 2, figsize=(12, 8))
    for ax, i in zip(osie.flat, range(len(tab["wariant"]))):
        nazwa = tab["wariant"][i]
        emp, teor = d[f"{nazwa}_imse_emp"], d[f"{nazwa}_imse_teor"]
        h_e, imse_e = int(tab["H_opt_emp"][i]), tab["IMSE_min_emp"][i]
        h_t, imse_t = int(tab["H_opt_teor"][i]), tab["IMSE_min_teor"][i]

        ax.plot(H, emp, color=KOLOR_EMP, linewidth=1.8, label="MSE empiryczne")
        if z_teoria:
            ax.plot(H, teor, "--", color=KOLOR_TEOR, linewidth=1.6, label="MSE teoretyczne")
            ax.plot(h_e, imse_e, "o", color=KOLOR_EMP, markersize=7, zorder=5,
                    label="minimum empiryczne")
            ax.plot(h_t, imse_t, "x", color=KOLOR_TEOR, zorder=6, markersize=8, markeredgewidth=2,
                    label="minimum teoretyczne")
            tekst = (rf"emp.: $H_{{\mathrm{{opt}}}}$ = {h_e}, $\mathrm{{MSE}}_{{\mathrm{{min}}}}$ = {liczba(imse_e, 4)}" + "\n"
                     + rf"teor.: $H_{{\mathrm{{opt}}}}$ = {h_t}, $\mathrm{{MSE}}_{{\mathrm{{min}}}}$ = {liczba(imse_t, 4)}")
        else:
            ax.plot(h_e, imse_e, "o", color=KOLOR_EMP, markersize=7, zorder=5,
                    label="minimum empiryczne")
            tekst = rf"$H_{{\mathrm{{opt}}}}$ = {h_e}, $\mathrm{{MSE}}_{{\mathrm{{min}}}}$ = {liczba(imse_e, 4)}"
        ax.annotate(tekst, xy=(h_e, imse_e), xytext=(0.14, 0.40), textcoords="axes fraction",
                    fontsize=8.5, arrowprops={"arrowstyle": "->", "color": "black", "lw": 1})
        ax.set_title(rf"{nazwa}: $\alpha$ = {liczba(tab['alfa'][i])}, "
                     rf"$v$ = {liczba(tab['v'][i])}", fontsize=12)
        ax.set_xlim(0, H[-1] + 1)
        _osie(ax, r"Horyzont $H$", "MSE")
        _legenda(ax)
    fig.tight_layout()
    return fig


def e3_warianty():
    zapisz(_panele_e3(z_teoria=True), "E3_imse_warianty")


def e3_imse_empiryczne():
    zapisz(_panele_e3(z_teoria=False), "E3_imse_empiryczne")


# --- E4 ----------------------------------------------------------------------

def e4_dekompozycja():
    d = wczytaj_csv("E4_dekompozycja")
    p = wczytaj_podsumowanie()["E4"]
    H = d["H"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(H, d["bias2_teor"], "-.", color="tab:red", linewidth=2,
            label=r"Kwadrat obciążenia: $\mathrm{bias}^2(H) = \alpha^2 (H-1)^2 / 4$")
    ax.plot(H, d["var_teor"], "--", color="tab:green", linewidth=2,
            label=r"Wariancja: $\mathrm{Var}(H) = v / H$")
    ax.plot(H, d["imse_teor"], color="tab:blue", linewidth=2.5,
            label=r"Całkowity błąd: $\mathrm{MSE}(H) = \mathrm{bias}^2(H) + \mathrm{Var}(H)$")

    h_x = p["H_przeciecia_teor"]
    y_x = p["v"] / h_x
    ax.plot(h_x, y_x, "s", color="black", markersize=7,
            label=rf"Przecięcie składowych: $H \approx {liczba(h_x, 1, mat=True)}$")
    ax.plot(p["H_opt_teor"], p["IMSE_min_teor"], "o", color="tab:blue", markersize=7)
    _adnotacja(ax, p["H_opt_teor"], p["IMSE_min_teor"],
               _opis_minimum(p["H_opt_teor"], p["IMSE_min_teor"]), (0.45, 0.55), fontsize=10)

    ax.set_xlim(1, config.H_MAX)
    ax.set_ylim(0, 4 * p["IMSE_min_teor"])
    _osie(ax, r"Horyzont $H$", "Wartość składowej")
    _legenda(ax)
    zapisz(fig, "E4_dekompozycja")


# --- E5 ----------------------------------------------------------------------

def e5_realizacja():
    d = wczytaj_csv("E5_realizacja")
    p = wczytaj_podsumowanie()["E5"]
    H = d["H"]
    h_e, imse_e = p["H_opt_jednej_realizacji"], p["IMSE_min_jednej_realizacji"]
    h_t, imse_t = p["H_opt_teor"], p["IMSE_min_teor"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(H, d["imse_jednej_realizacji"], "o", color=KOLOR_EMP, markersize=3.5,
            label="MSE jednej realizacji")
    ax.plot(H, d["imse_teor"], "--", color=KOLOR_TEOR, linewidth=1.6, label="MSE teoretyczne")
    _minima(ax, h_e, imse_e, h_t, imse_t)
    _adnotacja(ax, h_e, imse_e,
               _opis_minimum(h_e, imse_e, "jedna realizacja: ") + "\n"
               + _opis_minimum(h_t, imse_t, "teoria: "), (0.25, 0.55), fontsize=10)
    ax.set_xlim(0, H[-1] + 1)
    ax.set_ylim(0, 1.3 * d["imse_teor"].max())
    _osie(ax, r"Horyzont $H$", "MSE")
    _legenda(ax)
    zapisz(fig, "E5_realizacja")


def e5_histogram_hopt():
    d = wczytaj_csv("E5_hopt_realizacji")
    p = wczytaj_podsumowanie()["E5"]
    s = p["statystyki_H_opt"]
    h = d["H_opt"].astype(int)
    krawedzie = np.arange(h.min() - 0.5, h.max() + 1.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(s["percentyl_5"], s["percentyl_95"], color="gray", alpha=0.15,
               label=f"percentyle 5–95: [{liczba(s['percentyl_5'])}; "
                     f"{liczba(s['percentyl_95'])}]")
    ax.hist(h, bins=krawedzie, color=KOLOR_EMP, alpha=0.7, edgecolor="black",
            label=rf"$H_{{\mathrm{{opt}}}}$ z {p['realizacje']} realizacji")
    ax.axvline(p["H_opt_teor"], color="green", linestyle="--", linewidth=2,
               label=rf"$H_{{\mathrm{{opt}}}}$ teoretyczne: {p['H_opt_teor']}")
    ax.axvline(s["srednia"], color="red", linestyle=":", linewidth=2,
               label=f"średnia: {liczba(s['srednia'], 2)}, mediana: {liczba(s['mediana'])}")
    ax.set_xticks(np.arange(h.min(), h.max() + 1))
    ax.set_ylim(0, 1.3 * np.bincount(h).max())
    _osie(ax, r"Horyzont $H_{\mathrm{opt}}$", "Liczba realizacji")
    _legenda(ax)
    zapisz(fig, "E5_histogram_hopt")


# --- E6 ----------------------------------------------------------------------

def e6_staly_parametr():
    d = wczytaj_csv("E6_staly_parametr")
    p = wczytaj_podsumowanie()["E6"]
    H = d["H"]
    h_e = p["H_opt_emp"]
    imse_e = d["imse_emp"][H == h_e][0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(H, d["imse_emp"], color=KOLOR_EMP, linewidth=1.8, label="MSE empiryczne")
    ax.plot(H, d["imse_teor"], "--", color=KOLOR_TEOR, linewidth=1.6,
            label=r"MSE teoretyczne $v/H$")
    _minima(ax, h_e, imse_e, p["H_opt_teor"], d["imse_teor"][-1])
    _adnotacja(ax, h_e, imse_e,
               _opis_minimum(h_e, imse_e, "emp.: ") + "\n"
               + _opis_minimum(p["H_opt_teor"], d["imse_teor"][-1], r"teor. ($H_{\mathrm{max}}$): "),
               (0.45, 0.35), fontsize=10)
    ax.set_xlim(0, H[-1] + 1)
    ax.set_ylim(0, 1.05 * d["imse_emp"].max())
    _osie(ax, r"Horyzont $H$", "MSE")
    _legenda(ax)
    zapisz(fig, "E6_staly_parametr")


def e6_wariancja_N():
    d = wczytaj_csv("E6_wariancja_N")
    p = wczytaj_podsumowanie()["E6"]
    N_gesto = np.geomspace(d["N"][0] * 0.8, d["N"][-1] * 1.25, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d["N"], d["var_emp"], "o", color=KOLOR_EMP, markersize=8,
            label=rf"wariancja empiryczna, $M$ = {p['M_b']}, "
                  rf"nachylenie {liczba(p['nachylenie_log_var_od_log_N'], 3)}")
    ax.plot(N_gesto, p["v"] / N_gesto, "--", color=KOLOR_TEOR, linewidth=1.6,
            label=rf"teoria $v/N$, nachylenie $-1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _osie(ax, "Liczba próbek N", r"$\mathrm{Var}[\hat{\theta}_{N-1}(N)]$")
    _legenda(ax)
    zapisz(fig, "E6_wariancja_N")


# --- E7 ----------------------------------------------------------------------

def e7_wrazliwosc():
    p = wczytaj_podsumowanie()["E7"]
    fig, osie = plt.subplots(1, 2, figsize=(12, 5.5))
    opisy = {
        "alfa": (r"$\alpha$", rf"$v$ = {liczba(config.E7_V_PRZY_ALPHA)}", "-2/3"),
        "v": (r"$v$", rf"$\alpha$ = {liczba(config.E7_ALPHA_PRZY_V)}", "1/3"),
    }
    for ax, (seria, (symbol, stala, wykl)) in zip(osie, opisy.items()):
        d = wczytaj_csv(f"E7_wrazliwosc_{seria}")
        w = p[seria]
        x = d[seria]
        x_gesto = np.geomspace(x.min() * 0.85, x.max() * 1.15, 100)
        alpha = d["alfa"][0] if seria == "v" else x_gesto
        v = d["v"][0] if seria == "alfa" else x_gesto
        wsp = np.polyfit(np.log(x), np.log(d["H_opt_emp"]), 1)

        ax.plot(x_gesto, (2 * v / alpha**2) ** (1 / 3), "--", color=KOLOR_TEOR, linewidth=1.6,
                label=rf"$(2v/\alpha^2)^{{1/3}}$, nachylenie ${wykl}$")
        ax.plot(x_gesto, np.exp(np.polyval(wsp, np.log(x_gesto))), color=KOLOR_EMP,
                linewidth=1.4,
                label=f"MNK (emp.), nachylenie {liczba(w['nachylenie_emp'], 3)}")
        ax.plot(x, d["H_opt_emp"], "o", color="red", markersize=8,
                label=r"$H_{\mathrm{opt}}$ empiryczne")
        ax.plot(x, d["H_opt_teor"], "x", color="green", markersize=10, markeredgewidth=2,
                label=rf"$H_{{\mathrm{{opt}}}}$ teoretyczne, nachylenie "
                      f"{liczba(w['nachylenie_teor_calkowite'], 3)}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _osie(ax, f"{symbol} (przy {stala})", r"$H_{\mathrm{opt}}$")
        _legenda(ax)
        dol, gora = ax.get_ylim()
        podzialka = np.array([3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 80, 100])
        ax.set_yticks(podzialka[(podzialka >= dol) & (podzialka <= gora)])
    zapisz(fig, "E7_wrazliwosc")
