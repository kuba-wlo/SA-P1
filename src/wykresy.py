"""Rysowanie wykresów na podstawie zapisanych tabel (bez obliczeń Monte Carlo)."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from src import config, dane  # noqa: E402

SZEROKOSC = 6.3  # cale, szerokość kolumny tekstu A4
KOLOR = {
    "niebieski": "#2a78d6", "pomaranczowy": "#eb6834", "morski": "#1baf7a",
    "czarny": "#0b0b0b", "szary": "#52514e", "jasny": "#b8b7b2", "siatka": "#e3e2de",
}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": KOLOR["siatka"], "grid.linewidth": 0.6,
    "axes.edgecolor": KOLOR["szary"], "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False,
    "savefig.dpi": config.DPI, "mathtext.fontset": "dejavusans",
})


# --- pomocnicze --------------------------------------------------------------

def liczba(x, miejsca=None, mat=True):
    """Liczba z przecinkiem dziesiętnym; `mat` - zapis do trybu matematycznego."""
    if abs(x) < 1e-12:
        x = 0.0
    s = f"{x:.{miejsca}f}" if miejsca is not None else f"{x:g}"
    return s.replace(".", "{,}" if mat else ",")


def _przecinek(ax):
    """Przecinek dziesiętny na osiach; na osiach log zwykłe liczby 1-2-5."""
    for os_ in (ax.xaxis, ax.yaxis):
        if os_.get_scale() == "log":
            os_.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            os_.set_minor_formatter(NullFormatter())
        os_.set_major_formatter(FuncFormatter(lambda x, _: liczba(x, mat=False)))


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
    for rozszerzenie in ("png", "pdf"):
        fig.savefig(config.KATALOG_WYKRESOW / f"{nazwa}.{rozszerzenie}", bbox_inches="tight")
    plt.close(fig)


# --- E1 ----------------------------------------------------------------------

def e1_generator():
    d = wczytaj_csv("E1_histogram")
    v = config.E1_V
    a = np.sqrt(6 * v)
    krawedzie = np.append(d["lewa_krawedz"], d["prawa_krawedz"][-1])

    fig, ax = plt.subplots(figsize=(SZEROKOSC, 3.2))
    ax.stairs(d["gestosc_emp"], krawedzie, fill=True, color=KOLOR["niebieski"], alpha=0.35,
              label=f"histogram ($10^{{{int(np.log10(config.E1_PROBKI))}}}$ próbek)")
    ax.stairs(d["gestosc_emp"], krawedzie, color=KOLOR["niebieski"], linewidth=0.8)
    z = np.linspace(-1.08 * a, 1.08 * a, 801)
    ax.plot(z, dane.gestosc(z, v), color=KOLOR["czarny"], linestyle="--",
            label="gęstość teoretyczna")
    for znak in (-1, 1):
        ax.axvline(znak * a, color=KOLOR["szary"], linestyle=":", linewidth=1)
    ax.text(a, 1.02 / a, rf"$a=\sqrt{{6v}}\approx{liczba(a, 2)}$", ha="right", va="bottom")
    ax.set_xlabel(r"$z_k$")
    ax.set_ylabel(r"gęstość prawdopodobieństwa")
    ax.set_ylim(0, 1.3 / a)
    ax.legend(loc="upper left")
    _przecinek(ax)
    zapisz(fig, "E1_generator")


# --- E2 ----------------------------------------------------------------------

def e2_przebiegi():
    d = wczytaj_csv("E2_przebiegi")
    H_lista = sorted(int(k[len("est_H"):]) for k in d if k.startswith("est_H"))
    h_1, h_opt, h_max = H_lista

    fig, ax = plt.subplots(figsize=(SZEROKOSC, 3.6))
    ax.plot(d["k"], d["theta"], linestyle="none", marker="o", markersize=2.2,
            color=KOLOR["jasny"], label=r"pomiary $\theta_k$")
    ax.plot(d["k"], d[f"est_H{h_1}"], color=KOLOR["niebieski"], linewidth=0.6, alpha=0.8,
            label=rf"$\hat{{\theta}}_k$, $H={h_1}$")
    ax.plot(d["k"], d["theta_gw"], color=KOLOR["czarny"], linewidth=1.4,
            label=r"$\theta^*_k$")
    ax.plot(d["k"], d[f"est_H{h_opt}"], color=KOLOR["pomaranczowy"], linestyle="--",
            linewidth=1.6, label=rf"$\hat{{\theta}}_k$, $H=H_{{opt}}={h_opt}$")
    ax.plot(d["k"], d[f"est_H{h_max}"], color=KOLOR["morski"], linestyle="-.",
            linewidth=1.6, label=rf"$\hat{{\theta}}_k$, $H={h_max}$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\theta$")
    ax.set_xlim(d["k"][0], d["k"][-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=5, handlelength=2.2,
              columnspacing=1.2)
    _przecinek(ax)
    zapisz(fig, "E2_przebiegi")


# --- E3 ----------------------------------------------------------------------

def e3_warianty():
    d = wczytaj_csv("E3_imse_krzywe")
    tab = wczytaj_csv("E3_imse_tabela")
    H = d["H"]

    fig, osie = plt.subplots(2, 2, figsize=(SZEROKOSC, 5.0), sharex=True)
    for ax, i in zip(osie.flat, range(len(tab["wariant"]))):
        nazwa = tab["wariant"][i]
        emp, teor = d[f"{nazwa}_imse_emp"], d[f"{nazwa}_imse_teor"]
        ax.plot(H, emp, color=KOLOR["niebieski"], marker="o", markersize=2.5, markevery=4,
                linewidth=1.0)
        ax.plot(H, teor, color=KOLOR["czarny"], linestyle="--", linewidth=1.2)
        ax.plot(tab["H_opt_emp"][i], tab["IMSE_min_emp"][i], linestyle="none", marker="o",
                markersize=8, markerfacecolor="none", markeredgecolor=KOLOR["niebieski"],
                markeredgewidth=1.5)
        ax.plot(tab["H_opt_teor"][i], tab["IMSE_min_teor"][i], linestyle="none", marker="x",
                markersize=7, color=KOLOR["czarny"], markeredgewidth=1.5)
        ax.set_yscale("log")
        rosnie = emp[-1] > emp[0]
        ax.text(0.97 if rosnie else 0.5, 0.04 if rosnie else 0.97,
                rf"{nazwa}: $\alpha={liczba(tab['alfa'][i])}$, $v={liczba(tab['v'][i])}$"
                "\n" rf"$H_{{opt}}$: teor. {int(tab['H_opt_teor'][i])}, "
                rf"emp. {int(tab['H_opt_emp'][i])}",
                transform=ax.transAxes, ha="right" if rosnie else "center",
                va="bottom" if rosnie else "top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
        _przecinek(ax)
    for ax in osie[1]:
        ax.set_xlabel(r"$H$")
    for ax in osie[:, 0]:
        ax.set_ylabel(r"IMSE$(H)$")
    osie[0, 0].set_xlim(0, H[-1] + 1)

    uchwyty = [
        Line2D([], [], color=KOLOR["niebieski"], marker="o", markersize=3, linewidth=1.0,
               label="IMSE empiryczne"),
        Line2D([], [], color=KOLOR["czarny"], linestyle="--", label="IMSE teoretyczne"),
        Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor="none",
               markeredgecolor=KOLOR["niebieski"], markeredgewidth=1.5,
               label="minimum empiryczne"),
        Line2D([], [], linestyle="none", marker="x", markersize=6, color=KOLOR["czarny"],
               markeredgewidth=1.5, label="minimum teoretyczne"),
    ]
    fig.legend(handles=uchwyty, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.tight_layout()
    zapisz(fig, "E3_imse_warianty")


# --- E4 ----------------------------------------------------------------------

def e4_dekompozycja():
    d = wczytaj_csv("E4_dekompozycja")
    p = wczytaj_podsumowanie()["E4"]
    H = d["H"]
    co = 3

    fig, ax = plt.subplots(figsize=(SZEROKOSC, 3.8))
    serie = [
        ("imse", "IMSE$(H)$", KOLOR["czarny"], "-", "o"),
        ("bias2", r"$\mathrm{bias}^2(H)$", KOLOR["pomaranczowy"], "--", "s"),
        ("var", r"$\mathrm{Var}(H)$", KOLOR["morski"], "-.", "^"),
    ]
    for klucz, opis, kolor, styl, znacznik in serie:
        ax.plot(H, d[f"{klucz}_teor"], color=kolor, linestyle=styl,
                label=f"{opis} teoretycznie")
        ax.plot(H[::co], d[f"{klucz}_emp"][::co], linestyle="none", marker=znacznik,
                markersize=4, markerfacecolor="none", markeredgecolor=kolor,
                label=f"{opis} empirycznie")

    ax.plot(H, d["bias2_teor"] + d["var_teor"] / config.M, color=KOLOR["szary"],
            linestyle=":", linewidth=1.2,
            label=rf"$\mathrm{{bias}}^2(H)+\mathrm{{Var}}(H)/M$, $M={config.M}$")

    h_opt, imse_min = p["H_opt_teor"], p["IMSE_min_teor"]
    h_x = p["H_przeciecia_teor"]
    ax.axvline(h_opt, color=KOLOR["szary"], linestyle=":", linewidth=1)
    ax.axvline(h_x, color=KOLOR["szary"], linestyle=":", linewidth=1)
    ax.plot(h_opt, imse_min, marker="x", markersize=8, markeredgewidth=1.6,
            color=KOLOR["czarny"], linestyle="none")
    ax.text(h_opt - 1, 2.2, rf"$H_{{opt}}={h_opt}$" "\n"
            rf"(emp. {p['H_opt_emp']})", ha="right", va="bottom")
    ax.text(h_x + 1, 2.2, r"$\mathrm{bias}^2=\mathrm{Var}$" "\n"
            rf"$H\approx{liczba(h_x, 1)}$", ha="left", va="bottom")

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 20)
    ax.set_xlim(0, H[-1] + 1)
    ax.set_xlabel(r"$H$")
    ax.set_ylabel("wartość składowej")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=3, handlelength=2.2)
    _przecinek(ax)
    zapisz(fig, "E4_dekompozycja")
