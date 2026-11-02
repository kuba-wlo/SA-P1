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
    """Przecinek dziesiętny na osiach; na osiach log zwykłe liczby (x: 1-2-5, y: 1-2-3-5)."""
    for os_, subs in ((ax.xaxis, (1.0, 2.0, 5.0)), (ax.yaxis, (1.0, 2.0, 3.0, 5.0))):
        if os_.get_scale() == "log":
            os_.set_major_locator(LogLocator(subs=subs))
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


# --- E5 ----------------------------------------------------------------------

def e5_realizacja():
    d = wczytaj_csv("E5_realizacja")
    p = wczytaj_podsumowanie()["E5"]
    H = d["H"]

    fig, ax = plt.subplots(figsize=(SZEROKOSC, 3.4))
    ax.plot(H, d["imse_jednej_realizacji"], linestyle="none", marker="o", markersize=3,
            markerfacecolor="none", markeredgecolor=KOLOR["niebieski"],
            label="IMSE$(H)$ jednej realizacji")
    ax.plot(H, d["imse_teor"], color=KOLOR["czarny"], linestyle="--",
            label="IMSE$(H)$ teoretyczne")
    ax.plot(p["H_opt_jednej_realizacji"], p["IMSE_min_jednej_realizacji"], linestyle="none",
            marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor=KOLOR["pomaranczowy"], markeredgewidth=1.6,
            label=rf"minimum jednej realizacji, $H={p['H_opt_jednej_realizacji']}$")
    ax.plot(p["H_opt_teor"], p["IMSE_min_teor"], linestyle="none", marker="x", markersize=8,
            color=KOLOR["czarny"], markeredgewidth=1.6,
            label=rf"minimum teoretyczne, $H_{{opt}}={p['H_opt_teor']}$")
    ax.set_yscale("log")
    ax.set_xlim(0, H[-1] + 1)
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"IMSE$(H)$")
    ax.legend(loc="lower right")
    _przecinek(ax)
    zapisz(fig, "E5_realizacja")


def e5_histogram_hopt():
    d = wczytaj_csv("E5_hopt_realizacji")
    p = wczytaj_podsumowanie()["E5"]
    s = p["statystyki_H_opt"]
    h = d["H_opt"].astype(int)
    krawedzie = np.arange(h.min() - 0.5, h.max() + 1.5)

    fig, ax = plt.subplots(figsize=(SZEROKOSC, 3.2))
    ax.hist(h, bins=krawedzie, color=KOLOR["niebieski"], alpha=0.45,
            edgecolor=KOLOR["niebieski"], linewidth=0.8,
            label=rf"$H_{{opt}}$ z {p['realizacje']} realizacji ($N={p['N']}$)")
    ax.axvline(p["H_opt_teor"], color=KOLOR["czarny"], linestyle="--",
               label=rf"$H_{{opt}}$ teoretyczne: {p['H_opt_teor']}")
    ax.axvline(s["srednia"], color=KOLOR["pomaranczowy"], linestyle=":", linewidth=1.8,
               label=f"średnia: {liczba(s['srednia'], 2, mat=False)}")
    ax.axvspan(s["percentyl_5"], s["percentyl_95"], color=KOLOR["jasny"], alpha=0.25,
               label=f"percentyle 5–95: [{liczba(s['percentyl_5'], mat=False)}; "
                     f"{liczba(s['percentyl_95'], mat=False)}]")
    ax.set_xlabel(r"$H_{opt}$ wyznaczone z jednej realizacji")
    ax.set_ylabel("liczba realizacji")
    ax.set_xticks(np.arange(h.min(), h.max() + 1))
    ax.legend(loc="upper right")
    _przecinek(ax)
    zapisz(fig, "E5_histogram_hopt")


# --- E6 ----------------------------------------------------------------------

def e6_staly_parametr():
    d = wczytaj_csv("E6_staly_parametr")
    p = wczytaj_podsumowanie()["E6"]
    H = d["H"]

    fig, ax = plt.subplots(figsize=(SZEROKOSC, 3.4))
    ax.plot(H, d["imse_emp"], linestyle="none", marker="o", markersize=3.5,
            markerfacecolor="none", markeredgecolor=KOLOR["niebieski"],
            label="IMSE$(H)$ empiryczne")
    ax.plot(H, d["imse_teor"], color=KOLOR["czarny"], linestyle="--",
            label=rf"$v/H$, $v={liczba(p['v'])}$")
    ax.plot(p["H_opt_emp"], d["imse_emp"][H == p["H_opt_emp"]][0], linestyle="none",
            marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor=KOLOR["pomaranczowy"], markeredgewidth=1.6,
            label=rf"minimum empiryczne, $H={p['H_opt_emp']}$")
    ax.plot(p["H_opt_teor"], d["imse_teor"][-1], linestyle="none", marker="x", markersize=8,
            color=KOLOR["czarny"], markeredgewidth=1.6,
            label=rf"$H_{{opt}}=H_{{max}}={p['H_opt_teor']}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"IMSE$(H)$")
    ax.legend(loc="upper right")
    _przecinek(ax)
    zapisz(fig, "E6_staly_parametr")


def e6_wariancja_N():
    d = wczytaj_csv("E6_wariancja_N")
    p = wczytaj_podsumowanie()["E6"]
    N_gesto = np.geomspace(d["N"][0] * 0.8, d["N"][-1] * 1.25, 100)

    fig, ax = plt.subplots(figsize=(SZEROKOSC, 3.4))
    ax.plot(N_gesto, p["v"] / N_gesto, color=KOLOR["czarny"], linestyle="--",
            label=rf"$v/N$, $v={liczba(p['v'])}$")
    ax.plot(d["N"], d["var_emp"], linestyle="none", marker="o", markersize=6,
            markerfacecolor="none", markeredgecolor=KOLOR["niebieski"], markeredgewidth=1.4,
            label=rf"wariancja empiryczna ($M={p['M_b']}$), "
                  rf"nachylenie ${liczba(p['nachylenie_log_var_od_log_N'], 3)}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$ ($H=N$)")
    ax.set_ylabel(r"$\mathrm{Var}[\hat{\theta}_{N-1}(N)]$")
    ax.legend(loc="upper right")
    _przecinek(ax)
    zapisz(fig, "E6_wariancja_N")


# --- E7 ----------------------------------------------------------------------

def e7_wrazliwosc():
    p = wczytaj_podsumowanie()["E7"]
    fig, osie = plt.subplots(1, 2, figsize=(SZEROKOSC, 3.3))
    opisy = {
        "alfa": (r"$\alpha$", rf"$v={liczba(2.0)}$", "-2/3"),
        "v": (r"$v$", rf"$\alpha={liczba(0.05)}$", "1/3"),
    }
    for ax, (seria, (os_x, stala, wykl)) in zip(osie, opisy.items()):
        d = wczytaj_csv(f"E7_wrazliwosc_{seria}")
        w = p[seria]
        x = d[seria]
        x_gesto = np.geomspace(x.min() * 0.85, x.max() * 1.15, 100)
        alpha = d["alfa"][0] if seria == "v" else x_gesto
        v = d["v"][0] if seria == "alfa" else x_gesto
        ax.plot(x_gesto, (2 * v / alpha**2) ** (1 / 3), color=KOLOR["szary"], linestyle=":",
                linewidth=1.3, label=rf"$(2v/\alpha^2)^{{1/3}}$, nachyl. ${wykl}$")
        wsp = np.polyfit(np.log(x), np.log(d["H_opt_emp"]), 1)
        ax.plot(x_gesto, np.exp(np.polyval(wsp, np.log(x_gesto))), color=KOLOR["niebieski"],
                linewidth=1.1,
                label=rf"MNK (emp.), nachyl. ${liczba(w['nachylenie_emp'], 3)}$")
        ax.plot(x, d["H_opt_emp"], linestyle="none", marker="o", markersize=6,
                markerfacecolor="none", markeredgecolor=KOLOR["niebieski"],
                markeredgewidth=1.4, label=r"$H_{opt}$ empiryczne")
        ax.plot(x, d["H_opt_teor"], linestyle="none", marker="x", markersize=6,
                color=KOLOR["czarny"], markeredgewidth=1.3,
                label=rf"$H_{{opt}}$ teor. (całk.), nachyl. "
                      rf"${liczba(w['nachylenie_teor_calkowite'], 3)}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{os_x} (przy {stala})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        _przecinek(ax)
        h = np.concatenate([d["H_opt_emp"], d["H_opt_teor"]])
        podzialka = np.array([3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 80, 100])
        ax.set_yticks(podzialka[(podzialka >= 0.9 * h.min()) & (podzialka <= 1.1 * h.max())])
    osie[0].set_ylabel(r"$H_{opt}$")
    fig.tight_layout()
    zapisz(fig, "E7_wrazliwosc")
