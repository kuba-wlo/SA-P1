"""Eksperymenty E1-E7: obliczenia i zapis wyników do CSV/JSON."""

import csv
import json

import numpy as np

from src import config, dane, estymator, imse, teoria


# --- zapis wyników -----------------------------------------------------------

def _tekst(x):
    if isinstance(x, (str, int, np.integer)):
        return str(x)
    return f"{float(x):.10g}"


def zapisz_csv(nazwa, kolumny):
    """Zapisuje słownik kolumn (równej długości) do results/tables/<nazwa>.csv."""
    config.KATALOG_TABEL.mkdir(parents=True, exist_ok=True)
    klucze = list(kolumny)
    n = len(kolumny[klucze[0]])
    with open(config.KATALOG_TABEL / f"{nazwa}.csv", "w", newline="", encoding="utf-8") as f:
        pisz = csv.writer(f)
        pisz.writerow(klucze)
        for i in range(n):
            pisz.writerow([_tekst(kolumny[k][i]) for k in klucze])


def _json(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def zapisz_podsumowanie(klucz, wartosc):
    """Dopisuje (lub nadpisuje) wpis `klucz` w results/podsumowanie.json."""
    sciezka = config.KATALOG_WYNIKOW / "podsumowanie.json"
    calosc = json.loads(sciezka.read_text(encoding="utf-8")) if sciezka.exists() else {}
    calosc[klucz] = json.loads(json.dumps(wartosc, default=_json))
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    sciezka.write_text(json.dumps(calosc, indent=2, ensure_ascii=False), encoding="utf-8")


def _zakres_1proc(H, wartosci):
    """Najmniejsze i największe H, dla których IMSE <= 1,01*IMSE_min."""
    H_ok = H[wartosci <= 1.01 * wartosci.min()]
    return int(H_ok.min()), int(H_ok.max())


# --- E1: weryfikacja generatora ---------------------------------------------

def e1_generator():
    v = config.E1_V
    a = np.sqrt(6 * v)
    z = dane.zaklocenia(dane.generator("E1"), v, config.E1_PROBKI)

    gestosc, krawedzie = np.histogram(z, bins=config.E1_KOSZE, range=(-a, a), density=True)
    srodki = (krawedzie[:-1] + krawedzie[1:]) / 2
    zapisz_csv("E1_histogram", {
        "lewa_krawedz": krawedzie[:-1], "prawa_krawedz": krawedzie[1:],
        "gestosc_emp": gestosc, "gestosc_teor": dane.gestosc(srodki, v),
    })

    emp = [z.mean(), z.var(), np.max(np.abs(z))]
    teor = [0.0, v, a]
    zapisz_csv("E1_generator", {
        "wielkosc": ["srednia", "wariancja", "max_abs_z"],
        "empirycznie": emp, "teoretycznie": teor,
    })
    wynik = {"v": v, "probki": config.E1_PROBKI, "srednia": emp[0], "wariancja": emp[1],
             "max_abs_z": emp[2], "sqrt_6v": a,
             "blad_wzgl_wariancji": abs(emp[1] - v) / v}
    zapisz_podsumowanie("E1", wynik)
    return wynik


# --- E2: przebiegi czasowe ---------------------------------------------------

def e2_przebiegi():
    nazwa = config.E2_WARIANT
    alpha, v = config.WARIANTY[nazwa]
    h_opt, _ = teoria.h_opt(alpha, v, config.H_MAX)
    theta_gw, theta = dane.pomiary(dane.generator("E2"), config.THETA_0, alpha, v, config.N)

    p = config.E2_POCZATEK_WYCINKA
    assert p >= config.H_MAX - 1, "wycinek musi zawierać tylko pełne okna"
    wycinek = slice(p, p + config.E2_DLUGOSC_WYCINKA)
    kolumny = {"k": np.arange(config.N)[wycinek], "theta": theta[wycinek],
               "theta_gw": theta_gw[wycinek]}
    for H in (2, h_opt, config.H_MAX):
        kolumny[f"est_H{H}"] = estymator.okno_przesuwne(theta, H)[wycinek]
    zapisz_csv("E2_przebiegi", kolumny)

    wynik = {"wariant": nazwa, "alpha": alpha, "v": v, "H": [2, h_opt, config.H_MAX],
             "k_od": p, "k_do": p + config.E2_DLUGOSC_WYCINKA - 1}
    zapisz_podsumowanie("E2", wynik)
    return wynik


# --- E3: IMSE(H) dla W1-W4 ---------------------------------------------------

def e3_warianty():
    H = np.asarray(config.H_VALUES)
    krzywe = {"H": H}
    tabela = {k: [] for k in (
        "wariant", "alfa", "v", "H_opt_teor", "H_opt_przybl", "IMSE_min_teor",
        "H_opt_emp", "IMSE_min_emp", "dH", "max_blad_wzgl",
        "H_1proc_teor_od", "H_1proc_teor_do", "H_1proc_emp_od", "H_1proc_emp_do")}

    for i, (nazwa, (alpha, v)) in enumerate(config.WARIANTY.items()):
        w = imse.imse_mc(dane.generator("E3", i), config.THETA_0, alpha, v)
        teor = teoria.imse(H, alpha, v)
        h_t, imse_t = teoria.h_opt(alpha, v, config.H_MAX)
        j = int(np.argmin(w["imse"]))
        krzywe[f"{nazwa}_imse_emp"] = w["imse"]
        krzywe[f"{nazwa}_imse_teor"] = teor
        krzywe[f"{nazwa}_bias2_emp"] = w["bias2"]
        krzywe[f"{nazwa}_var_emp"] = w["var"]

        for klucz, wart in zip(tabela, (
                nazwa, alpha, v, h_t, teoria.h_opt_przyblizenie(alpha, v), imse_t,
                int(H[j]), w["imse"][j], int(H[j]) - h_t,
                np.max(np.abs(w["imse"] - teor) / teor),
                *_zakres_1proc(H, teor), *_zakres_1proc(H, w["imse"]))):
            tabela[klucz].append(wart)

    zapisz_csv("E3_imse_krzywe", krzywe)
    zapisz_csv("E3_imse_tabela", tabela)
    wynik = [dict(zip(tabela, wiersz)) for wiersz in zip(*tabela.values())]
    zapisz_podsumowanie("E3", wynik)
    return wynik


# --- E4: dekompozycja IMSE (W2) ----------------------------------------------

def e4_dekompozycja():
    nazwa = config.E4_WARIANT
    alpha, v = config.WARIANTY[nazwa]
    H = np.asarray(config.H_VALUES)
    w = imse.imse_mc(dane.generator("E4"), config.THETA_0, alpha, v)
    b2_t, var_t = teoria.bias2(H, alpha), teoria.wariancja(H, v)
    zapisz_csv("E4_dekompozycja", {
        "H": H, "bias2_emp": w["bias2"], "var_emp": w["var"], "imse_emp": w["imse"],
        "bias2_teor": b2_t, "var_teor": var_t, "imse_teor": b2_t + var_t,
    })

    h_t, imse_t = teoria.h_opt(alpha, v, config.H_MAX)
    j = int(np.argmin(w["imse"]))
    # Przy skończonym M: E[bias2_emp] = bias^2 + Var/M, E[var_emp] = Var*(M-1)/M
    wynik = {
        "wariant": nazwa, "alpha": alpha, "v": v,
        "H_opt_teor": h_t, "IMSE_min_teor": imse_t,
        "H_opt_emp": int(H[j]), "IMSE_min_emp": w["imse"][j],
        "H_przeciecia_teor": teoria.h_przeciecia(alpha, v),
        "H_przeciecia_emp": int(H[np.argmax(w["bias2"] >= w["var"])]),
        "srednia_bias2_emp_minus_teor": np.mean(w["bias2"] - b2_t),
        "srednia_oczekiwana_Var_przez_M": np.mean(var_t / config.M),
    }
    zapisz_podsumowanie("E4", wynik)
    return wynik


# --- E5: pojedyncza realizacja wobec uśrednienia (W4, N = 500) ---------------

def e5_realizacja():
    nazwa = config.E5_WARIANT
    alpha, v = config.WARIANTY[nazwa]
    H = np.asarray(config.H_VALUES)
    theta_gw, theta = dane.pomiary(dane.generator("E5"), config.THETA_0, alpha, v,
                                   config.E5_N, M=config.E5_REALIZACJE)
    per = imse.imse_realizacji(theta, theta_gw, H, config.H_MAX)
    h_opt_real = H[np.argmin(per, axis=1)]
    teor = teoria.imse(H, alpha, v)
    h_t, imse_t = teoria.h_opt(alpha, v, config.H_MAX)

    zapisz_csv("E5_realizacja", {"H": H, "imse_jednej_realizacji": per[0],
                                 "imse_srednie": per.mean(axis=0), "imse_teor": teor})
    zapisz_csv("E5_hopt_realizacji", {"realizacja": np.arange(len(h_opt_real)),
                                      "H_opt": h_opt_real})
    stat = {
        "srednia": h_opt_real.mean(), "mediana": np.median(h_opt_real),
        "percentyl_5": np.percentile(h_opt_real, 5),
        "percentyl_95": np.percentile(h_opt_real, 95),
        "min": int(h_opt_real.min()), "max": int(h_opt_real.max()),
        "odch_std": h_opt_real.std(ddof=1),
    }
    zapisz_csv("E5_statystyki_hopt", {k: [w] for k, w in stat.items()})

    wynik = {"wariant": nazwa, "alpha": alpha, "v": v, "N": config.E5_N,
             "realizacje": config.E5_REALIZACJE, "H_opt_teor": h_t, "IMSE_min_teor": imse_t,
             "H_opt_jednej_realizacji": int(h_opt_real[0]),
             "IMSE_min_jednej_realizacji": per[0].min(),
             "H_opt_sredniego_IMSE": int(H[np.argmin(per.mean(axis=0))]),
             "statystyki_H_opt": stat}
    zapisz_podsumowanie("E5", wynik)
    return wynik


# --- E6: stały parametr (W0) -------------------------------------------------

def e6_staly_parametr():
    alpha, v = config.WARIANT_STALY["W0"]
    H = np.asarray(config.H_VALUES)

    # (a) IMSE(H) wobec v/H
    w = imse.imse_mc(dane.generator("E6", 0), config.THETA_0, alpha, v)
    teor = teoria.imse(H, alpha, v)
    zapisz_csv("E6_staly_parametr", {"H": H, "imse_emp": w["imse"], "imse_teor": teor,
                                     "bias2_emp": w["bias2"], "var_emp": w["var"]})

    # (b) wariancja estymaty dla H = N (średnia ze wszystkich N pomiarów)
    blok = 250
    war_emp = []
    for i, N in enumerate(config.E6_N_LISTA, start=1):
        rng = dane.generator("E6", i)
        estymaty = []
        for start in range(0, config.E6_M, blok):
            m = min(blok, config.E6_M - start)
            theta_gw, theta = dane.pomiary(rng, config.THETA_0, alpha, v, N, M=m)
            estymaty.append(estymator.okno_przesuwne(theta, N)[:, -1] - theta_gw[-1])
        war_emp.append(np.var(np.concatenate(estymaty), ddof=0))
    N_lista = np.asarray(config.E6_N_LISTA)
    war_emp = np.asarray(war_emp)
    war_teor = v / N_lista
    zapisz_csv("E6_wariancja_N", {"N": N_lista, "var_emp": war_emp, "var_teor": war_teor,
                                  "stosunek_emp_teor": war_emp / war_teor})
    nachylenie = np.polyfit(np.log(N_lista), np.log(war_emp), 1)[0]

    wynik = {
        "alpha": alpha, "v": v,
        "H_opt_teor": teoria.h_opt(alpha, v, config.H_MAX)[0],
        "H_opt_emp": int(H[np.argmin(w["imse"])]),
        "IMSE_H_max_emp": w["imse"][-1], "IMSE_H_max_teor": teor[-1],
        "max_blad_wzgl_a": np.max(np.abs(w["imse"] - teor) / teor),
        "max_bias2_emp": np.max(w["bias2"]),
        "M_b": config.E6_M, "nachylenie_log_var_od_log_N": nachylenie,
        "max_blad_wzgl_b": np.max(np.abs(war_emp / war_teor - 1)),
    }
    zapisz_podsumowanie("E6", wynik)
    return wynik


# --- E7: wrażliwość H_opt na alpha i v ---------------------------------------

def _nachylenie(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def e7_wrazliwosc():
    H = np.asarray(config.H_VALUES)
    serie = {
        "alfa": [(a, config.E7_V_PRZY_ALPHA) for a in config.E7_ALPHA_LISTA],
        "v": [(config.E7_ALPHA_PRZY_V, v) for v in config.E7_V_LISTA],
    }
    pod = 0
    wynik = {}
    for seria, punkty in serie.items():
        kol = {k: [] for k in ("alfa", "v", "H_opt_teor", "H_opt_przybl", "H_opt_emp",
                               "IMSE_min_teor", "IMSE_min_emp")}
        for alpha, v in punkty:
            w = imse.imse_mc(dane.generator("E7", pod), config.THETA_0, alpha, v)
            pod += 1
            h_t, imse_t = teoria.h_opt(alpha, v, config.H_MAX)
            j = int(np.argmin(w["imse"]))
            for k, x in zip(kol, (alpha, v, h_t, teoria.h_opt_przyblizenie(alpha, v),
                                  int(H[j]), imse_t, w["imse"][j])):
                kol[k].append(x)
        zapisz_csv(f"E7_wrazliwosc_{seria}", kol)

        x = np.asarray(kol["alfa" if seria == "alfa" else "v"])
        wynik[seria] = {
            "nachylenie_emp": _nachylenie(x, kol["H_opt_emp"]),
            "nachylenie_teor_calkowite": _nachylenie(x, kol["H_opt_teor"]),
            "nachylenie_przyblizenia": _nachylenie(x, kol["H_opt_przybl"]),
            "nachylenie_teoretyczne": -2 / 3 if seria == "alfa" else 1 / 3,
            "punkty": [dict(zip(kol, p)) for p in zip(*kol.values())],
        }

    zapisz_csv("E7_nachylenia", {
        "seria": list(wynik),
        **{k: [wynik[s][k] for s in wynik] for k in (
            "nachylenie_emp", "nachylenie_teor_calkowite", "nachylenie_przyblizenia",
            "nachylenie_teoretyczne")},
    })
    zapisz_podsumowanie("E7", wynik)
    return wynik
