# Author: Jacob Shaw
# Date: 26. April 2023
# Theoretische Physik
# 1.3 Wassertropfen

import time
import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt


def graph_data(data, lab):
    
    m, n = 2, 2
    fig, ax = plt.subplots(nrows=m, ncols=n, num='Wassertropfen')  # Abbildung m-Zeilen x n-Spalten

    k = 0
    for i in range(m):
        for j in range(n):
            k += 1
            # Achsenbeschriftung
            ax[i, j].set_xlabel(lab[0])
            ax[i, j].set_ylabel(lab[k])
            # Achsenintervall
            ax[i, j].set_xlim(0, max(data[0]))
            ax[i, j].set_ylim(0, 1.1 * max(data[k]))
            # Graf der Funktion
            ax[i, j].plot(data[0], data[k], c='k')
    
    plt.tight_layout()
    plt.show()

    return True


def run(a=1, rho=1, Ro=1):

    t0 = time.time()  # t0

    print(f"\nProportionalitaetskoeffizient: alpha = {a}\n"
          f"Anfangsradius: R_o = {Ro}\n"
          f"Dichte: rho = {rho}\n")

    # Zeitvektor
    dt = 0.01
    t = np.arange(0, 5+dt, dt)

    # Funktionen
    eta = a * t / 4 + Ro  # funktion abh. v. d. Zeit
    R = a * t / 4 + Ro  # Radiusfunktion abh. v. d. Zeit
    m = 4 * np.pi * rho * R ** 3 / 3  # Massenfunktion abh. v. d. Zeit

    # Konstante
    eta_dt = a / 4  # d/dt eta = const.
    c1 = 2 * g * Ro**4 * a**-2  # v0 = 0
    c2 = -c1 * Ro**-2  # ro = 0

    # Bewegungsgleichung (nullte, erste und zweite zeitliche Ableitung)
    r = c1 * eta**-2 + g * t**2 / 8 + g * Ro * t / a + c2
    v = -2 * c1 * eta**-3 * eta_dt + g * t / 4 + g * Ro / a
    a = 6 * c1 * eta_dt**2 * eta**-4 + g / 4

    # Datenmatrix mit den zugehoerigen Labeln
    data = np.array([t, m, r, v, a])
    lab = ['Zeit [s]', 'Masse [kg]', 'Ort [m]', r'Geschwindigkeit [ms$^{-1}$]', r'Beschleunigung [ms$^{-2}$]']

    t1 = time.time()  # t1
    print(f"\ndurchgefuehrt in {t1 - t0} Sekunden")

    # plot
    graph_data(data, lab)

    return True


if __name__ == '__main__':
    run()

