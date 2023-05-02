# Author: Jacob Shaw
# Date: 27. April 2023
# Theoretische Physik
# 0.3 Freier Fall
# VEREINFACHTE VERSION

import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt


def freier_fall(h, t):
    """generates position and velocity for free fall"""

    null = 0 * t  # Nullektor der Laenge t.size
    eins = 1 + null  # Vektor lauter eins der Laenge t.size

    # Bewegungsgleichungen
    x = -g * t**2 / 2 + h  # Ort [m]
    v = -g * t  # Geschwindigkeit [m/s]
    a = -g * eins  # Beschleunigung [ms^-2]

    return x, v, a


def fall_mit_luftwiderstand(h, m, gam, t):
    """generates position and velocity arrays with resistance"""

    # Konstante als Funktion der Anfangsbedingungen
    c1 = -g * m**2 * gam**-2
    c2 = h - c1

    # Bewegungsgleichungen
    x = c1 * np.exp(gam * t / m) + g * m * t / gam + c2  # Ort [m]
    v = c1 * np.exp(gam * t / m) * gam / m + g * m / gam  # Geschwindigkeit [m/s]
    a = c1 * np.exp(gam * t / m) * gam ** 2 * m ** -2  # Beschleunigung [ms^-2]

    return x, v, a


def graph_data(data, gamma):
    """plots position vs time"""

    # Abbildung
    fig, ax = plt.subplots(nrows=3, num="Freier Fall")

    # Einstellungen zur Abbildung (nicht notwendig)
    ax[0].set_ylim(0, data[0][0][0])

    ax[0].set_ylabel("x [m]")
    ax[1].set_ylabel(r"v [ms$^{-1}$]")  # r"$ $" is for latex mathmode
    ax[2].set_ylabel(r"a [ms$^{-2}$]")  # r"$ $" is for latex mathmode
    ax[2].set_xlabel("t [s]")

    # Plot Schleife
    for i in range(3):
        ax[i].set_xlim(0, data[-1][-1])

        # plot freier Fall
        ax[i].plot(data[-1], data[i][0], label="Freier Fall")

        # plot schleife fuer Fall mit Luftwiderstand
        for j in range(len(data[0])-1):
            lab = r"$\gamma$ = " + f"{gamma[j]}"  # latex and f string don't mix well
            ax[i].plot(data[-1], data[i][j+1], label=lab)  # plot Fall mit Luftwiderstand

    fig.tight_layout()  # looks better when figure small
    ax[0].legend()  # show legend on zeroth (first) plot
    plt.show()  # show plot

    return True


def main(h=1_000, m=75):

    gamma = [-1, -2, -5, -10]  # Luftwiderstandkonstante

    dt = 0.01  # Zeitstufe
    tf = np.sqrt(2 * h / g) * 1.5   # Endzeit
    t = np.arange(0, tf + dt, dt)  # Zeitvektor t.size [s]

    # Init data array
    data = [[], [], [], t]

    # Freier Fall
    x, v, a = freier_fall(h, t)
    data[0].append(x)  # append to data array
    data[1].append(v)  # append to data array
    data[2].append(a)  # append to data array

    # Mit Luftwiderstand
    for gam in [-1, -2, -5, -10]:
        x, v, a = fall_mit_luftwiderstand(h, m, gam, t)
        data[0].append(x)
        data[1].append(v)
        data[2].append(a)

    # grafisch darstellen
    graph_data(data, gamma)

    return True


if __name__ == '__main__':
    main()
