# Author: Jacob Shaw
# Date: 23. April 2023
# Theoretische Physik
# 0.3 Freier Fall

import time
import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def freier_fall(h, t):
    """generates position and velocity for free fall"""
    # Nutzliche Vektoren
    null = 0 * t  # die Nullvektor Laenge t.size
    eins = 1 + 0 * t  # Vektor lauter eins der Laenge t.size

    # Matrizen 3 x t.size
    r = np.array([null, null, -g * t**2 / 2 + h])  # Ortsmatrix [m]
    v = np.array([null, null, -g * t])  # Geschwindigkeitsmatrix [m/s]
    a = np.array([null, null, -g * eins])  # Beschleunigungsmatrix [ms^-2]

    return r, v, a


def reibung(h, m, gam, t):
    """generates position and velocity arrays with resistance"""
    null = 0 * t  # die Nullvektor der Laenge t.size

    # Konstante als Funktion der Anfangsbedingungen
    c1 = -g * m**2 * gam**-2
    c2 = h - c1

    # Bewegungsgleichungen
    x3 = c1 * np.exp(gam * t / m) + g * m * t / gam + c2
    v3 = c1 * np.exp(gam * t / m) * gam / m + g * m / gam
    a3 = c1 * np.exp(gam * t / m) * gam ** 2 * m ** -2

    # Matrizen 3 x t.size
    r = np.array([null, null, x3])  # Ortsmatrix [m]
    v = np.array([null, null, v3])  # Geschwindigkeitsmatrix [m/s]
    a = np.array([null, null, a3])  # Beschleunigungsmatrix [ms^-2]

    return r, v, a


def graph_data(t, data, lab):
    a, b, c, d = data.shape

    m, n = 2, 2
    fig, ax = plt.subplots(nrows=m, ncols=n, num='Freier Fall')  # Abbildung m-Zeilen x n-Spalten

    k = 0
    for i in range(m):
        for j in range(n):

            if k == a:  # break loop if all plotted
                break

            # Achsenbeschriftung
            ax[i, j].set_xlabel('Zeit [s]')
            ax[i, j].set_ylabel(lab[k][0])

            # Achsenintervall
            Min, Max = [], []
            for ii in range(c):
                Min.append(np.min(data[k, :, ii]))
                Max.append(np.max(data[k, :, ii]))

            if k == 0:  # position graph
                Min[2] = 0

            ax[i, j].set_xlim(0, max(t))
            ax[i, j].set_ylim(Min[2], Max[2])  # z aka x_3 Achse

            # Graf der Funktion
            for ii in range(len(data[k])):
                ax[i, j].plot(t, data[k, ii, 2], label=lab[k][ii+1])

            k += 1

    # animation
    ax[1, 1].set_xlim(0, b + 1)
    ax[1, 1].set_ylim(0, 1.2 * data[0, 0, 2, 0])

    ax[1, 1].set_xlabel("Massenpunkt")
    ax[1, 1].set_ylabel("Hoehe [m]")

    # plot massen bei t0
    masse = []
    for ii in range(b):
        masse_tmp, = ax[1, 1].plot(ii+1, data[0, ii, 2, 0], linestyle="", marker="o", label="Fall mit Reibung")
        masse.append(masse_tmp)

    def animate(i):  # animation schleife
        j = int(i % t.size)  # modulo Laenge der zeit (loop)

        for jj in range(len(masse)):
            masse[jj].set_ydata(data[0, jj, 2, j])  # ortsvektor aktualisieren

        return masse

    ani = animation.FuncAnimation(fig, animate, interval=50, blit=False, save_count=len(t))

    fig.tight_layout()  # looks better when figure small

    # writer = animation.PillowWriter(fps=30, metadata=dict(artist='Jacob Shaw'), bitrate=1800)
    # ani.save('freier_Fall.gif', writer=writer)

    ax[0, 0].legend()  # show legend on zeroth (first) plot
    plt.show()  # show plot

    return True


def run(h=1_000, m=85):
    t0 = time.time()  # t0

    gam = [-1, -5, -10, -20]

    t1, dt = np.sqrt(2 * h / g) * 2, 0.4  # Endzeit, Zeitstufe
    t = np.arange(0, t1+dt, dt)  # Zeitvektor t.size [s]

    # Datenmatrix
    data = np.empty((3, len(gam)+1, 3, t.size))
    lab = [[], [], []]

    data[0, 0, :], data[1, 0, :], data[2, 0, :] = freier_fall(h, t)
    lab[0].append('Hoehe [m]')
    lab[1].append(r'Geschwindigkeit [ms$^{-1}$]')
    lab[2].append(r'Beschleunigung [ms$^{-2}$]')

    for j in range(3):
        lab[j].append("Freier Fall")

    for i in range(len(gam)):
        data[0, i+1, :], data[1, i+1, :], data[2, i+1, :] = reibung(h, m, gam[i], t)
        for j in range(3):
            lab[j].append(r"$\gamma$ = " + f"{gam[i]}")

    t1 = time.time()  # t1
    print(f"\nwurde in {t1 - t0} Sekunden durchgefuehrt")

    # grafisch darstellen
    graph_data(t, data, lab)

    return True


if __name__ == '__main__':
    run()
