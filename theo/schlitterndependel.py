# Author: Jacob Shaw
# Date: 23. Mai 2023
# Theoretische Physik
# 5.2 Schlitternde Pendel

import time
import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def graph_data(t, r1, r2, l, M, m, phi_0, k=1.1):

    fig, ax = plt.subplots(num='Schlittende Pendel')  # Abbildung

    # Achsenbeschriftung
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    # Achsenintervalle
    ax.set_xlim(k * -l, k * l)
    ax.set_ylim(k * -l, (k-1) * l)

    ax.plot([k * -l, k * l], [0, 0], c='k')

    # plot massen bei t0
    rx = [r1[0, 0], r2[0, 0]]
    ry = [r1[1, 0], r2[1, 0]]
    print(r1[:, 0])
    masse, = ax.plot(rx, ry, linestyle="-", marker="o", c='k')

    def animate(i):  # animation schleife
        j = int(i % t.size)  # modulo Laenge der zeit (loop)

        rx = [r1[0, j], r2[0, j]]
        ry = [r1[1, j], r2[1, j]]

        masse.set_xdata(rx)  # ortsvektor aktualisieren
        masse.set_ydata(ry)  # ortsvektor aktualisieren

        return masse

    ani = animation.FuncAnimation(fig, animate, interval=10, blit=False, save_count=len(t))

    fig.tight_layout()

    # writer = animation.PillowWriter(fps=60, metadata=dict(artist='Jacob Shaw'), bitrate=1800)
    # ani.save(f'{l}-{M}-{m}-{phi_0*180/np.pi}.gif', writer=writer)

    # ax[0, 0].legend()  # Legend
    plt.show()

    return True


def run(M=1, m=100, l=1, phi_0=np.pi/4):
    t0 = time.time()  # t0

    # Zeit
    t1, dt = 5, 0.01  # Endzeit, Zeitstufe
    t = np.arange(0, t1+dt, dt)  # Zeitvektor t.size [s]

    # Daten
    lam = np.sqrt((g / l) * (m + M) / (m + M - 1))
    phi = phi_0 * np.cos(lam * t)
    s = -phi * m * l / (m + M)

    r1 = np.zeros((2, t.size))
    r1[0, :] = s

    r2 = np.zeros((2, t.size))
    r2[0, :] = r1[0, :] + l * np.sin(phi)  # x
    r2[1, :] = r1[1, :] - l * np.cos(phi)  # y

    t1 = time.time()  # t1
    print(f"\nwurde in {t1 - t0} Sekunden durchgefuehrt")

    # grafisch darstellen
    graph_data(t, r1, r2, l, M, m, phi_0)

    return True


if __name__ == '__main__':
    run()
