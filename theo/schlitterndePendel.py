# Author: Jacob Shaw
# Date: 23. Mai 2023
# Theoretische Physik
# 5.2 Schlitternde Pendel

import time
import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def graph_data(t, r1, r2, l, m1, m2, phi_0, k=1.1):

    fig, ax = plt.subplots(num='Schlittende Pendel')  # Abbildung

    # Achsenbeschriftung
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    # x limits
    xm = min(np.min(r1[0]), np.min(r2[0]))
    xM = max(np.max(r1[0]), np.max(r2[0]))

    if (abs(xm) < l) and (abs(xM) < l):
        xm = -l
        xM = l

    elif abs(xm) > abs(xM):
        xM = -xm

    else:
        xm = -xM

    # Achsenintervalle
    ax.set_xlim(k * xm, k * xM)
    ax.set_ylim(k * -l, (k-1) * l)

    ax.plot([k * xm, k * xM], [0, 0], c='k')

    # plot massen bei t0
    rx = [r1[0, 0], r2[0, 0]]
    ry = [r1[1, 0], r2[1, 0]]
    masse, = ax.plot(rx, ry, linestyle="-", marker="o", c='k')

    def animate(i):  # animation schleife
        j = int(i % t.size)  # modulo Laenge der zeit (loop)

        rx = [r1[0, j], r2[0, j]]
        ry = [r1[1, j], r2[1, j]]

        masse.set_xdata(rx)  # x Koordinaten aktualisieren
        masse.set_ydata(ry)  # y Koordinaten aktualisieren

        return masse

    ani = animation.FuncAnimation(fig, animate, interval=10, blit=False, save_count=len(t))

    fig.tight_layout()  # looks better when figure small

    writer = animation.PillowWriter(fps=60, metadata=dict(artist='Jacob Shaw'), bitrate=1800)
    ani.save(f'assets/schlitternde-Pendel/{l}-{m1}-{m2}-{phi_0*180/np.pi}.gif', writer=writer)

    # ax.legend()  # show legend on zeroth (first) plot
    # plt.show()  # show plot

    return True


def run(m1=1, m2=2, l=1, phi_0=np.pi/4):

    # Zeitvektor
    t1, dt = 5, 0.01  # Endzeit, Zeitstufe
    t = np.arange(0, t1+dt, dt)  # Zeitvektor t.size [s]

    # Daten
    w = np.sqrt(g * (m2 + m1) / (m1 * l))
    phi = phi_0 * np.cos(w * t)
    s = (phi_0 - phi) * m2 * l / (m2 + m1)

    # Ortsvektor m1
    r1 = np.zeros((2, t.size))
    r1[0, :] = s

    # Ortsvektor m2
    r2 = np.zeros((2, t.size))
    r2[0, :] = r1[0, :] + l * np.sin(phi)  # x
    r2[1, :] = r1[1, :] - l * np.cos(phi)  # y

    # grafisch darstellen
    graph_data(t, r1, r2, l, m1, m2, phi_0)

    return True


def main_loop():

    t0 = time.time()

    trials = [
        [1, 1, 1, np.pi / 4],
        [1, 1, 2, np.pi / 4],
        [1, 1, 10, np.pi / 4],
        [1, 1, 1000, np.pi / 4],
        [1, 2, 1, np.pi / 4],
        [1, 10, 1, np.pi / 4],
        [1, 100, 1, np.pi / 4]
    ]

    for trial in trials:
        l, m1, m2, phi_0 = trial
        run(l, m1, m2, phi_0)

    t1 = time.time()

    print(f'executed in {t1-t0} seconds')

    return True


def eingabe():
    
    try:
        a = input(f'{}')
        return int(a)
    except Exception as e:
        print('input must be Integer!')

def custom_input():
    l =

if __name__ == '__main__':
    custom_input()
