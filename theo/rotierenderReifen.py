# Author: Jacob Shaw
# Date: 04. Juni 2023
# Theoretische Physik
# 7.1 Messepunkt in einem rotierenden Reifen

import time
import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def graph_data(massenpunkt, reifen, R, save=False, show=True, k=1.1):

    fig = plt.figure(num='Rotierender Reifen',figsize=(1, 1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Achsenbeschriftung
    ax.set_xlabel('x1 [m]')
    ax.set_ylabel('x2 [m]')
    ax.set_zlabel('x3 [m]')

    # Achsenintervalle
    ax.set_xlim(k * -R, k * R)
    ax.set_ylim(k * -R, k * R)
    ax.set_zlim(k * -R, k * R)

    # Anfangsposition der Masse
    x1, x2, x3 = massenpunkt[0]
    masse, = ax.plot(x1, x2, x3, linestyle='dashed', marker='o', c='k', label='massenpunkt')

    # Anfangsposition des Ringes
    x1, x2, x3 = reifen[0]
    ring = ax.plot_wireframe(x1, x2, x3, color='grey', label='ring')

    fig.tight_layout()  # looks better when figure small
    # ax.legend()  # show legend on zeroth (first) plot

    for i in range(100000):
        j = int(i % len(massenpunkt))  # modulo Laenge der zeit (loop)

        # Massenpunkt
        x1, x2, x3 = massenpunkt[j]
        masse.set_data(x1, x2)  # x1,x2 Koordinaten aktualisieren
        masse.set_3d_properties(x3)  # x3 Koordinaten aktualisieren

        # Ring
        ring.remove()
        x1, x2, x3 = reifen[j]
        ring = ax.plot_wireframe(x1, x2, x3, color='grey', label='ring')

        plt.pause(.0001)

    plt.clf()

    return True


def graph_data_schale(r, R, w, save=False, show=True, k=1.1):

    fig = plt.figure(num='Rotierender Reifen', figsize=(1, 1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Sphere
    theta, phi = np.linspace(0, np.pi/2, 20), np.linspace(0, 2*np.pi, 20)
    PHI, THETA = np.meshgrid(phi, theta)

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    def animate(i):  # animation schleife
        j = int(i % len(r))  # modulo Laenge der zeit (loop)

        ax.cla()

        # Achsenbeschriftung
        ax.set_xlabel('x1 [m]')
        ax.set_ylabel('x2 [m]')
        ax.set_zlabel('x3 [m]')

        # Achsenintervalle
        ax.set_xlim(k * -R, k * R)
        ax.set_ylim(k * -R, k * R)
        ax.set_zlim(k * -R, k * R)

        # Einstellungen
        fig.tight_layout()  # looks better when figure small
        # ax.legend()  # show legend on zeroth (first) plot

        # Schale
        ax.plot_wireframe(X, Y, -Z, color='grey')

        # Massenpunkt
        x1, x2, x3 = r[j]
        ax.scatter(x1, x2, x3, c='k', label='massenpunkt')
        ax.plot(r[:j, 0], r[:j, 1], r[:j, 2], c='k')

    ani = animation.FuncAnimation(fig, animate, interval=10, blit=False, save_count=len(r))

    writer = animation.PillowWriter(fps=60, metadata=dict(artist='Jacob Shaw'), bitrate=1800)

    if save:
        ani.save(f'assets/rotierenderReifen/{R}-{w}.gif', writer=writer)

    if show:
        plt.show()  # show plot

    plt.clf()

    return True


def berechnungen(m=1, R=1, w=1.5*np.pi, phi_0=0):

    # Zeitvektor
    f = w / (2 * np.pi)  # Frequenz
    T = 1 / f  # Periodendauer
    n = 120  # Anzahl der Punkten
    t1, dt = T, T/n  # Endzeit, Zeitstufe
    t = np.arange(0, t1+dt, dt)  # Zeitvektor t.size [s]

    # Daten
    z = g / (R * w ** 2)
    # theta = np.arccos(z)
    A = np.sqrt(1 - z**2)
    phi = w * t + phi_0
    L_phi = m * R**2 * w * A

    # Ortsvektor
    r = np.zeros((t.size, 3))
    r[:, 0] = A * np.cos(phi)  # x1
    r[:, 1] = A * np.sin(phi)  # x2
    r[:, 2] = -z  # x3
    # r = [A * np.cos(phi), A * np.sin(phi), -z]

    # Ortsvektor vom Ursprung
    origin = [0, 0, 0]
    massepunkt = []
    for i in range(len(r)):
        x_tmp = []
        for j in range(3):
            x_tmp.append([origin[j], r[i, j]])
        massepunkt.append(x_tmp)

    print(f'L_phi = {L_phi}')

    # Ring
    m = 40
    reifen = np.zeros((t.size, 3, m, 1))
    for i in range(len(phi)):

        theta = np.linspace(0, 2 * np.pi, m)
        PHI, THETA = np.meshgrid(phi[i], theta)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        reifen[i, 0, :] = X
        reifen[i, 1, :] = Y
        reifen[i, 2, :] = Z

    # grafisch darstellen
    graph_data(massepunkt, reifen, R)
    graph_data_schale(r, R, w)

    return True


if __name__ == '__main__':
    t0 = time.time()

    berechnungen()

    t1 = time.time()

    print(f'executed in {t1 - t0} seconds')
