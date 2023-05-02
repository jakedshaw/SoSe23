import numpy as np
from scipy.constants import epsilon_0
import matplotlib.pyplot as plt
import time


def init_feld():

    Q = []
    for qi, ri in zip([5e-6, -1e-6, 2e-6], [np.array([0, 1]), np.array([1, 1]), np.array([1, 0])]):
        Q.append([qi, ri])

    dx = []
    for dxi in [0.01, 0.01]:
        dx.append(dxi)
    # Epsilon > 0 fuer konvergenz
    eps = 0.5 * np.linalg.norm(dx)

    # Definitionsbereich
    x1 = np.arange(-1, 3+dx[0], dx[0])
    x2 = np.arange(-1, 2+dx[1], dx[1])
    x = [x1, x2]
    r = np.array(np.meshgrid(x1, x2, indexing='ij'))
    # Wertebereich
    E = np.zeros((len(x1), len(x2), len(x)+2))

    return r, E, Q, eps


def kraftfeld(p, Q, eps):
    """
    :param p: Punkt [x1, x2, ..., xn]
    :param Q: Q[0] Ladung, Q[1] = [x1, x2, ..., xn] ort der Ladung
    :return: E[i, j] Elektrisches Feld bei Punkt p
    """

    Ei = np.zeros(len(p)+2)
    phi = 0
    for q in Q:
        r = p - q[1]
        if np.linalg.norm(r) < eps:  # Analysis II
            return 0 * Ei

        Ei[:-2] += q[0] * r * np.linalg.norm(r)**-3
        phi += q[0] / np.linalg.norm(r)

    Ei_norm = np.linalg.norm(Ei[:-1])
    Ei_einheit = Ei[:-2] / np.linalg.norm(Ei[:-2])

    Ei[:-2] = Ei_einheit
    Ei[-2] = Ei_norm
    Ei[-1] = phi

    return Ei


def berechnungen(r, E, Q, eps):

    for i in range(r.shape[1]):
        for j in range(r.shape[2]):
            p = np.array([r[0, i, j], r[1, i, j]])
            Ei = kraftfeld(p, Q, eps)
            E[i, j] = Ei

    E /= 4 * np.pi * epsilon_0  # const.

    return r, E, Q


def kraftfeld2(p, q, rq, eps):
    rx1 = p[0] - rq[0]
    rx2 = p[1] - rq[1]

    q = q * np.ones(rx1.shape)

    norm = np.hypot(rx1, rx2)
    for i in range(norm.shape[0]-1):
        for j in range(norm.T.shape[0]-1):
            if norm[i, j] < eps:  # Analysis II
                q[i, j] = 0

    ex1, ex2 = q * rx1 * norm**-3, q * rx2 * norm**-3

    phi = q / norm

    return ex1, ex2, phi


def berechnungen2(r, E, Q, eps):

    for q in Q:
        p = np.array([r[0], r[1]])
        ex1, ex2, phi = kraftfeld2(p, *q, eps)
        E[:, :, 0] += ex1
        E[:, :, 1] += ex2
        E[:, :, -1] += phi

    E /= 4 * np.pi * epsilon_0  # const.

    norm = np.hypot(E[:, :, 0], E[:, :, 1])
    E[:, :, -2] = norm  # Betrag des elek Feldes
    E[:, :, 0] = E[:, :, 0] / norm  # Einheitsvektor
    E[:, :, 1] = E[:, :, 1] / norm  # Einheitsvektor

    return r, E, Q


def graph_feld(r, E, Q):

    # Abbildung elektrisches Feld
    fig, ax = plt.subplots(num=f"E-feld")

    # Einstellungen
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(r[0].min(), r[0].max())
    ax.set_ylim(r[1].min(), r[1].max())
    ax.set_xlabel("Abstand [m]")
    ax.set_ylabel("Abstand [m]")

    # E-Feld
    # ax.streamplot(r[0, :, 0], r[1, 0, :], E[:, 0, 0], E[0, :, 1])
    skip = (slice(None, None, 10), slice(None, None, 10))
    ax.quiver(r[0][skip], r[1][skip], E[:, :, 0][skip], E[:, :, 1][skip], np.log(E[:, :, -2][skip]))

    # Ladungen (Quellen bzw. Senken)
    farbe_q = {True: '#aa0000', False: '#0000aa'}
    for q in Q:
        ax.scatter(q[1][0], q[1][1], c=farbe_q[q[0] > 0])

    fig.tight_layout()

    # Abbildung elektrische Potential
    fig1, ax1 = plt.subplots(num=f"Elektrische Potential")

    # Einstellungen
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim(r[0].min(), r[0].max())
    ax1.set_ylim(r[1].min(), r[1].max())
    ax1.set_xlabel("Abstand [m]")
    ax1.set_ylabel("Abstand [m]")

    # elektrische Potential
    CS = ax1.contour(r[0], r[1], E[:, :, -1], 150, colors='k')
    # ax.clabel(CS, fontsize=9, inline=True)

    # Ladungen (Quellen bzw. Senken)
    farbe_q = {True: '#aa0000', False: '#0000aa'}
    for q in Q:
        ax1.scatter(q[1][0], q[1][1], c=farbe_q[q[0] > 0])

    fig1.tight_layout()
    plt.show()

    return True


def run():
    r, E, Q, eps = init_feld()
    t0 = time.time()
    r, E, Q = berechnungen2(r, E, Q, eps)
    t1 = time.time()
    # r, E, Q = berechnungen2(r, E, Q, eps)
    t2 = time.time()
    print(f"1: {t1 - t0}s\n2: {t2-t1}s")
    graph_feld(r, E, Q)


if __name__ == "__main__":
    run()
