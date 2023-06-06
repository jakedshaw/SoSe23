# Author: Jacob Shaw
# Date: 04. Juni 2023
# Theoretische Physik
# Shapes

import time
import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    r = np.zeros((3, t.size))
    r[0, :] = A * np.cos(phi)  # x1
    r[1, :] = A * np.sin(phi)  # x2
    r[2, :] = -z  # x3

    reifen = np.zeros((3, t.size))


    print(f'L_phi = {L_phi}')

    """
    # Sphere
    theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
    THETA, PHI = np.meshgrid(theta, phi)

    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    """

    # Ring
    theta = np.linspace(0, 2 * np.pi, 20)
    PHI, THETA = np.meshgrid(phi[0], theta)

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # grafisch darstellen
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r[0, 0], r[1, 0], r[2, 0], zdir='z', c='k')
    ax.plot_wireframe(X, Y, Z, color='grey')
    plt.show()

    # graph_data(t, r1, r2, l, m1, m2, phi_0)

    return True


if __name__ == '__main__':
    berechnungen()
