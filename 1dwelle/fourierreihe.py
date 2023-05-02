# Author: Jacob Shaw
# Date: April 2023
# Mathematisches Studieren
# Fourierreihen

import numpy as np
from numpy import sin, cos, pi


def fourierreihe(c, x, dx, t, uo, vo, k_max):
    """
    u(x,t) = sum_{k=1}^{oo} b_k * sin(kx)
    """
    # INITIALIZE AUSLENKUNGSMATRIX
    u = np.zeros((len(x), len(t)))  # m

    b = np.zeros((len(t), k_max))  # INIT KOEFFIZIENTENMATRIX

    for i in range(len(t)):  # DURCH DIE ZEIT ITERIEREN

        u_i = x * 0  # AUSLENKUNG BEI ZEIT t

        for k in range(1, k_max + 1):  # DURCH K ITERIEREN (REIHE)

            wk = k * c  # KREISFREQUENZ

            f = sin(k * x) * (uo * cos(wk * t[i]) + (vo / wk) * sin(wk * t[i]))  # INTEGRAND

            integral = dx * np.sum((f[1:] + f[:-1]) / 2)  # TRAPEZ

            b_k = (2 / pi) * integral  # ZEITKOMPONENT DER FOURIERREIHE

            b[i, k - 1] = np.sum(b_k)  # SUMME DER KOEFFIZIENTEN UEBER DIE LAENGE L BEI WELLENZAHL k

            u_i += sin(k * x) * b_k  # ZUR SUMME ADDIEREN

        u[:, i] = u_i  # append Auslenkung for time t

    return u, b


if __name__ == '__main__':
    pass
