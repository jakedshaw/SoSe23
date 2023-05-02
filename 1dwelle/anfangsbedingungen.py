# Author: Jacob Shaw
# Date: April 2023
# Mathematisches Studieren
# Fourierreihen

import numpy as np
from numpy import sin, pi
import os


def u01(L, x):
    A = 0.02  # 5 cm
    k = 3  # 1, 2, ...
    return A * sin(k*x)


def u02(L, x):
    s = L/6
    u_s = 0.02  # 5 cm
    m1 = u_s / s
    m2 = -u_s / (L - s)

    u_o = []
    for xi in x:

        m = m1
        if xi > s:
            m = m2

        u_o.append(m*(xi - s) + u_s)

    return np.array(u_o)


def u03(L, x):
    """square wave"""
    s = L/3
    u_s = 0.02  # 5 cm

    u_o = []
    for xi in x:

        u = 0
        if s < xi < 2*s:
           u = u_s

        u_o.append(u)

    return np.array(u_o)


def u04(L, x):
    k = 2
    return 1.25 * sin(k * x) * u02(L, x)


def u05(L, x):
    """sawtooth wave"""
    k = 2
    return 1.25 * sin(k * x) * u03(L, x)


def u06(L, x):
    s = L / 6
    u_s = 0.02  # 2 cm

    u_o = []
    for xi in x:

        u = 0
        if s < xi < 2 * s:
            u = u_s * sin((xi - s) * pi / s)

        u_o.append(u)

    return np.array(u_o)


def u07(L, x):
    """wave packet"""
    k = 45  # 1, 2, ...
    return 1.25 * sin(k * x) * u06(L, x)


def v01(L, x):
    return 250 * u02(L, x)


def select_init():

    u0, v0 = False, False

    a = input("\n  (1) for purely plucked string"
              "\n  (2) for plucked string"
              "\n  (3) for striked string"
              "\n  (4) for wave packet\n"
              ""
              "\nInput: ")

    if a == "1":
        u0 = u02

    elif a == "2":
        u0 = u02
        v0 = v01

    elif a == "3":
        v0 = v01

    elif a == "4":
        u0 = u07

    else:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Invalid Input!")
        return select_init()

    return u0, v0


def init_funkt(u0, v0):
    """Initializes variables"""

    # KONSTANTEN UND VERAENDERLICHEN
    tau = 770  # SAITENSPANNUNG [N]
    rho = 6.2e-3  # MASSE PRO METER LAENGE [kg]
    c = np.sqrt(tau / rho)  # AUSBREITUNGS- / PHASENGESCHWINDIGKEIT [m/s]
    print(f"\nAUSBREITUNGGESCHWINDIGKEIT: c = {c} m/s")

    To = 2 * pi / c  # PERIODENDAUER [s]
    print(f"GRUNDPERIODENDAUER: T_o = {To} s")
    print(f"GRUNDFREQUENZ: w_o = {1/To} Hz")

    perioden = 2
    L, dx = pi, 0.01  # LAENGE UND FEINHEIT [m]
    t_f, dt = perioden * To, 0.0001  # MAX ZEIT [s]

    # INITIALIZE ZEIT- UND ORTSARRAYS
    x = np.arange(0, L + dx, dx)  # m
    t = np.arange(0, t_f + dt, dt)  # s

    # ANFANGSBEDINGUNGEN
    uo = 0 * x  # ANFANGSAUSLENKUNG [m]
    if u0:
        uo = u0(L, x)  # m

    vo = 0 * x  # ANFANGSGESCHWINDIGKEIT [m/s]
    if v0:
        vo = v0(L, x)  # m/s

    return c, x, dx, t, dt, uo, vo, perioden


if __name__ == '__main__':
    pass
