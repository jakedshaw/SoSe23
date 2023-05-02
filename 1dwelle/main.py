# Author: Jacob Shaw
# Date: April 2023
# Mathematisches Studieren
# Fourierreihen

import time
import os
import anfangsbedingungen as ab
import fourierreihe as fr
import plot


def run(k_max=100):
    os.system('cls' if os.name == 'nt' else 'clear')

    u0, v0 = ab.select_init()

    t0 = time.time()  # STARTZEIT

    c, x, dx, t, dt, uo, vo, perioden = ab.init_funkt(u0, v0)

    u, b = fr.fourierreihe(c, x, dx, t, uo, vo, k_max)

    t1 = time.time()  # ENDZEIT
    print(f"\nexecuted in {t1 - t0} seconds")  # VERLAUFENE ZEIT

    plot.plot_welle(x, t, dt, u, uo, vo, b, k_max, perioden)


if __name__ == '__main__':
    run()
