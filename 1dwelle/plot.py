import matplotlib.pyplot as plt
import numpy as np
from numpy import sin
import matplotlib.animation as animation


def plot_welle(x, t, dt, u, uo, vo, b, k_max, perioden, scale=0.025):

    # FIGURE 1
    fig1, ax = plt.subplots(num="FOURIER TRANSFORM", nrows=2)

    # rechungen
    uf = x * 0
    for k in range(1, len(b[0])+1):
        uf += b[0, k-1] * sin(k*x)  # reconstruction for t=0

    color = np.empty(b.shape, dtype=str)
    for i in range(len(t)):
        for k in range(len(b[0])):
            if b[i, k] < 0:
                color[i, k] = "r"
            else:
                color[i, k] = "k"


    x *= 100  # m -> cm
    uo *= 100  # m -> cm
    uf *= 100  # m -> cm
    t *= 1000  # s -> ms
    u *= 100  # m -> cm
    mx = max(x.max(), x.min(), key=abs)
    mu = 2 * max(u.max(), u.min(), key=abs)
    m = max(mx, mu)

    # einstellungen
    ax[0].set_xlim([0, m])
    ax[0].set_ylim([scale * (-m / 2), scale * m / 2])
    ax[0].set_xlabel("LAENGE [cm]")
    ax[0].set_ylabel("AUSLENKUNG [cm]")
    # plot

    # ax[0].plot(x, uf, c="gray")  ###############################################################
    # ax[0].plot(x, uo, c="k")                    Anfangsbedingungen zeigen
    # ax[0].plot(x, vo, c="r")     ###############################################################

    saite, = ax[0].plot(x, u[:, 0], c="k")
    ts = ax[0].text(250, 0.7 * scale * m / 2, "")

    # einstellungen
    ax[1].set_xlim([0, len(b[0]) + 1])
    ax[1].set_ylim([b.min(), b.max()])
    ax[1].set_xlabel(r"$ k \in\mathbf{N}_{" + f"{k_max}" + "} $")
    # ax[1].set_xlabel(r"WELLENZAHL ($ k \in\mathbf{N}_{" + f"{k_max}" + "} $)")
    ylab = ax[1].set_ylabel("")
    # plot
    bar = ax[1].bar(np.arange(1, len(b[0]) + 1), b[1], color=color[0])

    def animate(i):
        j = i % len(t)  # loop

        for k in range(len(b[0])):  # update time dep coeff
            bar[k].set_height(b[j, k])
            bar[k].set_color(color[j, k])

        ylab.set_text(rf"$b_k({int(t[j])}ms)$")
        saite.set_ydata(u[:, j])  # update string
        ts.set_text(r"$u(x, t)|_{t=" + f"{int(t[j])}" + r"ms}$")

    ani = animation.FuncAnimation(fig1, animate, interval=1, blit=False, save_count=len(t))

    fig1.tight_layout()

    # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Jacob Shaw'), bitrate=1800)
    # ani.save('1dWellengleichung.gif', writer=writer)

    # FIGURE 2 3D PLOT
    fig, ax = plt.subplots(num="1D WELLENGLEICHUNG", subplot_kw={"projection": "3d"})

    scale2 = 1
    fin = int(2 * len(t) / perioden)

    # einstellungen
    ax.set_xlim([0, m])
    ax.set_zlim([scale2 * scale * (-m / 2), scale2 * scale * m / 2])
    ax.set_xlabel("LAENGE [cm]")
    ax.set_ylabel("ZEIT [ms]")
    ax.set_zlabel("AUSLENKUNG [cm]")

    X, T = np.meshgrid(x, t[:fin])

    # plot
    ax.plot_surface(X, T, u[:, :fin].T, edgecolor="k", lw=0.5, rstride=8, cstride=8, alpha=0.3, color="grey")

    plt.show()


if __name__ == '__main__':
    pass
