import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import os


def graph_niveaulinien(data, nam, dire, save, show):

    # Abbildung
    fig, ax = plt.subplots(num=f"Niveaulinien {nam[1]}")

    # Einstellungen
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(nam[0])
    ax.set_xlim(data[0].min(), data[0].max())
    ax.set_ylim(data[1].min(), data[1].max())
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # Niveaulinien
    ax.contour(data[0], data[1], data[2], 20, colors='k')

    fig.tight_layout()

    if save:
        plt.savefig(f"{dire}Niveaulinien-{nam[1]}.png")

    if show:
        plt.show()

    return True


def graph_kurven(data, nam, dire, ar, save, show):

    # Abbildung
    fig, ax = plt.subplots(num=f"Kurven {nam[1]}")

    # Einstellungen
    if ar:
        ax.set_aspect('equal', adjustable='box')
    ax.set_title(nam[0])
    ax.set_xlim(data[0].min(), data[0].max())
    ax.set_ylim(data[1].min(), data[1].max())
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # Plot
    ax.plot(data[0], data[1], c='k')

    fig.tight_layout()

    if save:
        plt.savefig(f"{dire}Kurven-{nam[1]}.png")

    if show:
        plt.show()

    return True


def graph_3d_kurven(data, nam, dire, save, show):

    # Abbildung
    fig, ax = plt.subplots(num=f"Kurven {nam[1]}", subplot_kw={"projection": "3d"})

    # Einstellungen
    # ax.set_aspect('equal', adjustable='box')
    ax.set_title(nam[0])
    ax.set_xlim(data[0].min(), data[0].max())
    ax.set_ylim(data[1].min(), data[1].max())
    ax.set_ylim(data[2].min(), data[2].max())
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")

    # 3D Plot
    ax.plot(data[0], data[1], data[2], c='k')

    fig.tight_layout()

    if save:
        plt.savefig(f"{dire}Kurven-{nam[1]}.png")

    if show:
        plt.show()

    return True


def graph_vektorfeld(data, nam, dire, save, show):

    # Abbildung
    fig, ax = plt.subplots(num=f"Vektorfeld  {nam[1]}")

    # Einstellungen
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(nam[0])
    ax.set_xlim(data[0].min(), data[0].max())
    ax.set_ylim(data[1].min(), data[1].max())
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # Vektorfeld
    ax.quiver(data[0], data[1], data[2], data[3])

    fig.tight_layout()

    if save:
        plt.savefig(f"{dire}Vektorfeld-{nam[1]}.png")

    if show:
        plt.show()

    return True


def niveau(dire, save, show):

    a1, b1, dx1 = -2 * pi, 2 * pi, 0.01
    a2, b2, dx2 = -2 * pi, 2 * pi, 0.01

    x1 = np.arange(a1, b1 + dx1, dx1)
    x2 = np.arange(a2, b2 + dx2, dx2)

    x, y = np.meshgrid(x1, x2, indexing='ij')

    f1 = x ** 2 - y ** 3
    f2 = (x ** 2 - 1) ** 2 + y ** 2
    f3 = sin(x) * cos(y)

    nam = [
        [r"$f(x,y)=x^2-y^3$", "a"],
        [r"$f(x,y)=(x^2-1)^2+y^2$", "b"],
        [r"$f(x,y)=\sin(x)\cdot \cos(y)$", "c"]
    ]

    for fi, nami in zip([f1, f2, f3], nam):
        data = [x, y, fi]
        graph_niveaulinien(data, nami, dire, save, show)

    return True


def kurven(dire, save, show):

    a1, b1, dt1 = -2 * pi, 2 * pi, 0.01
    a2, b2, dt2 = 0, 2 * pi, 0.01

    t1 = np.arange(a1, b1 + dt1, dt1)
    t2 = np.arange(a2, b2 + dt2, dt2)

    f1 = np.ones((2, t1.size))
    f2 = f1.copy()
    f3 = np.ones((3, t2.size))

    f1[0], f1[1] = t1**2, t1**3
    f2[0], f2[1] = sin(3 * t1) * cos(t1), sin(3 * t1) * sin(t1)
    f3[0], f3[1], f3[2] = t2, t2 * cos(t2), t2 * sin(t2)

    nam = [
        [r"$\gamma (x,y)=(t^2, t^3)$", "a"],
        [r"$\gamma (x,y)=(t, t\cdot \sin(t), t\cdot \cos(t))$", "b"],
        [r"$\gamma (x,y)=(\sin(3t)\cdot \cos(t), \sin(3t)\cdot \sin(t))$", "c"]
    ]

    for fi, nami, ari in zip([f1, f2], [nam[0], nam[2]], [False, True]):
        graph_kurven(fi, nami, dire, ari, save, show)

    graph_3d_kurven(f3, nam[1], dire, save, show)

    return True


def vektorfeld(dire, save, show):

    a1, b1, dx1 = -2 * pi, 2 * pi, 0.5
    a2, b2, dx2 = -2 * pi, 2 * pi, 0.5

    x1 = np.arange(a1, b1 + dx1, dx1)
    x2 = np.arange(a2, b2 + dx2, dx2)

    x, y = np.meshgrid(x1, x2, indexing='ij')

    f1 = np.ones((2, x.shape[0], x.shape[1]))
    f2 = f1.copy()
    f3 = f1.copy()

    eins = np.ones(x.shape)  # multiplikatives neutrales Element
    null = np.zeros(x.shape)  # additives neutrales Element

    f1[0], f1[1] = y, null  # sheering (Spielkarten)
    f2[0], f2[1] = x, y  # Quelle
    f3[0], f3[1] = -y, x  # Wirbel

    nam = [
        [r"$f(x,y)=(y, 0)$", "a"],
        [r"$f(x,y)=(x, y)$", "b"],
        [r"$f(x,y)=(-y, x)$", "c"]
    ]

    for fi, nami in zip([f1, f2, f3], nam):
        data = [x, y, fi[0], fi[1]]
        graph_vektorfeld(data, nami, dire, save, show)

    return True


def run(save=False, show=True):
    # plt.rcParams['figure.dpi'] = 300  # mag nicht
    plt.rcParams['savefig.dpi'] = 300  # sehr gut! - 80 fuer kleine bilder
    dir_path = os.path.dirname(os.path.realpath(__file__))

    dire = f"{dir_path}/H1/"
    if not os.path.exists(dire):
        os.makedirs(dire)

    niveau(dire, save, show)
    kurven(dire, save, show)
    vektorfeld(dire, save, show)

    return True


if __name__ == "__main__":
    run()
