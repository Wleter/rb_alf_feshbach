from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt

def plot() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")

    return fig, ax

@dataclass
class AxesArray:
    array: Any
    nrows: int
    ncols: int
    
    def __getitem__(self, key) -> Axes:
        return self.array[key]
    
    def __iter__(self):
        return AxesIter(self, 0)

@dataclass
class AxesIter:
    axes: AxesArray
    current: int

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.axes.ncols * self.axes.nrows:
            raise StopIteration
        else:
            if self.axes.ncols == 1 or self.axes.nrows == 1:
                self.current += 1

                return self.axes[self.current - 1]
            else:
                j = self.current % self.axes.ncols
                i = self.current // self.axes.ncols

                self.current += 1
                return self.axes[i, j]

def plot_many(nrows: int, ncols: int, shape = None, sharex = False, sharey = False) -> tuple[Figure, AxesArray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=shape, sharex = sharex, sharey = sharey)

    axes: AxesArray = AxesArray(axes, nrows, ncols)
    if ncols == 1 or nrows == 1:
        for i in range(ncols * nrows):
            axes[i].grid()
            axes[i].tick_params(which='both', direction="in")
    else:
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].grid()
                axes[i, j].tick_params(which='both', direction="in")

    return fig, axes

def plot_scattering(ax: Axes, x, y_re, y_im, label: str = ""):
    lines = ax.plot(x, y_re, label=label)
    ax.plot(x, y_im, linestyle="--", color=lines[0].get_color())

def load(filename: str, delimiter: str = '\t', skiprows:int = 1):
    data = np.loadtxt(f'{filename}', delimiter=delimiter, skiprows=skiprows)

    return data

def plot_surface(
    r, 
    potential_dec, 
    angle_no = 100, 
    max_val = 100,
    add_0 = True,
    levels = 50,
    fig_ax: tuple[Figure, Axes] | None = None
) -> tuple[Figure, Axes]:
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    polar = np.linspace(-np.pi, np.pi, angle_no)

    potential_mesh = np.zeros((len(r), len(polar)))
    for i in range(len(r)):
        potential_mesh[i, :] = np.polynomial.legendre.legval(np.cos(polar), potential_dec[i, :])

    max_val = 100
    if add_0:
        # add (0, 0)
        r = np.concatenate(([0], r))
        polar = np.concatenate(([0], polar))
        potential_mesh = np.concatenate((potential_mesh[:, 0:1], potential_mesh), axis=1)
        potential_mesh = np.concatenate((max_val * np.ones_like(potential_mesh[0:1, :]), potential_mesh), axis=0)

    potential_mesh = np.clip(potential_mesh, -np.inf, max_val)

    ax.contourf(polar, r, potential_mesh, levels=levels)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    return fig, ax
