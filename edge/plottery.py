# -*- coding: utf-8 -*-
"""
    hosts the function `fancy`
"""

import numpy  as np
import pandas as pd
import pylab  as pl

from matplotlib.colors import ListedColormap, Normalize


RED   = lambda alpha : (1.00, 0.33, 0.00, alpha)
GRAY  = lambda scale, alpha : (scale, scale, scale, alpha)
GREEN = lambda alpha : (0.00, 0.75, 0.00, alpha)
BLUE  = lambda alpha : (0.00, 0.33, 1.00, alpha)


def fancy(data, flavour, index, y=None, **kwargs):
    """
        generates fancy plots in different flavours
    """
    N = data.shape[0]
    dim = int(np.sqrt((data.shape[1] - 16) / 2))
    x_max = f"x_{dim - 1}{dim - 1}"
    z_max = f"z_{dim - 1}{dim - 1}"

    if(np.isscalar(flavour)):
        flavour = [flavour]
    if(np.isscalar(index)):
        index = [index]
    elif(index is None):
        index = [0]
    num = max(len(flavour), len(index))
    if(len(flavour) < num):
        flavour = list(flavour) * num
    if(len(index) < num):
        index = list(index) * num

    bias = kwargs.pop("bias", 0.4)
    bins = kwargs.pop("bins", 42)

    pl.set_cmap(pl.cm.binary)
    my_cmap = ListedColormap([[0, 0, 0, 0], [0, 0, 0, 1]])

    fig, ax = pl.subplots(nrows=1, ncols=min(num, 3), dpi=360)
    fig.set_figwidth(fig.get_figwidth() * 2.5 + (4.4 * (num < 3))) ## FIXME

    x_positions, y_positions = np.meshgrid(np.arange(dim + 1) / dim, np.arange(dim + 1) / dim)

    for counter in range(num):
        pixels    = np.array(data.loc[index[counter], "x_00":x_max]).reshape((dim, dim))
        shuffled  = np.array(data.loc[index[counter], "z_00":z_max]).reshape((dim, dim))
        slope     = data.loc[index[counter], "y_slope"]
        intercept = data.loc[index[counter], "y_intercept"]
        Ax, Ay    = data.loc[index[counter], "y_Ax"], data.loc[index[counter], "y_Ay"]
        Bx, By    = data.loc[index[counter], "y_Bx"], data.loc[index[counter], "y_By"]
        area      = data.loc[index[counter], "y_area"]

        if(flavour[counter] == "initial"):
            b1 = bias / 2
            b2 = 1 - b1
            ax[counter].scatter(
                    [(Ax + Bx) / 2], [(Ay + By) / 2], c=[RED(1)], s=100,
                    marker='o', zorder=2)
            ax[counter].plot(
                    [0, 1], [intercept, intercept + slope],
                    color=RED(1), linewidth=kwargs.get("linewidth", 3), zorder=1)
            ax[counter].plot([b1, b2, b2, b1, b1], [b1, b1, b2, b2, b1],
                    color=GRAY(0.5, 1), linewidth=1, zorder=0)
            ax[counter].set_title(f"#{index[counter]:04d}")
            ax[counter].set_aspect("equal")
            ax[counter].set_xlim((0, 1))
            ax[counter].set_xlabel("x")
            ax[counter].set_ylim((0, 1))
            ax[counter].set_ylabel("y")

        elif(flavour[counter] == "integral"):
            ax[counter].plot([0, 1], [intercept, intercept + slope],
                    color=RED(1), linewidth=kwargs.get("linewidth", 3), zorder=2)
            ax[counter].pcolormesh(x_positions, y_positions, np.zeros(pixels.shape), cmap=my_cmap,
                    norm=Normalize(vmin=0, vmax=1), edgecolor=GRAY(0.5, 1),
                    linewidth=1, zorder=1)
            ax[counter].fill_between([0, 1], [intercept, intercept + slope],
                    color=GRAY(0, 1), linewidth=0, zorder=0)
            ax[counter].set_title(f"#{index[counter]:04d}")
            ax[counter].set_aspect("equal")
            ax[counter].set_xlim((0, 1))
            ax[counter].set_ylim((0, 1))
            ax[counter].tick_params(axis="both", which="both",
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        elif(flavour[counter] == "average"):
            pixels = np.mean(data.loc[:, "x_00":x_max], axis=0).values.reshape((dim, dim))
            ax[counter].pcolormesh(x_positions, y_positions, pixels,
                    norm=Normalize(vmin=0, vmax=1), edgecolor=GRAY(0.5, 1),
                    linewidth=1, zorder=0)
            ax[counter].set_title("average")
            ax[counter].set_aspect("equal")
            ax[counter].set_xlim((0, 1))
            ax[counter].set_ylim((0, 1))
            ax[counter].tick_params(axis="both", which="both",
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        elif(flavour[counter] == "line"):
            ax[counter].plot([0, 1], [intercept, intercept + slope],
                    color=RED(1), zorder=1, **kwargs)
            ax[counter].pcolormesh(x_positions, y_positions, pixels,
                    norm=Normalize(vmin=0, vmax=1), edgecolor=GRAY(0.5, 1),
                    linewidth=1, zorder=0)
            ax[counter].set_title(f"#{index[counter]:04d}")
            ax[counter].set_aspect("equal")
            ax[counter].set_xlim((0, 1))
            ax[counter].set_ylim((0, 1))
            ax[counter].tick_params(axis="both", which="both",
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        elif(flavour[counter] == "lines"):
            y1 = np.array(data.loc[:, "y_intercept"])
            y2 = np.array(data.loc[:, "y_slope"]) + y1
            for ii in range(N):
                ax[counter].plot([0,1], [y1[ii], y2[ii]], color=BLUE(0.02), zorder=0)
            ax[counter].set_title("straight lines")
            ax[counter].set_aspect("equal")
            ax[counter].set_xlim((0, 1))
            ax[counter].set_ylim((0, 1))
            ax[counter].tick_params(axis="both", which="both",
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        elif(flavour[counter] == "angles"):
            p = np.array(data.loc[:, "y_angle"])
            ax[counter].hist(p, bins=bins, range=(-np.pi / 2, np.pi / 2), color=GRAY(0, 0.5), zorder=1)
            m = N / bins
            s = 3 * np.sqrt(m)
            ax[counter].plot([-2, 2], [m, m], color=BLUE(1), zorder=0, **kwargs)
            ax[counter].fill_between([-2, 2], m + s, y2=m - s,
                    color=BLUE(0.25), linewidth=0, zorder=2)
            ax[counter].set_title("histogram of angles")
            ax[counter].set_xlim((-np.pi / 2, np.pi / 2))
            #ax[counter].set_xlabel("angle")
            ax[counter].set_xticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2], minor=False)
            ax[counter].set_xticklabels(["$-\pi/2$" ,"$-\pi/4$","0","$\pi/4$","$\pi/2$"])
            ax[counter].set_ylim(N / bins * 0.5, N / bins * 1.5)

        elif(flavour[counter] == "area"):
            if(y is not None):
                y_area = y[index[counter]]
                ax[counter].fill_between([-2, 2], y_area, y2=-1,
                        facecolor=GRAY(1, 0), edgecolor=GREEN(1), zorder=3, hatch="/", linewidth=1)
            ax[counter].plot([-2, 2], [area, area],
                    color=RED(1), zorder=2, **kwargs)
            ax[counter].pcolormesh(x_positions, y_positions, pixels,
                    norm=Normalize(vmin=0, vmax=1), edgecolor=GRAY(0.5, 1),
                    linewidth=1, zorder=0)
            ax[counter].set_title(f"#{index[counter]:04d}")
            ax[counter].set_aspect("equal")
            ax[counter].set_xlim((0, 1))
            ax[counter].set_ylim((0, 1))
            ax[counter].set_ylabel("area")
            ax[counter].tick_params(axis="both", which="both",
                    bottom=False, top=False, left=True, right=False,
                    labelbottom=False, labeltop=False, labelleft=True, labelright=False)

        elif(flavour[counter] == "areas"):
            a = np.array(data.loc[:, "y_area"])
            ax[counter].hist(a, bins=bins, range=(0, 1), color=GRAY(0, 0.5), zorder=1)
            m = N / bins
            s = 3 * np.sqrt(m)
            ax[counter].plot([-2, 2], [m, m], color=BLUE(1), zorder=0, **kwargs)
            ax[counter].fill_between([-2, 2], m + s, y2=m - s,
                    color=BLUE(0.25), linewidth=0, zorder=2)
            ax[counter].set_title("histogram of areas")
            ax[counter].set_xlim((0, 1))
            #ax[counter].set_xlabel("area")
            ax[counter].set_ylim(N / bins * 0.5, N / bins * 1.5)

        elif(flavour[counter] == "shuffle"):
            ax[counter].pcolormesh(x_positions, y_positions, shuffled,
                    norm=Normalize(vmin=0, vmax=1), edgecolor=GRAY(0.5, 1),
                    linewidth=1, zorder=0)
            ax[counter].set_title(f"#{index[counter]:04d} shuffled")
            ax[counter].set_aspect("equal")
            ax[counter].set_xlim((0, 1))
            ax[counter].set_ylim((0, 1))
            ax[counter].tick_params(axis="both", which="both",
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)


def gaze(flavour, inputs=None, **kwargs):
    """
        generates stunning plots in different flavours
    """
    if(np.isscalar(flavour)):
        flavour = [flavour]
    num = len(flavour)

    fig, ax = pl.subplots(nrows=1, ncols=min(num, 2), dpi=360)
    if(num == 1):
        ax = [ax]
    fig.set_figwidth(fig.get_figwidth() * 2.5) ## TODO

    losses  = kwargs.pop("losses", None)
    weights = kwargs.pop("weights", None)
    compare = kwargs.pop("compare", None)
    angles  = kwargs.pop("angles", None)

    for counter in range(num):

        if(flavour[counter] == "loss"):
            epochs = len(losses)
            ax[counter].semilogy(
                    np.arange(epochs) + 1, losses,
                    color=BLUE(1), zorder=0, **kwargs)
            ax[counter].set_title("loss evolution")
            ax[counter].set_xlim((0, epochs))
            ax[counter].set_xlabel("epochs")
            ax[counter].set_ylabel("loss")

        if(flavour[counter] == "weights"):
            epochs = len(weights[0])
            for ii in range(weights.shape[0]):
                ax[counter].plot(
                        np.arange(epochs) + 1, weights[ii, :],
                        color=BLUE(0.1), zorder=1, **kwargs)
            ax[counter].plot(
                    [0, epochs], np.ones((2,)) / len(weights),
                    color=GRAY(0.5, 1), zorder=0, **kwargs)
            ax[counter].set_title("weight evolution")
            ax[counter].set_xlim((0, epochs))
            ax[counter].set_xlabel("epochs")
            ax[counter].set_ylim((-0.1, 0.1))
            ax[counter].set_yticks(np.linspace(-0.1, 0.1, 5), minor=False)
            ax[counter].set_ylabel("weight")

        if(flavour[counter] == "compare"):
            sort = np.argsort(compare[0])
            true = compare[0][sort]
            pred = compare[1][sort]
            ax[counter].plot(
                    true, 1 - (true - pred),
                    color=BLUE(1), zorder=1, **kwargs)
            ax[counter].plot(
                    [0, 1], [1, 1],
                    color=GRAY(0.5, 1), zorder=0, **kwargs)
            ax[counter].set_title(f"{str(compare[2])} comparison")
            ax[counter].set_xlim((0, 1))
            ax[counter].set_xlabel(f"(true) {str(compare[2])}")
            ax[counter].set_ylabel(f"1 - (true - predicted) {str(compare[2])}")

        if(flavour[counter] == "depend"):
            sort = np.argsort(angles)
            true = compare[0][sort]
            pred = compare[1][sort]
            ax[counter].plot(
                    angles[sort], 1 - (true - pred),
                    color=BLUE(1), zorder=1, **kwargs)
            ax[counter].plot(
                    [0, 1], [1, 1],
                    color=GRAY(0.5, 1), zorder=0, **kwargs)
            ax[counter].set_title(f"{str(compare[2])} comparison")
            ax[counter].set_xlim((-np.pi / 2, np.pi / 2))
            ax[counter].set_xlabel("angle")
            ax[counter].set_xticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2], minor=False)
            ax[counter].set_xticklabels(["$-\pi/2$" ,"$-\pi/4$","0","$\pi/4$","$\pi/2$"])
            ax[counter].set_ylabel(f"1 - (true - predicted) {str(compare[2])}")

