"""
This creates Figure 2 - protein factors.
(a) heatmap of the proteins in four subplots
(b) ?
(c-e) components vs time for various treatments.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .common import subplotLabel
from tensorpack import perform_CP
from tensorpack.cmtf import cp_normalize
from ..dataHelpers import Tensor_LINCS_CCLE, proteinNames, reorder_table


def getsetup(figsize):
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    gs = fig.add_gridspec(3, 6)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])
    return ([ax1, ax2, ax3, ax4], fig)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getsetup((50, 10))

    tensor = import_LINCS_CCLE()

    tFac = perform_CP(tensor.to_numpy(), r=5, maxiter=2000, progress=True)
    tFac = cp_normalize(tFac)

    treatment = pd.DataFrame(tFac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=tensor.coords[tensor.dims[0]])
    tm = pd.DataFrame(tFac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=tensor.coords[tensor.dims[1]])

    g1 = sns.heatmap(treatment, cmap="PRGn", center=0, yticklabels=True, cbar=True, ax=ax[0])
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0)
    g1.set_title("treatment")

    g2 = sns.heatmap(tm, cmap="PRGn", center=0, yticklabels=True, cbar=True, ax=ax[1])
    g2.set_yticklabels(g2.get_yticklabels(), rotation=0)
    g2.set_title("times")

    facTwo1 = pd.DataFrame(tFac.factors[2][:295], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=proteinNames())  # proteins
    RNAseq = pd.read_csv("tfac/data/ohsu/module_expression.csv", delimiter=",")
    facTwo2 = pd.DataFrame(tFac.factors[2][295:], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=list(RNAseq["Unnamed: 0"]))  # gene modules

    facTwo1 = reorder_table(facTwo1)
    facTwo2 = reorder_table(facTwo2)

    sns.heatmap(facTwo1.T, ax=ax[2], cmap="PRGn", center=0)
    sns.heatmap(facTwo2.T, ax=ax[3], cmap="PRGn", center=0)

    # Add subplot labels
    subplotLabel(ax)

    return f
