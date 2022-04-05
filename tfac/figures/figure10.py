""" A file to make Marc's plot. """

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from tensorly.decomposition import parafac
from .common import subplotLabel
from tensorpack import Decomposition
from tensorpack.plot import tfacr2x
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize
from ..dataHelpers import Tensor_LINCS_MEMA, reorder_table

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.markerscale"] = 1.5
matplotlib.rcParams['font.sans-serif'] = "Arial"


matplotlib.use('AGG')

ppfac = lambda x, r: parafac(x, rank=r, n_iter_max=100, tol=1e-9, linesearch=True)

def getsetup(figsize):
    sns.set(style="whitegrid", font_scale=1.0, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.2})
    fig = plt.figure(figsize=figsize, tight_layout=True)

    gs = fig.add_gridspec(3, 10)
    ax1 = fig.add_subplot(gs[0, :6]) # off
    ax2 = fig.add_subplot(gs[0, 6:]) # r2x
    ax3 = fig.add_subplot(gs[1, :2]) # comp1
    ax4 = fig.add_subplot(gs[1, 4:6]) # comp2
    ax5 = fig.add_subplot(gs[1, 8:]) # comp3
    ax6 = fig.add_subplot(gs[2, :2]) # buffer

    return ([ax1, ax2, ax3, ax4, ax5, ax6], fig)

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getsetup((12, 12))

    ax[0].axis('off')


    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    tm = Decomposition(MCF10A.to_numpy(), max_rr=5, method=ppfac)
    tm.perform_tfac()
    tm.perform_PCA(flattenon=2)

    tfacr2x(ax[1], tm)
    ax[1].set_title("")
    for i in range(5):
        ax[1].get_xticklabels()[i].set_fontsize(15)
        ax[1].get_yticklabels()[i].set_fontsize(15)

    ax[1].get_yticklabels()[5].set_fontsize(15)
    ax[1].set_ylabel("R2X", fontsize=15)
    ax[1].set_xlabel("Components #", fontsize=15)

    components_MEMA(MCF10A, [ax[2], ax[3], ax[4], ax[5]])

    return f

def components_MEMA(tensor, ax):
    """ Plots most significant components separately in the supplementary figures. """
    tFac = parafac(tensor.to_numpy(), 4, n_iter_max=2000, linesearch=True, tol=1e-9) # tensor is xarray type
    tFac = cp_flip_sign(tFac, 2)
    tFac = cp_normalize(tFac)
    labels = [f"{i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)]

    for ii in range(tensor.ndim):
        facZero = pd.DataFrame(tFac.factors[ii], columns=labels, index=tensor.coords[tensor.dims[ii]])
        facZero = reorder_table(facZero)

        feature_l_ind = facZero.abs().nlargest(10, facZero.columns).index

        g0 = sns.heatmap(facZero.loc[feature_l_ind.sort_values()], ax=ax[ii], cmap="PiYG", cbar=0, center=0, vmin=-1, vmax=1)
        g0.set_title(str(tensor.dims[ii]))

    g0 = sns.heatmap(facZero.loc[feature_l_ind.sort_values()], ax=ax[ii+1], cbar=True, cmap="PiYG", center=0, vmin=-1, vmax=1)

    for i in range(4):
        ax[0].get_xticklabels()[i].set_fontsize(15)
        ax[1].get_xticklabels()[i].set_fontsize(15)
        ax[2].get_xticklabels()[i].set_fontsize(15)
    for i in range(10):
        ax[0].get_yticklabels()[i].set_fontsize(12)
        ax[1].get_yticklabels()[i].set_fontsize(12)
        ax[2].get_yticklabels()[i].set_fontsize(12)
    for i in range(3):
        ax[i].set_title(ax[i].get_title(), fontsize=15)