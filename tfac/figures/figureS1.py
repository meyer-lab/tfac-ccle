"""
Figure to show each components largest weights for CCLE Data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .common import subplotLabel
from tensorpack import perform_CP
from tensorpack.cmtf import cp_normalize
from ..dataHelpers import import_LINCS_CCLE, proteinNames, reorder_table

def getsetup(figsize):
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    gs = fig.add_gridspec(2, 5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[0, 4])
    
    ax7 = fig.add_subplot(gs[1, 0])
    ax8 = fig.add_subplot(gs[1, 1])
    ax9 = fig.add_subplot(gs[1, 2])
    ax10 = fig.add_subplot(gs[1, 3])
    ax11 = fig.add_subplot(gs[1, 4])

    return ([ax1, ax2, ax3, ax4, ax5], [ax7, ax8, ax9, ax10, ax11], fig)

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, ax1, f = getsetup((50, 10))

    tensor, drugs, times = import_LINCS_CCLE()

    tFac = perform_CP(tensor, r=5, maxiter=2000, progress=True)
    tFac = cp_normalize(tFac)

    facTwo1 = pd.DataFrame(tFac.factors[2][:295], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=proteinNames()) # proteins
    RNAseq = pd.read_csv("tfac/data/ohsu/module_expression.csv", delimiter=",")
    facTwo2 = pd.DataFrame(tFac.factors[2][295:], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=list(RNAseq["Unnamed: 0"])) # gene modules

    facTwo1 = reorder_table(facTwo1)
    facTwo2 = reorder_table(facTwo2)

    pthreshold = 0.2
    gthreshold = 0.5
    for axis1, axis2, col in zip(ax, ax1, facTwo1.keys()):
        prots = facTwo1[[col]]
        geneMods = facTwo2[[col]]
        prots_thresholded = prots[((prots > pthreshold) | (prots < -pthreshold))].dropna()
        gene_thresholded = geneMods[((geneMods > gthreshold) | (geneMods < -gthreshold))].dropna()

        g0 = sns.heatmap(prots_thresholded.T, ax=axis1, cmap="PRGn", center=0, vmin=-1, vmax=1)
        g0.set_title(f"Proteins, {col}")
        g1 = sns.heatmap(gene_thresholded.T, ax=axis2, cmap="PRGn", center=0, vmin=-1, vmax=1)
        g1.set_title(f"Gene Modules, {col}")

    return f 
