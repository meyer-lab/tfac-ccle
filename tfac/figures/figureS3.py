"""
Figure to show each components largest weights for MEMA Data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .common import subplotLabel
from tensorpack import perform_CP
from tensorpack.cmtf import cp_normalize
from ..dataHelpers import import_LINCS_MEMA, proteinNames, reorder_table

def getsetup(figsize):
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    gs = fig.add_gridspec(3, 6)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[0, 4])
    ax6 = fig.add_subplot(gs[0, 5])
    
    ax7 = fig.add_subplot(gs[1, 0])
    ax8 = fig.add_subplot(gs[1, 1])
    ax9 = fig.add_subplot(gs[1, 2])
    ax10 = fig.add_subplot(gs[1, 3])
    ax11 = fig.add_subplot(gs[1, 4])
    ax12 = fig.add_subplot(gs[1, 5])

    ax13 = fig.add_subplot(gs[2, 0])
    ax14 = fig.add_subplot(gs[2, 1])
    ax15 = fig.add_subplot(gs[2, 2])
    ax16 = fig.add_subplot(gs[2, 3])
    ax17 = fig.add_subplot(gs[2, 4])
    ax18 = fig.add_subplot(gs[2, 5])

    return ([ax1, ax2, ax3, ax4, ax5, ax6], [ax7, ax8, ax9, ax10, ax11, ax12], [ax13, ax14, ax15, ax16, ax17, ax18], fig)

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, ax1, ax2, f = getsetup((50, 10))

    MCF10A, ligand, ecm, meas = import_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")

    tFac = perform_CP(MCF10A, r=6, maxiter=2000, progress=True)
    tFac = cp_normalize(tFac)

    fac1 = pd.DataFrame(tFac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=ligand) # ligans
    fac2 = pd.DataFrame(tFac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=ecm) # ecms
    fac3 = pd.DataFrame(tFac.factors[2], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=meas) # measuremetns

    fac1 = reorder_table(fac1)
    fac2 = reorder_table(fac2)
    fac3 = reorder_table(fac3)

    for axis1, axis2, axis3, col in zip(ax, ax1, ax2, fac1.keys()):
        ligands = fac1[[col]]
        ecms = fac2[[col]]
        measurements = fac3[[col]]

        ligands_l_ind = ligands.abs().nlargest(20, col).index
        ecms_l_ind = ecms.abs().nlargest(20, col).index
        measurements_l_ind = measurements.abs().nlargest(20, col).index

        g0 = sns.heatmap(ligands.loc[ligands_l_ind].sort_values([col]), ax=axis1, cmap="PRGn", center=0, vmin=-1, vmax=1)
        g0.set_title(f"Ligands, {col}")
        g1 = sns.heatmap(ecms.loc[ecms_l_ind].sort_values([col]), ax=axis2, cmap="PRGn", center=0, vmin=-1, vmax=1)
        g1.set_title(f"ECM, {col}")
        g2 = sns.heatmap(measurements.loc[measurements_l_ind].sort_values([col]), ax=axis3, cmap="PRGn", center=0, vmin=-1, vmax=1)
        g2.set_title(f"Measurements, {col}")

    return f 
