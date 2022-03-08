"""
Figure to show each components largest weights for MEMA Data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize
from ..dataHelpers import import_LINCS_MEMA, proteinNames, reorder_table

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((20, 12), (3, 5))

    MCF10A, ligand, ecm, meas = import_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")

    plot_components(MCF10A, ligand, ecm, meas, ax)

    return f 

def plot_components(tensor, ligand, ecm, meas, ax): 

    tFac = parafac(tensor, 5, n_iter_max=2000, linesearch=True, tol=1e-9)
    tFac = cp_flip_sign(tFac, 2)
    tFac = cp_normalize(tFac)

    fac1 = pd.DataFrame(tFac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=ligand) # ligans
    fac2 = pd.DataFrame(tFac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=ecm) # ecms
    fac3 = pd.DataFrame(tFac.factors[2], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)], index=meas) # measuremetns

    fac1 = reorder_table(fac1)
    fac2 = reorder_table(fac2)
    fac3 = reorder_table(fac3)

    for c, col in enumerate(fac1.keys()):
        ligands = fac1[[col]]
        ecms = fac2[[col]]
        measurements = fac3[[col]]

        ligands_l_ind = ligands.abs().nlargest(20, col).index
        ecms_l_ind = ecms.abs().nlargest(20, col).index
        measurements_l_ind = measurements.abs().nlargest(20, col).index

        g0 = sns.heatmap(ligands.loc[ligands_l_ind].sort_values([col]), ax=ax[c], cmap="PRGn", center=0, vmin=-1, vmax=1)
        g0.set_title(f"Ligands, {col}")
        g1 = sns.heatmap(ecms.loc[ecms_l_ind].sort_values([col]), ax=ax[5+c], cmap="PRGn", center=0, vmin=-1, vmax=1)
        g1.set_title(f"ECMs, {col}")
        g2 = sns.heatmap(measurements.loc[measurements_l_ind].sort_values([col]), ax=ax[10+c], cmap="PRGn", center=0, vmin=-1, vmax=1)
        g2.set_title(f"Measurements, {col}")
