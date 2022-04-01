"""
Figure to show each components largest weights for CCLE Data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from tensorpack import perform_CP
from tensorpack.cmtf import cp_normalize
from ..dataHelpers import Tensor_LINCS_CCLE, proteinNames, reorder_table


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 5))

    tensor = import_LINCS_CCLE()

    tFac = perform_CP(tensor.to_numpy(), r=5, maxiter=2000, progress=True)
    tFac = cp_normalize(tFac)

    facTwo1 = pd.DataFrame(tFac.factors[2][:295], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[2].shape[1] + 1)], index=proteinNames())  # proteins
    RNAseq = pd.read_csv("tfac/data/ohsu/module_expression.csv", delimiter=",")
    facTwo2 = pd.DataFrame(tFac.factors[2][295:], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.factors[2].shape[1] + 1)], index=list(RNAseq["Unnamed: 0"]))  # gene modules

    facTwo1 = reorder_table(facTwo1)
    facTwo2 = reorder_table(facTwo2)

    for c, col in enumerate(facTwo1.keys()):
        prots = facTwo1[[col]]
        geneMods = facTwo2[[col]]

        prots_l_ind = prots.abs().nlargest(10, col).index
        gene_l_ind = geneMods.abs().nlargest(10, col).index

        g0 = sns.heatmap(prots.loc[prots_l_ind].sort_values([col]), ax=ax[c], cmap="PRGn", center=0, vmin=-1, vmax=1)
        g0.set_title(f"Proteins, {col}")
        g1 = sns.heatmap(geneMods.loc[gene_l_ind].sort_values([col]), ax=ax[5 + c], cmap="PRGn", center=0, vmin=-1, vmax=1)
        g1.set_title(f"Gene Modules, {col}")

    return f