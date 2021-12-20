"""
This creates Figure 1 - Partial Tucker R2X and Gene Projected Factors R2X.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from tensorpack import Decomposition, perform_CP
from tensorpack.plot import *
from ..dataHelpers import form_tensor, proteinNames


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((20, 20), (2, 4))

    tensor, drugs, time = form_tensor()
    ## With new figure making function in tensorpack
    t = Decomposition(tensor, max_rr=7)
    t.perform_tfac()
    tfacr2x(ax[0], t)

    tFac = perform_CP(tensor, r=6, tol=1e-6)
    treatment = pd.DataFrame(tFac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=list(drugs))
    time = pd.DataFrame(tFac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=time)
    proteins1 = pd.DataFrame(tFac.factors[2][57367:57450], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=proteinNames()[0:83])
    proteins2 = pd.DataFrame(tFac.factors[2][57450:57520], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=proteinNames()[83:153])
    proteins3 = pd.DataFrame(tFac.factors[2][57520:57590], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=proteinNames()[153:223])
    proteins4 = pd.DataFrame(tFac.factors[2][57590:], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=proteinNames()[223:])
    genes = pd.DataFrame(tFac.factors[2][0:80], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)])

    g1 = sns.heatmap(treatment, cmap="PRGn", center=0, xticklabels=True, yticklabels=True, cbar=False, vmin=-1.0, vmax=1.0, ax=ax[1])
    g1.set_yticklabels(g1.get_yticklabels(), rotation = 0)
    g1.set_title("treatments")
    g2 = sns.heatmap(time, cmap="PRGn", center=0, yticklabels=True, cbar=True, vmin=-1.0, vmax=1.0, ax=ax[2])
    g2.set_yticklabels(g2.get_yticklabels(), rotation = 0)
    g2.set_title("time")
    g3 = sns.heatmap(genes, cmap="PRGn", center=0, yticklabels=True, cbar=True, vmin=-1.0, vmax=1.0, ax=ax[3])
    g3.set_yticklabels(g3.get_yticklabels(), rotation = 0)
    g3.set_title("genes")
    g4 = sns.heatmap(proteins1, cmap="PRGn", center=0, yticklabels=True, cbar=True, vmin=-0.01, vmax=0.01, ax=ax[4])
    g4.set_yticklabels(g4.get_yticklabels(), rotation = 0)
    g4.set_title("proteins")
    g5 = sns.heatmap(proteins2, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=-0.01, vmax=0.01, ax=ax[5])
    g5.set_yticklabels(g5.get_yticklabels(), rotation = 0)
    g5.set_title("proteins")
    g6 = sns.heatmap(proteins3, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=-0.01, vmax=0.01, ax=ax[6])
    g6.set_yticklabels(g6.get_yticklabels(), rotation = 0)
    g6.set_title("proteins")
    g7 = sns.heatmap(proteins4, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=-0.01, vmax=0.01, ax=ax[7])
    g7.set_yticklabels(g7.get_yticklabels(), rotation = 0)
    g7.set_title("proteins")
    # Add subplot labels
    subplotLabel(ax)

    return f
