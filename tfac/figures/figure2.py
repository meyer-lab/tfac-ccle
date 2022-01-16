"""
This creates Figure 2 - protein factors.
(a) heatmap of the proteins in four subplots
(b) ?
(c-e) components vs time for various treatments.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from tensorpack import perform_CP
from ..dataHelpers import form_tensor, proteinNames


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((20, 20), (3, 4))

    tensor, drugs, times = form_tensor()

    tFac = perform_CP(tensor, r=6)

    # proteins - subplot (a)
    proteins = [pd.DataFrame(tFac.factors[2][57367:57450], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=proteinNames()[0:83]),
    pd.DataFrame(tFac.factors[2][57450:57520], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=proteinNames()[83:153]),
    pd.DataFrame(tFac.factors[2][57520:57590], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=proteinNames()[153:223]),
    pd.DataFrame(tFac.factors[2][57590:], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=proteinNames()[223:])]

    for i in range(4):
        g = sns.heatmap(proteins[i], cmap="PRGn", center=0, yticklabels=True, cbar=True, vmin=-0.01, vmax=0.01, ax=ax[i])
        g.set_yticklabels(g.get_yticklabels(), rotation = 0)
        g.set_title("proteins")

    # ? - subplot (b)

    # components vs time - subplots (c-f)

    # Add subplot labels
    subplotLabel(ax)

    return f
