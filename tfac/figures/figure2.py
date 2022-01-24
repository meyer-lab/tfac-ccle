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
    ax, f = getSetup((8, 16), (3, 2))

    tensor, drugs, times = form_tensor()

    tFac = perform_CP(tensor, r=4)

    treatment = pd.DataFrame(tFac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=drugs)
    tm = pd.DataFrame(tFac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=times)

    g1 = sns.heatmap(treatment, cmap="PRGn", center=0, yticklabels=True, cbar=True, vmin=-1., vmax=1., ax=ax[0])
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0)
    g1.set_title("treatment")

    g2 = sns.heatmap(tm, cmap="PRGn", center=0, yticklabels=True, cbar=True, vmin=-1., vmax=1., ax=ax[1])
    g2.set_yticklabels(g2.get_yticklabels(), rotation=0)
    g2.set_title("times")

    for i in range(2, 6):
        ax[i].axis("off")
    # ? - subplot (b)

    # components vs time - subplots (c-f)

    # Add subplot labels
    subplotLabel(ax)

    return f
