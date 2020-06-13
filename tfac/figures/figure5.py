"""
This creates Figure 5 - MRSA R2X for parafac2.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorly.decomposition import parafac2
import tensorly as tl
from tensorly.metrics.regression import variance as tl_var
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import form_MRSA_tensor
from ..tensor import R2Xparafac2


tl.set_backend("numpy")
components = 38
variance = 1

tensor_slices, cytokines, geneIDs = form_MRSA_tensor(variance)

AllR2X = []
for i in range(1, components + 1):
    parafac2tensor = None
    best_error = np.inf
    for run in range(1):
        decomposition, errors = parafac2(tensor_slices, i, return_errors=True)
        if best_error > errors[-1]:
            best_error = errors[-1]
            parafac2tensor = decomposition
    AllR2X.append(R2Xparafac2(tensor_slices, parafac2tensor))
df = pd.DataFrame(AllR2X)

comps = []
for i in range(1, components + 1):
    comps.append(i)
df['Component'] = comps

df.columns = ['Cytokines', 'GeneIDs', 'Component']
test = pd.melt(df, id_vars=['Component'])



def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 7), (1, 1))
    b = sns.scatterplot(data=test, x='Component', y='value', hue='variable', style='variable', ax=ax[0], s=100)
    b.set_xlabel("Component", fontsize=20)
    b.set_ylabel("R2X", fontsize=20)
    b.tick_params(labelsize=15)
    plt.legend(prop={'size': 15})
    ax[0].set_ylim(0, 1)

    return f

