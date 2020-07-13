"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import form_parafac2_tensor, ohsu_var
from ..tensor import OHSU_parafac2_decomp, R2Xparafac2


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 3
    col = 6
    ax, f = getSetup((24, 11), (row, col))
    R2X_OHSU(ax[0])
    subplotLabel(ax)
    return f


def R2X_OHSU(ax):
    '''Creates R2X for OHSU PARAFAC2'''
    p2slices, treatmentsTime, proteins, chromosomes, IFproteins, histones, geneExpression, RNAGenes, RPPAProteins = form_parafac2_tensor()
    p2slicesB = ohsu_var(p2slices)
    r2x = []
    components = 10
    for i in range(1, components + 1):
        parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, i)
        r2x.append(R2Xparafac2(p2slicesB, parafac2tensor))
    df = pd.DataFrame(r2x)
    comps = []
    for i in range(1, components + 1):
        comps.append(i)
    df['Component'] = comps
    df.columns = ['Proteins', 'Chromosomes', 'IFproteins', 'Histones', 'Gene Expression', 'RNA Genes', 'RPPA Proteins', 'Component']
    test = pd.melt(df, id_vars=['Component'])
    b = sns.scatterplot(data=test, x='Component', y='value', hue='variable', style='variable', ax=ax, s=100)
    b.set_xlabel("Component", fontsize=20)
    b.set_ylabel("R2X", fontsize=20)
    b.set_title("OHSU PARAFAC2")
    b.tick_params(labelsize=15)
    plt.legend(prop={'size': 15})
    ax[0].set_ylim(0, 1)
    