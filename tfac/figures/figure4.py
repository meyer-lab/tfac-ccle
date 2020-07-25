"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import form_parafac2_tensor, ohsu_var
from ..tensor import OHSU_parafac2_decomp, R2Xparafac2, projections_to_factors
import matplotlib.pyplot as plt


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 3
    col = 6
    ax, f = getSetup((24, 11), (row, col))
    R2X_OHSU(ax[0])
    subplotLabel(ax)
    OHSU_comp_plots(10)
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

def OHSU_comp_plots(comps):
    p2slices, treatmentsTime, proteins, chromosomes, IFproteins, histones, geneExpression, RNAGenes, Rproteins = form_parafac2_tensor()
    p2slicesB = ohsu_var(p2slices)
    parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, 5)
    weights, transform = projections_to_factors(parafac2tensor)
    C = parafac2tensor[1][2]
    D = parafac2tensor[2][0]
    components = comps
    treatments = ['BMP2_24', 'BMP2_48', 'EGF_24', 'EGF_48', 'HGF_24', 'HGF_48', 'IFNg_24', 'IFNg_48', 'OSM_24', 'OSM_48','PBS_24','PBS_48','TGFb_24','TGFb_48']
    df = pd.DataFrame(C, index = treatments)
    trmt = df.to_numpy()
    for x in range(comps):
        BMP, EGF, HGF, IFNg, OSM, PBS, TGFb = [], [], [], [], [], [], []
        plt.figure(x)
        plt.figure(figsize=(10,10))
        plt.xticks([0,2], ['0','24','48'])
        plt.xlabel('Time (hr)')
        plt.title('Component ' + str(x+1))
        for y in range(7):
            if y == 0:
                BMP.append(trmt[(y*2), x])
                BMP.append(trmt[(y*2)+1,x])
            elif y == 1:
                EGF.append(trmt[(y*2), x])
                EGF.append(trmt[(y*2)+1,x])
            elif y == 2:
                HGF.append(trmt[(y*2), x])
                HGF.append(trmt[(y*2)+1, x])
            elif y == 3:
                IFNg.append(trmt[(y*2), x])
                IFNg.append(trmt[(y*2)+1, x])
            elif y == 4:
                OSM.append(trmt[(y*2), x])
                OSM.append(trmt[(y*2)+1, x])
            elif y == 5:
                PBS.append(trmt[(y*2), x])
                PBS.append(trmt[(y*2)+1, x])
            elif y == 6:
                TGFb.append(trmt[(y*2), x])
                TGFb.append(trmt[(y*2)+1, x])
        plt.plot(BMP)
        plt.plot(EGF)
        plt.plot(HGF)
        plt.plot(IFNg)
        plt.plot(OSM)
        plt.plot(PBS)
        plt.plot(TGFb)
        plt.legend(['BMP', 'EGF', 'HGF', 'IFNg', 'OSM', 'PBS', 'TGFb'])
    