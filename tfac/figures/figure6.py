"""
This creates Figure 6. Includes OHSU R2X, treatments by components, and outliers by data slice (experiment) plots.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import getSetup  # subplotLabel
from ..Data_Mod import form_parafac2_tensor, ohsu_var
from ..tensor import OHSU_parafac2_decomp, R2Xparafac2, projections_to_factors

p2slices, treatmentsTime, proteins, chromosomes, IFproteins, histones, geneExpression, RNAGenes, Rproteins = form_parafac2_tensor()
p2slicesB = ohsu_var(p2slices)
components = 5
parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, components)
weights, transform = projections_to_factors(parafac2tensor)
LINCSproteins = transform[1][0]
#atacChr = transform[1][1]
GCPHistones = transform[1][3]
L1000GeneExp = transform[1][4]
#RNAGeneSeq = transform[1][5]
RPPAproteins = transform[1][6]
C = parafac2tensor[1][2]
df = pd.DataFrame(C[:-1, :])
df.columns = ["1", "2", "3", "4", "5"]
treatments = ['BMP2', 'BMP2', 'EGF', 'EGF', 'HGF', 'HGF', 'IFNg', 'IFNg', 'OSM', 'OSM', 'PBS', 'PBS', 'TGFb', 'TGFb', 'BMP2', 'EGF', 'HGF', 'IFNg', 'OSM', 'PBS', 'TGFb']
times = [24, 48] * 7 + [0] * 7
zeros = pd.DataFrame(C[-1:, :])
zeros.columns = ["1", "2", "3", "4", "5"]
for _ in range(7):
    df = pd.concat((df, zeros))
df["Times"] = times
df["Treatments"] = treatments


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 6
    col = 5
    ax, f = getSetup((40, 40), (row, col))
    R2X_OHSU(ax[0])
    for comp in range(1, 6):
        OHSU_comp_plots(df, comp, ax[comp + 4])
    for comps in range(5):
        proteinBoxPlot(ax[10 + comps], LINCSproteins[:, comps], comps + 1, proteins)
        proteinBoxPlot(ax[15 + comps], GCPHistones[:, comps], comps + 1, histones)
        proteinBoxPlot(ax[20 + comps], L1000GeneExp[:, comps], comps + 1, geneExpression)
        proteinBoxPlot(ax[25 + comps], RPPAproteins[:, comps], comps + 1, Rproteins)
    # subplotLabel(ax)
    return f


def R2X_OHSU(ax):
    '''Creates R2X for OHSU PARAFAC2'''
    r2x = []
    compR2X = 10
    for i in range(1, compR2X + 1):
        parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, i)
        r2x.append(R2Xparafac2(p2slicesB, parafac2tensor))
    df = pd.DataFrame(r2x)
    comps = []
    for i in range(1, compR2X + 1):
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
    ax.set_ylim(0, 1)


def OHSU_comp_plots(df, comps, ax):
    sns.lineplot(data=df, x="Times", y=str(comps), hue="Treatments", ax=ax)
    ax.set_xlabel('Time (hr)')
    ax.set_title('Component ' + str(comps))


def proteinBoxPlot(ax, transformed, componentIn, sliceRows):
    '''Plots protein component in partial tucker factorization space with annotation of some outliers'''
    df = pd.DataFrame(transformed)
    prots = outliersForPlot(df, sliceRows)
    sns.boxplot(data=df, ax=ax)
    ax.set_xlabel("Component " + str(componentIn))
    ax.set_ylabel('Component Value')
    ax.set_title('Protein Factors')
    for component in prots:
        offset_side = 0
        for outlier in prots[component]:
            if outlier[3]:
                if offset_side == 0:
                    ax.text(outlier[0] + .05, outlier[1] - .005, outlier[2], horizontalalignment='left', size='large', color='black', weight=100)
                    offset_side = 1
                elif offset_side == 1:
                    ax.text(outlier[0] - .05, outlier[1] - .005, outlier[2], horizontalalignment='right', size='large', color='black', weight=100)
                    offset_side = 0
            else:
                offset_side = 1 - offset_side


def outliersForPlot(transformed, sliceRows):
    '''Determines outliers based on IQR range by component and returns dictionary by component that allows annotation for OHSU PARAFAC2 analysis'''
    df = transformed.copy(deep=True)
    df['Proteins'] = sliceRows
    Q1 = df.quantile(.25)
    Q3 = df.quantile(.75)
    IQR = Q3 - Q1
    prots = {}
    for i in range(df.columns.size - 1):
        prots[i] = []
        for _, col in df.iterrows():
            if (col[i] < (Q1[i] - 1.7 * IQR[i])) or (col[i] > (Q3[i] + 1.7 * IQR[i])):
                tup = [i, col[i], col['Proteins'][:-1], True, True]
                prots[i].append(tup)
        prots[i].sort(key=lambda x: x[1])
        for idx, tup in enumerate(prots[i]):
            if idx < len(prots[i]) - 4:
                if tup[1] > prots[i][idx + 2][1] - .012 and tup[3] == tup[4]:
                    random1 = np.random.choice([0, 1, 1])
                    prots[i][idx + (random1 * 2)][3] = False
                    tup[4] = False
                    prots[i][idx + 2][4] = False
                elif tup[1] > prots[i][idx + 2][1] - .012 and tup[3]:
                    prots[i][idx + 2][3] = False
                    prots[i][idx + 2][4] = False
                if tup[1] > prots[i][idx + 4][1] - .012 and tup[3]:
                    random2 = np.random.randint(0, 2)
                    prots[i][idx + random2 * 4][3] = False
                    prots[i][idx + 4][4] = False
    return prots
