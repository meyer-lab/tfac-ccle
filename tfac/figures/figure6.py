"""
This creates Figure 6 - ROC Curve.
"""
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import LINCSCleanUp, dataCleanUp, form_parafac2_tensor, ohsu_var
from ..tensor import OHSU_parafac2_decomp, R2Xparafac2, projections_to_factors

p2slices, treatmentsTime, proteins, chromosomes, IFproteins, histones, geneExpression, RNAGenes, Rproteins = form_parafac2_tensor()
p2slicesB = ohsu_var(p2slices)
components = 5
parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, components)
weights, transform = projections_to_factors(parafac2tensor)
LINCSproteins = transform[1][0]
atacChr = transform[1][1]
GCPHistones = transform[1][3]
L1000GeneExp = transform[1][4]
RNAGeneSeq = transform[1][5]
RPPAproteins = transform[1][6]

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 7
    col = 5
    ax, f = getSetup((48, 56), (row, col))
    
    transformPlot(0, LINCSproteins, proteins)
    transformPlot(5, atacChr, chromosomes)
    transformPlot(10, GCPHistones, histones)
    transformPlot(15, L1000GeneExp, geneExpression)
    transformPlot(20, RNAGeneSeq, RNAGenes)
    transformPlot(25, RPPAproteins, Rproteins)

    # Add subplot labels
    subplotLabel(ax)

    return f


#R2X
#comp plots
def transformPlot(seed, transformSlice, sliceRow):
    '''Plots box plot along each component for each slice measurement label after projection to factor transformation'''
    col = np.size(transformSlice, 1)
    for a in range(col):
        proteinBoxPlot(ax[seed + a], transformSlice[:,a], a+1, sliceRow)

        
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


def outliersForPlot(transform, sliceRows):
    '''Determines outliers based on IQR range by component and returns dictionary by component that allows annotation for OHSU PARAFAC2 analysis'''
    df = transform.copy(deep=True)
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