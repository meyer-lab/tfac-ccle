"""
This creates Figure 2. Includes Treatments vs Time on Component 2 and box plots for each data slice on Component 2.
"""

import pandas as pd
from .figureCommon import getSetup, subplotLabel
from ..tensor import OHSU_parafac2_decomp, projections_to_factors
from ..Data_Mod import form_parafac2_tensor, ohsu_var, OHSU_comp_plots, proteinBoxPlot
p2slices, chromosomes, IFproteins, histones, geneExpression, RNAGenes, Rproteins = form_parafac2_tensor()
p2slicesB = ohsu_var(p2slices)
components = 5
parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, components)
weights, transform = projections_to_factors(parafac2tensor)
LINCSproteins = transform[1][0]
GCPHistones = transform[1][3]
L1000GeneExp = transform[1][4]
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
    row = 1
    col = 5
    ax, f = getSetup((24, 6), (row, col))
    OHSU_comp_plots(df, 2, ax[0])
    proteinBoxPlot(ax[1], LINCSproteins[:, 1], 2, proteins)
    proteinBoxPlot(ax[2], GCPHistones[:, 1], 2, histones)
    proteinBoxPlot(ax[3], L1000GeneExp[:, 1], 2, geneExpression)
    proteinBoxPlot(ax[4], RPPAproteins[:, 1], 2, Rproteins)
    subplotLabel(ax)
    return f
    