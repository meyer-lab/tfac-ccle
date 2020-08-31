"""
This creates Figure 3. Includes Treatments vs Time on Component 3 and box plots for each data slice on Component 3.
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
GCPHistones = transform[1][2]
L1000GeneExp = transform[1][3]
RPPAproteins = transform[1][5]
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
    row = 2
    col = 3
    ax, f = getSetup((30, 6), (row, col))
    OHSU_comp_plots(df, 3, ax[0])
    proteinBoxPlot(ax[1], GCPHistones[:, 2], 3, histones)
    proteinBoxPlot(ax[2], L1000GeneExp[:, 2], 3, geneExpression)
    proteinBoxPlot(ax[3], RPPAproteins[:, 2], 3, Rproteins)
    subplotLabel(ax)
    return f
    