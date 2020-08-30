"""
This creates Figure 6. Includes OHSU R2X.
"""
import pandas as pd
from .figureCommon import getSetup, subplotLabel
from ..tensor import OHSU_parafac2_decomp, projections_to_factors
from ..Data_Mod import form_parafac2_tensor, ohsu_var, OHSU_comp_plots, proteinBoxPlot, R2X_OHSU
p2slices, treatmentsTime, proteins, chromosomes, IFproteins, histones, geneExpression, RNAGenes, Rproteins = form_parafac2_tensor()
p2slicesB = ohsu_var(p2slices)
components = 5
parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, components)
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
    col = 1
    ax, f = getSetup((10, 10), (row, col))
    R2X_OHSU(ax[0], p2slicesB)
    subplotLabel(ax)
    return f
    