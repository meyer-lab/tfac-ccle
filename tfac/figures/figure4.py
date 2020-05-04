"""
This creates Figure 4 - All CP decomp protein component plots
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from .figure1 import treatmentPlot, timePlot, proteinPlot
from ..Data_Mod import form_tensor
from ..tensor import cp_decomp, reorient_factors

tensor, treatments, times = form_tensor()
results = cp_decomp(tensor, 8)
comps = reorient_factors(results[1])


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 4))
    
    for i in range (1,8):
        proteinPlot(ax[i-1], comps[2], i, i+1)
    
    # Add subplot labels
    subplotLabel(ax)
    return f
